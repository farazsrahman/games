"""
Unified RPS LLM game script for OpenAI (gpt-4o-mini) and Groq (llama-3.1-8b-instant).

Usage:
    # Requires GROQ and OPENAI keys
    #   export OPENAI_API_KEY=sk-...
    #   export GROQ_API_KEY=gsk_...
"""
import os
from tqdm import trange
# ---- Provider selection ----

from openai import OpenAI
oai_client = OpenAI()
OPT_MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-5.1")
MAX_TOKENS = int(os.environ.get("OPENAI_MAX_TOKENS", "16"))

from groq import Groq
groq_client = Groq()
AGENT_MODEL_NAME = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
# Reasoning-enabled Groq models often consume a large thinking budget before
# producing their final answer. Give them ample room so the content tokens
# are not truncated away.
MAX_TOKENS = int(os.environ.get("GROQ_MAX_TOKENS", "512"))

from games.game import Game

# ---- General LLM Game Prompts + Functions ----

AGENT_SYSTEM_PROMPT = """
You are about to play a two-player game against another large language model.

The GAME-PROMPT will provide instructions on how to play.
The STRATEGY-PROMPT will provide instructions on how you must play.
A third-party optimizer has determined the STRATEGY-PROMPT to be (approximately) optimal
against your current opponent. You must follow the STRATEGY-PROMPT as closely as possible.

OUTPUT RULES (CRITICAL):

- Each GAME-PROMPT will provide a very specific formatting according to the action space of the game. You MUST use this action space or you will LOSE.
- Do NOT include any other text, words, spaces, punctuation, or formatting.
- No explanations, no reasoning, no markdown, no preamble, no quotes.
""".strip()

def call_model(user_content: str, conversation_history: list = None) -> str:
    """
    Call the selected LLM (OpenAI or Groq) and return the raw move string.
    
    Args:
        user_content: The current user message
        conversation_history: List of previous messages in format [{"role": "user", "content": ...}, ...]
    """
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    
    # Add conversation history if provided
    if conversation_history:
        messages.extend(conversation_history)
    
    # Add current user message
    messages.append({"role": "user", "content": user_content})
    
    resp = groq_client.chat.completions.create(
        model=AGENT_MODEL_NAME,
        max_tokens=MAX_TOKENS,
        temperature=1.0,
        messages=messages,
    )
    return resp.choices[0].message.content.strip()

OPT_SYSTEM_PROMPT = """
You are about to recieve the transcript of a two player game which includes
a GAME-PROMPT, two STRATEGY-PROMPTs and a TRANSCRIPT in the different rounds of the game. 

The GAME-PROMPT provided the players instructions on how the game is to be played
Each agent will have a STRATEGY-PROMPT. The STRATEGY-PROMPT contains instructions 
that each agent MUST follow when playing the game. 

The TRANSCRIPT will provide information about how the players perform against eachother.
The format of the TRANSCRIPT will look like ('<player1-action>', '<player2-action>', 'payoff')
where positive payoff indicates player1 wins negative indicates player 2 wins and vice versa.

You are third-party optimizer for the first player and must reason through
the transcript of the game, indentify weaknessses in the player's actions 
and update the STRATEGY-PROMPT of player 1 to increase the probability 
of beating player 2.

To do this, first create a reasoning summary about the opposing player's actions.
Then devise how you can exploit this strategy to perform strictly better than them.
The strategy need not generalize to all agents, but it should perform better against this one.

OUTPUT RULES (CRITICAL):
- Use reasoning to carefully consider what the best new strategy_prompt is but do NOT include reasoning in the response
- Return a string that has the following format \"STRATEGY-PROMPT: <strategy-description-here>\"
""".strip()

def get_opt_prompt(u_prompt: str, v_prompt: str, transcript: str, game_prompt: str) -> str:
    opt_prompt = f"""
    {game_prompt}\n\n

    Player 1 {u_prompt}\n\n

    Player 2 {v_prompt}\n\n

    Transcript {transcript}\n\n
    """.strip()

    return opt_prompt

# ---- RPS Specific Prompts + Functions ----

def get_rps_prompt(n_games: int = None, inform_game_count: bool = False) -> str:
    """
    Generate the RPS game prompt, optionally including the number of games.
    
    Args:
        n_games: Total number of games in the match
        inform_game_count: Whether to inform the agent about the number of games
    """
    base_prompt = """
GAME-PROMPT:
You are an agent playing a game called rock-paper-scissors.

In this game there are two players and each must pick one of rock, paper, or scissors.
After both players choose their item, they are revealed and the result of the game is determined:

- If the same item is chosen, the result is a tie.
- If different items are chosen:
  - rock beats scissors
  - scissors beats paper
  - paper beats rock
""".strip()
    
    if inform_game_count and n_games is not None:
        base_prompt += f"\n\nThis match consists of {n_games} rounds. You will play {n_games} games against your opponent."
    
    base_prompt += """

OUTPUT RULES (CRITICAL):
- When it is time to choose your move, you MUST respond with exactly ONE character:
  R (rock), P (paper), or S (scissors).
- Your response MUST be a single capital letter: "R", "P", or "S".
- Do NOT include any other text, words, spaces, punctuation, or formatting.
- No explanations, no reasoning, no markdown, no preamble, no quotes.
- If you output anything other than exactly one of R / P / S, you immediately lose the game.

Once you have received this GAME-PROMPT and your STRATEGY-PROMPT, choose your move
according to the STRATEGY-PROMPT and reply with your move.
""".strip()
    
    return base_prompt

# Default RPS prompt for backward compatibility
rps_prompt = get_rps_prompt()

rock_prompt = """
STRATEGY-PROMPT: Always play rock.
""".strip()

paper_scissors_prompt = """
STRATEGY-PROMPT: Play paper or scissors with equal probability.
""".strip()


# ---- Game evaluation ----

def calculate_rps_payout(move_u: str, move_v: str) -> int:
    u = move_u.strip().upper()
    v = move_v.strip().upper()

    if u == v:
        return 0

    # Define what beats what
    beats = {
        "R": "S",
        "P": "R",
        "S": "P",
    }

    # -1 if move_u beats move_v, 1 if move_v beats move_u
    if beats.get(u) == v:
        return -1
    else:
        return 1

def format_transcript(transcript: list, round_num: int = None, is_player_u: bool = True) -> str:
    """
    Format the transcript of previous games for inclusion in the prompt.
    
    Args:
        transcript: List of tuples (move_u, move_v, payout) from previous rounds
        round_num: Current round number (1-indexed)
        is_player_u: True if formatting for player 1, False for player 2
    """
    if not transcript:
        return "No previous rounds have been played yet."
    
    lines = []
    if round_num is not None:
        lines.append(f"Previous rounds (you are about to play round {round_num}):")
    else:
        lines.append("Previous rounds:")
    
    for i, (move_u, move_v, payout) in enumerate(transcript, 1):
        if is_player_u:
            my_move = move_u
            opp_move = move_v
            # For player 1: payout < 0 means they won, payout > 0 means opponent won
            result = "tie" if payout == 0 else ("you won" if payout < 0 else "opponent won")
        else:
            my_move = move_v
            opp_move = move_u
            # For player 2: payout > 0 means they won, payout < 0 means opponent won
            result = "tie" if payout == 0 else ("you won" if payout > 0 else "opponent won")
        
        lines.append(f"Round {i}: You played {my_move}, opponent played {opp_move} - {result}")
    
    return "\n".join(lines)

def evaluate(u_prompt: str, v_prompt: str, game_prompt: str, 
             transcript_u: list = None, transcript_v: list = None,
             round_num: int = None):
    """
    Evaluate two agents by calling the chosen provider with their respective strategy prompts.
    Includes transcript of previous games if provided.
    
    Args:
        u_prompt: Strategy prompt for player 1
        v_prompt: Strategy prompt for player 2
        game_prompt: Game instructions prompt
        transcript_u: Previous game history for player 1 (list of (move_u, move_v, payout) tuples)
        transcript_v: Previous game history for player 2 (list of (move_u, move_v, payout) tuples)
        round_num: Current round number (1-indexed)
    
    Returns:
        Tuple of (move_u, move_v, payout)
    """
    # Build conversation history for player 1
    history_u = []
    if transcript_u:
        for i, (mu, mv, p) in enumerate(transcript_u, 1):
            history_u.append({
                "role": "user",
                "content": f"{game_prompt}\n\n{u_prompt}\n\n{format_transcript(transcript_u[:i], i, is_player_u=True)}"
            })
            history_u.append({
                "role": "assistant",
                "content": mu
            })
    
    # Build conversation history for player 2
    history_v = []
    if transcript_v:
        for i, (mu, mv, p) in enumerate(transcript_v, 1):
            history_v.append({
                "role": "user",
                "content": f"{game_prompt}\n\n{v_prompt}\n\n{format_transcript(transcript_v[:i], i, is_player_u=False)}"
            })
            history_v.append({
                "role": "assistant",
                "content": mv
            })
    
    # Current round prompts with transcript
    transcript_text_u = format_transcript(transcript_u, round_num, is_player_u=True) if transcript_u else ""
    transcript_text_v = format_transcript(transcript_v, round_num, is_player_u=False) if transcript_v else ""
    
    full_u = f"{game_prompt}\n\n{u_prompt}"
    if transcript_text_u:
        full_u += f"\n\n{transcript_text_u}"
    
    full_v = f"{game_prompt}\n\n{v_prompt}"
    if transcript_text_v:
        full_v += f"\n\n{transcript_text_v}"
    
    move_u = call_model(full_u, history_u)
    move_v = call_model(full_v, history_v)

    payout = calculate_rps_payout(move_u, move_v)

    return move_u, move_v, payout

def improve(u_prompt: str, v_prompt: str, n_games: int, game_prompt: str, 
            inform_game_count: bool = False):
    """
    Evaluate two agents in a game n_games times, then collect information and have a big reasoning LLM
    upgrade the prompt.
    
    Args:
        u_prompt: Strategy prompt for player 1
        v_prompt: Strategy prompt for player 2
        n_games: Number of games to play
        game_prompt: Game instructions prompt
        inform_game_count: Whether to inform agents about the number of games
    
    Returns:
        Tuple of (new_u_prompt, transcript, average_payout)
    """
    transcript = []
    for round_num in trange(1, n_games + 1):
        move_u, move_v, payout = evaluate(
            u_prompt, v_prompt, game_prompt,
            transcript_u=transcript, transcript_v=transcript,
            round_num=round_num
        )
        transcript.append((move_u, move_v, payout))

    opt_prompt=get_opt_prompt(u_prompt, v_prompt, transcript, game_prompt)

    resp = oai_client.chat.completions.create(
        model=OPT_MODEL_NAME,
        # max_tokens=,
        temperature=1.0,
        messages=[
            {"role": "system", "content": OPT_SYSTEM_PROMPT},
            {"role": "user", "content": opt_prompt},
        ],
    )
    return resp.choices[0].message.content.strip(), transcript, sum([t[2]for t in transcript]) / len(transcript)

class LLMRockPaperScissors(Game):

    def __init__(self, n_games: int = 5, inform_game_count: bool = False):
        """
        Initialize the multi-turn RPS game.
        
        Args:
            n_games: Number of games to play in each match (default: 5)
            inform_game_count: Whether to inform agents about the number of games (default: False)
        """
        self.n_games = n_games
        self.inform_game_count = inform_game_count
        self.game_prompt = get_rps_prompt(n_games, inform_game_count)

    def play(self, u, v):
        """
        Play a single match between two agents.
        Returns the average payout for player 1 across all rounds.
        """
        transcript = []
        total_payout = 0.0
        
        for round_num in range(1, self.n_games + 1):
            move_u, move_v, payout = evaluate(
                u, v, self.game_prompt,
                transcript_u=transcript, transcript_v=transcript,
                round_num=round_num
            )
            transcript.append((move_u, move_v, payout))
            total_payout += payout
        
        # Return average payout (positive means u wins on average)
        return total_payout / self.n_games

    def improve(self, u, v, *, n_games=None, inform_game_count=None):
        """
        Improve agent u against agent v.
        
        Args:
            u: Strategy prompt for player 1
            v: Strategy prompt for player 2
            n_games: Number of games to play (defaults to self.n_games)
            inform_game_count: Whether to inform agents about game count (defaults to self.inform_game_count)
        """
        n_games = n_games if n_games is not None else self.n_games
        inform_game_count = inform_game_count if inform_game_count is not None else self.inform_game_count
        game_prompt = get_rps_prompt(n_games, inform_game_count)
        
        print("U:", u, "\t, V:", v)
        u_new, _, _ = improve(u, v, n_games, game_prompt, inform_game_count)
        print("U_NEW:", u_new)
        return u_new


def empirical_rps_distribution(player_prompt: str, n_games: int = 100):
    """
    Evaluates a player n_games times using call_model and returns empirical distribution over R, P, S.
    """
    outcomes = {'R': 0, 'P': 0, 'S': 0}
    for _ in range(n_games):
        full_prompt = f"{rps_prompt}\n\n{player_prompt}"
        move = call_model(full_prompt)
        # move extraction: ensure it's a single char in "RPS"
        if isinstance(move, str) and move in outcomes:
            outcomes[move] += 1

    total = sum(outcomes.values())
    if total == 0:
        return {k: 0.0 for k in outcomes}
    return {k: v / total for k, v in outcomes.items()}

# ---- Main ----

if __name__ == "__main__":
    p1 = rock_prompt
    p2 = paper_scissors_prompt
    
    # Default to 5 games, with option to inform agents about game count
    n_games = 5
    inform_game_count = False  # Toggle: set to True to inform agents about number of games
    
    game_prompt = get_rps_prompt(n_games, inform_game_count)

    print(f"Player 1 \n{p1}")
    print(f"Player 2 \n{p2}")
    print(f"Playing {n_games} games (inform_game_count={inform_game_count})")
    p1, _, ev = improve(p1, p2, n_games, game_prompt, inform_game_count)
    print(f"Player 1 ev = {ev}")

    print(f"NEW Player 1 \n{p1}")
    p2, _, nev = improve(p2, p1, n_games, game_prompt, inform_game_count)
    print(f"Player 1 ev = {-nev}")
    print(f"NEW Player 2 \n{p2}")
