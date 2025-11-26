"""
Unified RPS LLM game script for OpenAI (gpt-4o-mini) and Groq (llama-3.1-8b-instant).

Usage:
    # Requires GROQ and OPENAI keys
    #   export OPENAI_API_KEY=sk-...
    #   export GROQ_API_KEY=gsk_...
"""
import os
from tqdm import trange
from typing import List
# ---- Provider selection ----

from openai import OpenAI
oai_client = OpenAI()
OPT_MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-5.1")
MAX_OPT_TOKENS = int(os.environ.get("MAX_OPT_TOKENS", "512"))

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
        service_tier="flex"
    )
    # print("Service tier used:", resp.service_tier)
    return resp.choices[0].message.content.strip()

OPT_SYSTEM_PROMPT = """
You are about to receive a transcript of a two-player game which includes the following information:

- GAME-PROMPT: The instructions for how the game is played.
print("Service tier used:", resp.service_tier)- STRATEGY-PROMPT: The description that specifies how Player 1 is instructed to play.
- TRANSCRIPTS: The outcomes of one or more rounds between Player 1 and their opponent.

You will NOT receive any information about Player 2's strategy prompt or instructions.

The TRANSCRIPTS will provide information about how Player 1 performs against their opponent. The format of each transcript entry is ('<player1-action>', '<player2-action>', 'payoff'), where a positive payoff means Player 1 wins, and a negative payoff means Player 2 wins.

Your job as a third-party optimizer is to improve Player 1's STRATEGY-PROMPT by examining the transcript. Identify Player 1's weaknesses and suggest an updated STRATEGY-PROMPT that increases their chance of winning against the opponents seen in the transcript rounds.

First, consider how Player 1 performed in the games; think carefully about how you would update Player 1's strategy without changing it too much—try to preserve strengths that are not challenged in the transcript. Do not think generally about all possible opponents; only consider the specific rounds present in the transcripts.

OUTPUT RULES (CRITICAL):
- Use reasoning to carefully determine the best possible new strategy prompt, but DO NOT include any reasoning or summary in your response.
- Return ONLY a string of the following format: "STRATEGY-PROMPT: <strategy-description-here>"
""".strip()

def format_match_transcript(match_transcript: list) -> str:
    """
    Format a single match transcript (list of round tuples) into a string.
    
    Args:
        match_transcript: List of tuples (move_u, move_v, payout) from a single match
    
    Returns:
        Formatted string representation of the match
    """
    if not match_transcript:
        return "No rounds played."
    
    lines = []
    for round_idx, (move_u, move_v, payout) in enumerate(match_transcript, 1):
        lines.append(f"Round {round_idx}: ({move_u}, {move_v}, {payout})")
    
    return "\n".join(lines)

def get_opt_prompt(u_prompt: str, transcripts: list, game_prompt: str) -> str:
    """
    Format the optimization prompt for improving player 1's strategy.
    
    Args:
        u_prompt: Strategy prompt for player 1
        transcripts: List of match transcripts, where each match transcript is a list of (move_u, move_v, payout) tuples
        game_prompt: Game instructions prompt
    
    Returns:
        Formatted prompt string for the optimizer
    """
    opt_prompt = f"{game_prompt}\n\nPlayer 1 {u_prompt}\n\nTRANSCRIPTS\n\n"
    for idx, match_transcript in enumerate(transcripts):
        opt_prompt += f"GAME {idx}:\n{format_match_transcript(match_transcript)}\n\n"
    # print(opt_prompt)
    return opt_prompt

# ---- RPS Specific Prompts + Functions ----

def get_rps_prompt(n_games: int = None, inform_game_count: bool = False) -> str:
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

alternating_paper_scissors_prompt = """
STRATEGY-PROMPT: Alternate deterministically: first play paper, then scissors, then paper, then scissors, and so on.
""".strip()

random_prompt = """
STRATEGY-PROMPT: Choose randomly and uniformly between rock, paper, and scissors each round.
""".strip()

alternating_scissors_rock_prompt = """
STRATEGY-PROMPT: Alternate deterministically: first play scissors, then rock, then scissors, then rock, and so on.
""".strip()


example_population = [rock_prompt, alternating_paper_scissors_prompt, random_prompt, alternating_scissors_rock_prompt]

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

def improve(u_prompt: str, transcripts: List[list], game_prompt: str):
    """
    Improve player 1's strategy based on accumulated transcripts.
    
    Args:
        u_prompt: Strategy prompt for player 1
        transcripts: List of match transcripts, where each match transcript is a list of (move_u, move_v, payout) tuples
        game_prompt: Game instructions prompt
    
    Returns:
        Updated strategy prompt string
    """
    opt_prompt = get_opt_prompt(u_prompt, transcripts, game_prompt)

    resp = oai_client.chat.completions.create(
        model=OPT_MODEL_NAME,
        max_completion_tokens=MAX_OPT_TOKENS,
        temperature=1.0,
        messages=[
            {"role": "system", "content": OPT_SYSTEM_PROMPT},
            {"role": "user", "content": opt_prompt},
        ],
    )
    return resp.choices[0].message.content.strip()

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

    def play(self, u, v, *, return_transcript=False):
        """
        Play a single match between two agents.
        
        Args:
            u: Strategy prompt for player 1
            v: Strategy prompt for player 2
            return_transcript: If True, returns the full transcript list. If False, returns average payout.
        
        Returns:
            If return_transcript=True: List of (move_u, move_v, payout) tuples for all rounds
            If return_transcript=False: Average payout for player 1 across all rounds (positive means u wins on average)
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
        
        if return_transcript:
            return transcript
        else:
            # Return average payout (positive means u wins on average)
            return total_payout / self.n_games

    def improve(self, u, v, *, n_games=None, inform_game_count=None):
        """
        Improve agent u against agent v by playing a single match.
        """
        n_games = n_games if n_games is not None else self.n_games
        inform_game_count = inform_game_count if inform_game_count is not None else self.inform_game_count
        game_prompt = get_rps_prompt(n_games, inform_game_count)
        
        print("U:", u, "\t, V:", v)
        
        # Play a single match (multiple rounds) and collect transcript
        match_transcript = []
        for round_num in trange(1, n_games + 1):
            move_u, move_v, payout = evaluate(
                u, v, game_prompt,
                transcript_u=match_transcript, transcript_v=match_transcript,
                round_num=round_num
            )
            match_transcript.append((move_u, move_v, payout))
        
        # Print transcript for debugging
        print("Game Transcript:")
        for idx, (mu, mv, p) in enumerate(match_transcript, 1):
            print(f"Round {idx}: U: {mu}, V: {mv}, Payout: {p}")
        
        # Convert single match to list format expected by improve()
        transcripts = [match_transcript]
        u_new = improve(u, transcripts, game_prompt)
        print("U_NEW:", u_new)
        return u_new

    def improve_from_transcripts(self, u, transcripts):
        """
        Improve agent u based on accumulated transcripts from multiple matches.
        
        Args:
            u: Strategy prompt for player 1
            transcripts: List of match transcripts, where each match transcript is a list of (move_u, move_v, payout) tuples
        
        Returns:
            Updated strategy prompt string
        """
        print("U:", u, "\t, TRANSCRIPTS:", len(transcripts), "matches")
        u_new = improve(u, transcripts, self.game_prompt)
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
    # Instantiate LLMRockPaperScissors and use its methods
    from games.llms.multiturn_rps import LLMRockPaperScissors

    # Setup prompts and game parameters
    p1 = rock_prompt
    p2 = paper_scissors_prompt

    n_games = 5
    inform_game_count = False  # Toggle: set to True to inform agents about number of games

    rps_game = LLMRockPaperScissors(n_games=n_games, inform_game_count=inform_game_count)

    print(f"Playing {n_games} games (inform_game_count={inform_game_count})")

    p1_new = rps_game.improve(p1, p2)
    
    print(f"\n\n")

    p2_new = rps_game.improve(p2, p1_new)
