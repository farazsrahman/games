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
# AGENT_MODEL_NAME = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
AGENT_MODEL_NAME = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
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
- You may be playing a mult-round game, in this case ONLY ouput moves for a SINGLE round at at time
""".strip()

import time
from groq import APIStatusError

def call_model(user_content: str) -> str:
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    
    # Add current user message
    messages.append({"role": "user", "content": user_content})

    backoff = 3
    max_backoff = 96

    while True:
        try:
            resp = groq_client.chat.completions.create(
                model=AGENT_MODEL_NAME,
                max_tokens=MAX_TOKENS,
                temperature=1.0,
                messages=messages,
                service_tier="flex"
            )
            return resp.choices[0].message.content.strip()
        except APIStatusError as err:
            if getattr(err, "response", None) and hasattr(err.response, "json"):
                err_json = err.response.json()
                # Defensive: err_json may not be dict, but should be
                if isinstance(err_json, dict) and "error" in err_json:
                    err_msg = err_json["error"].get("message", "")
                    if "`flex` tier capacity exceeded" in err_msg:
                        if backoff > max_backoff:
                            raise
                        print(f"Retrying after {backoff} seconds due to flex tier capacity exceeded.")  # ADDED PRINT
                        time.sleep(backoff)
                        backoff *= 2
                        continue
            raise

OPT_SYSTEM_PROMPT = """
You are about to receive transcripts from multiple games between Player 1 and their opponent. Each transcript includes the following information:

- GAME-PROMPT: The instructions for how the game is played.
- STRATEGY-PROMPT: The description that specifies how Player 1 is instructed to play.
- TRANSCRIPTS: You will receive transcripts from multiple games. Each game may contain a variable number of rounds. Each round in a game is played against the same opponent. Different games may be played against different opponents.

You will NOT receive any information about Player 2's strategy prompt or instructions.

The TRANSCRIPTS will provide information about how Player 1 performs against their opponent. Each round within a game is formatted as "Round X: You played <action>, opponent played <action> - <result>", where the result indicates whether you (Player 1) won, the opponent won, or it was a tie.

Your job as a third-party optimizer is to improve Player 1's STRATEGY-PROMPT by examining the transcripts from all games. Identify Player 1's weaknesses and suggest an updated STRATEGY-PROMPT that increases their chance of winning against the opponents seen in these games.

First, consider how Player 1 performed across all the games; think carefully about how you would update Player 1's strategy without changing it too much to increase their played value in the above game. Do not think generally about all possible opponents; only consider the specific rounds present in the transcripts.

OUTPUT RULES (CRITICAL):
- Use reasoning to carefully determine the best possible new strategy prompt, but DO NOT include any reasoning or summary in your response.
- Return ONLY a string of the following format: "STRATEGY-PROMPT: <strategy-description-here>"
""".strip()

def transform_transcript_for_agent(transcript: list, is_player_1: bool) -> list:
    """
    Transform transcript so agent always sees themselves as player 1.
    
    Args:
        transcript: List of tuples (action_1, action_2, value) from player 1's perspective
        is_player_1: True if this agent is player 1, False if player 2
    
    Returns:
        List of tuples (my_action, opp_action, value) where value > 0 means this agent wins
    """
    if is_player_1:
        # No transformation needed - already from player 1's perspective
        return transcript
    else:
        # Transform: swap actions and negate value
        return [(action_2, action_1, -value) for action_1, action_2, value in transcript]

def get_opt_prompt(u_prompt: str, transcripts: list, game_prompt: str) -> str:
    """
    Format the optimization prompt for improving player 1's strategy.
    
    Args:
        u_prompt: Strategy prompt for player 1
        transcripts: List of game transcripts, where each game transcript is a list of (action_1, action_2, value) tuples
                    from player 1's perspective
        game_prompt: Game instructions prompt
    
    Returns:
        Formatted prompt string for the optimizer
    """
    n_games = len(transcripts)
    opt_prompt = f"{game_prompt}\n\nPlayer 1 {u_prompt}\n\nTRANSCRIPTS\n\n"
    opt_prompt += f"You are receiving transcripts from {n_games} game(s). Each game may contain a variable number of rounds.\n\n"
    for idx, game_transcript in enumerate(transcripts):
        transcript_formatted = format_transcript(game_transcript)
        opt_prompt += f"GAME {idx + 1}:\nTranscript from all rounds:\n{transcript_formatted}\n\n"
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
        base_prompt += f"\n\nThis game consists of {n_games} rounds."
    
    base_prompt += """
    OUTPUT RULES (CRITICAL):
    - When it is time to choose your move, you MUST respond with exactly ONE character:
    R (rock), P (paper), or S (scissors).
    - Your response MUST be a single capital letter: "R", "P", or "S".
    - You are being asked to make ONE move for THIS round only. Do NOT output multiple moves or a sequence of moves.
    - Do NOT include any other text, words, spaces, punctuation, or formatting.
    - No explanations, no reasoning, no markdown, no preamble, no quotes.
    - If you output anything other than exactly one of R / P / S, you immediately lose the game.

    Once you have received this GAME-PROMPT and your STRATEGY-PROMPT, choose your move
    according to the STRATEGY-PROMPT and reply with your move.
    """.strip()
    
    return base_prompt

# Default RPS prompt for backward compatibility
rps_prompt = get_rps_prompt()

alternating_paper_scissors_prompt = """
STRATEGY-PROMPT: Alternate deterministically: first play paper, then scissors, then paper, then scissors, and so on.
""".strip()

random_prompt = """
STRATEGY-PROMPT: Choose randomly and uniformly between rock, paper, and scissors each round.
""".strip()

alternating_scissors_rock_prompt = """
STRATEGY-PROMPT: Alternate deterministically: first play scissors, then rock, then scissors, then rock, and so on.
""".strip()

alternating_rock_scissors_prompt = """
STRATEGY-PROMPT: Alternate deterministically: first play rock, then scissors, then rock, then scissors, and so on.
""".strip()

example_population = [alternating_paper_scissors_prompt, random_prompt, alternating_scissors_rock_prompt]

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

    # 1 if move_u beats move_v (Player 1 wins), -1 if move_v beats move_u (Player 2 wins)
    if beats.get(u) == v:
        return 1
    else:
        return -1

def format_transcript(transcript: list) -> str:
    """
    Format the transcript for inclusion in the prompt.
    Transcript should be in format (my_action, opp_action, value) where value > 0 means "my" agent wins.
    Uses first-person format ("You played...") since the agent/optimizer always sees themselves as player 1.
    
    Args:
        transcript: List of tuples (my_action, opp_action, value) where the agent is always player 1
    
    Returns:
        Formatted string with round-by-round results, without any header
    """
    if not transcript:
        return ""
    
    lines = []
    for i, (my_action, opp_action, value) in enumerate(transcript, 1):
        result = "tie" if value == 0 else ("you won" if value > 0 else "opponent won")
        lines.append(f"Round {i}: You played {my_action}, opponent played {opp_action} - {result}")
    
    return "\n".join(lines)

def build_agent_prompt(game_prompt: str, strategy_prompt: str, transcript: list, round_num: int) -> str:
    """
    Build the full prompt for an agent including game prompt, strategy, and transcript.
    
    Args:
        game_prompt: Game instructions prompt
        strategy_prompt: Strategy prompt for the agent
        transcript: List of tuples (my_action, opp_action, value) from agent's perspective
        round_num: Current round number (1-indexed)
    
    Returns:
        Complete prompt string for the agent
    """
    prompt = f"{game_prompt}\n\n{strategy_prompt}"
    if transcript:
        transcript_formatted = format_transcript(transcript)
        prompt += f"\n\nPrevious rounds (you are about to play round {round_num}):\n{transcript_formatted}"
    return prompt

def evaluate(u_prompt: str, v_prompt: str, game_prompt: str, 
             transcript_u: list = None, transcript_v: list = None,
             round_num: int = None):
    """
    Evaluate two agents by calling the chosen provider with their respective strategy prompts.
    Includes transcript of previous rounds if provided.
    
    Args:
        u_prompt: Strategy prompt for player 1
        v_prompt: Strategy prompt for player 2
        game_prompt: Game instructions prompt
        transcript_u: Previous round history from player 1's perspective (list of (action_1, action_2, value) tuples)
        transcript_v: Previous round history from player 1's perspective (will be transformed for player 2)
        round_num: Current round number (1-indexed)
    
    Returns:
        Tuple of (move_u, move_v, payout) where payout > 0 means player 1 wins
    """
    # Transform transcripts so each agent sees themselves as player 1
    transcript_u_transformed = transform_transcript_for_agent(transcript_u, is_player_1=True) if transcript_u else None
    transcript_v_transformed = transform_transcript_for_agent(transcript_v, is_player_1=False) if transcript_v else None
    
    # Build prompts (transcript is already included in the prompt)
    full_u = build_agent_prompt(game_prompt, u_prompt, transcript_u_transformed, round_num)
    full_v = build_agent_prompt(game_prompt, v_prompt, transcript_v_transformed, round_num)
    
    move_u = call_model(full_u)
    move_v = call_model(full_v)
    max_invalid_attempts = 5
    attempt = 0
    while move_u not in ['R', 'P', 'S'] or move_v not in ['R', 'P', 'S']:
        attempt += 1
        if attempt > max_invalid_attempts:
            raise ValueError(f"Invalid move(s) after {max_invalid_attempts} attempts. move_u={repr(move_u)}, move_v={repr(move_v)}")
        if move_u not in ['R', 'P', 'S']:
            print(f"Invalid move_u: {repr(move_u)}, re-querying...")
            move_u = call_model(full_u)
        if move_v not in ['R', 'P', 'S']:
            print(f"Invalid move_v: {repr(move_v)}, re-querying...")
            move_v = call_model(full_v)
        time.sleep(2 ** attempt)

    return move_u, move_v, calculate_rps_payout(move_u, move_v)

def improve(u_prompt: str, transcripts: List[list], game_prompt: str):
    """
    Improve player 1's strategy based on accumulated transcripts.
    
    Args:
        u_prompt: Strategy prompt for player 1
        transcripts: List of game transcripts, where each game transcript is a list of (action_1, action_2, value) tuples
                    from player 1's perspective (value > 0 means player 1 wins)
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
            n_games: Number of rounds to play in each game (default: 5)
            inform_game_count: Whether to inform agents about the number of rounds (default: False)
        """
        self.n_games = n_games
        self.inform_game_count = inform_game_count
        self.game_prompt = get_rps_prompt(n_games, inform_game_count)

    def play(self, u, v, *, return_transcript=False):
        """
        Play a single game between two agents.
        
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

    def improve(self, u, v, **kwargs):
        """
        Improve agent u against agent v by playing a single game.
        """
        print("U:", u, "\t, V:", v)
        
        # Play a single game (multiple rounds) and collect transcript
        game_transcript = []
        for round_num in trange(1, self.n_games + 1):
            move_u, move_v, payout = evaluate(
                u, v, self.game_prompt,
                transcript_u=game_transcript, transcript_v=game_transcript,
                round_num=round_num
            )
            game_transcript.append((move_u, move_v, payout))
        
        # Print transcript for debugging
        print("Game Transcript:")
        for idx, (mu, mv, p) in enumerate(game_transcript, 1):
            print(f"Round {idx}: U: {mu}, V: {mv}, Payout: {p}")
        
        # Convert single game to list format expected by improve()
        transcripts = [game_transcript]
        u_new = improve(u, transcripts, self.game_prompt)
        print("U_NEW:", u_new)
        return u_new

    def improve_from_transcripts(self, u, transcripts):
        """
        Improve agent u based on accumulated transcripts from multiple games.
        
        Args:
            u: Strategy prompt for player 1
            transcripts: List of game transcripts, where each game transcript is a list of (move_u, move_v, payout) tuples
        
        Returns:
            Updated strategy prompt string
        """
        print("U:", u, "\t, TRANSCRIPTS:", len(transcripts), "games")
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
    from games.llms.multiturn_rps import LLMRockPaperScissors

    initial_agent = alternating_rock_scissors_prompt
    n_rounds = 8

    rps_game = LLMRockPaperScissors(n_games=n_rounds)

    # Step 0: Play games with the initial agent
    transcripts_0 = [rps_game.play(initial_agent, initial_agent, return_transcript=True) for _ in range(4)]
    payout_0 = sum(sum(p for *_, p in tr) / len(tr) for tr in transcripts_0) / len(transcripts_0)

    print(f"=== Step 0 (Initial agent) ===")
    print(f"Avg payout before: {payout_0:.3f}")

    # Step 1: First optimization
    agent_1 = rps_game.improve_from_transcripts(initial_agent, transcripts_0)
    transcript_1 = rps_game.play(agent_1, initial_agent, return_transcript=True)
    payout_1 = sum(p for *_, p in transcript_1) / len(transcript_1)
    print(f"\n=== Step 1 (First optimization) ===")
    print(f"Avg payout after step 1: {payout_1:.3f}")
    print("Transcript after first optimization:")
    print(format_transcript(transcript_1))
    print(f"Δ Improvement (step 1 vs initial): {payout_1 - payout_0:+.3f}")

    # Step 2: Second optimization
    transcripts_1 = [rps_game.play(agent_1, initial_agent, return_transcript=True) for _ in range(4)]
    agent_2 = rps_game.improve_from_transcripts(agent_1, transcripts_1)
    transcript_2 = rps_game.play(agent_2, initial_agent, return_transcript=True)
    payout_2 = sum(p for *_, p in transcript_2) / len(transcript_2)
    print(f"\n=== Step 2 (Second optimization) ===")
    print(f"Avg payout after step 2: {payout_2:.3f}")
    print("Transcript after second optimization:")
    print(format_transcript(transcript_2))
    print(f"Δ Improvement (step 2 vs step 1): {payout_2 - payout_1:+.3f}")
    print(f"Δ Improvement (step 2 vs initial): {payout_2 - payout_0:+.3f}")
