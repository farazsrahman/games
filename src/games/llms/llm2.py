"""
Unified RPS LLM game script for OpenAI (gpt-4o-mini) and Groq (llama-3.1-8b-instant).

Usage:
    # Requires GROQ and OPENAI keys
    #   export OPENAI_API_KEY=sk-...
    #   export GROQ_API_KEY=gsk_...
"""
import os
from tqdm import trange
from typing import List, Optional, Tuple
# ---- Provider selection ----

from openai import OpenAI
oai_client = OpenAI()
OPT_MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-5.1")
MAX_OPT_TOKENS = int(os.environ.get("MAX_OPT_TOKENS", "512") )

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

def call_model(user_content: str, *, max_attempts: int = 5, base_delay: float = 1.0, max_delay: float = 30.0) -> str:
    """
    Call the selected LLM (OpenAI or Groq) and return the raw move string.
    Retries up to max_attempts times on failure, with exponential backoff.
    """
    import time

    last_exception = None
    for attempt in range(max_attempts):
        try:
            resp = groq_client.chat.completions.create(
                model=AGENT_MODEL_NAME,
                max_tokens=MAX_TOKENS,
                temperature=1.0,
                messages=[
                    {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_exception = e
            print(f"[call_model retry]: Attempt {attempt + 1} of {max_attempts} failed with error: {e}")
            if attempt < max_attempts - 1:
                delay = min(base_delay * (2 ** attempt), max_delay)
                print(f"Waiting {delay:.2f} seconds before retrying...")
                time.sleep(delay)
            else:
                print(f"ERROR in call_model: All {max_attempts} attempts failed. Last error: {e}")
                raise e
    raise last_exception

OPT_SYSTEM_PROMPT = """
You are about to receive a transcript of a two-player game which includes the following information:

- GAME-PROMPT: The instructions for how the game is played.
- STRATEGY-PROMPT: The description that specifies how Player 1 is instructed to play.
- TRANSCRIPTS: The outcomes of one or more rounds between Player 1 and their opponent.

You will NOT receive any information about Player 2's strategy prompt or instructions.

The TRANSCRIPTS will provide information about how Player 1 performs against their opponent. The format of each transcript entry is ('<player1-action>', '<player2-action>', 'payoff'), where a positive payoff means Player 1 wins, and a negative payoff means Player 2 wins.

Your job as a third-party optimizer is to improve Player 1's STRATEGY-PROMPT by examining the transcript. Identify Player 1's weaknesses and suggest an updated STRATEGY-PROMPT that increases their chance of winning against the opponents seen in the transcript rounds.

First, consider how Player 1 performed in the games; think carefully about how you would update Player 1's strategy without changing it too much—try to preserve strengths that are not challenged in the transcript. Do not think generally about all possible opponents; only consider the specific rounds present in the transcripts.

OUTPUT RULES (CRITICAL):
- Use reasoning to carefully determine the best possible new strategy prompt, but DO NOT include any reasoning or summary in your response.
- Return ONLY a string of the following format: "STRATEGY-PROMPT: <strategy-description-here>"
""".strip()

def get_opt_prompt(u_prompt: str, transcripts: str, game_prompt: str) -> str:
    opt_prompt = f"{game_prompt}\n\nPlayer 1 {u_prompt}\n\nTRANSCRIPTS\n\n"
    for idx in range(len(transcripts)):
        opt_prompt += f"GAME {idx}:\n{transcripts[idx]}\n\n"
    print(opt_prompt)
    return opt_prompt

# ---- RPS Specific Prompts + Functions ----

rps_prompt = """
GAME-PROMPT:
You are an agent playing a game called rock-paper-scissors.

In this game there are two players and each must pick one of rock, paper, or scissors.
After both players choose their item, they are revealed and the result of the game is determined:

- If the same item is chosen, the result is a tie.
- If different items are chosen:
  - rock beats scissors
  - scissors beats paper
  - paper beats rock

STRATEGY PROMPT RULES:
- Strategies do not see any previous game transcripts so they should not refer to previous transcripts. 

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

rock_prompt = """
STRATEGY-PROMPT: Always play rock.
""".strip()

paper_scissors_prompt = """
STRATEGY-PROMPT: Play paper or scissors with equal probability.
""".strip()

random_prompt = """
STRATEGY-PROMPT: Choose randomly and uniformly between rock, paper, and scissors each round.
""".strip()

scissors_rock_prompt = """
STRATEGY-PROMPT: Play scissors or rock with equal probability.
""".strip()

example_population = [rock_prompt, paper_scissors_prompt, random_prompt, scissors_rock_prompt]

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

def evaluate(u_prompt: str, v_prompt: str, game_prompt: str):
    """
    Evaluate two agents by calling the chosen provider with their respective strategy prompts.
    Returns the two raw moves as strings.
    """
    full_u = f"{game_prompt}\n\n{u_prompt}"
    full_v = f"{game_prompt}\n\n{v_prompt}"

    move_u = call_model(full_u)
    move_v = call_model(full_v)

    payout = calculate_rps_payout(move_u, move_v)

    return move_u, move_v, payout

def improve(u_prompt: str, transcripts: List[str], game_prompt: str):
    """
    Evaluate two agents in a game n_games times, then collect information and have a big reasoning LLM
    upgrate the prompt.
    """

    opt_prompt=get_opt_prompt(u_prompt, transcripts, game_prompt)

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

    def play(self, u, v, *, return_transcript=False):
        """
        returns transcript if return_transcript=True else just returns value of game
        """
        transcript = evaluate(u, v, rps_prompt)        
        return transcript if return_transcript else transcript[2]

    def improve(self, u, v, *, n_games=10):
        print("U:", u, "\t, V:", v)

        transcripts = []
        for _ in trange(n_games):
            transcripts.append(evaluate(u, v, rps_prompt))

        u_new = improve(u, transcripts, rps_prompt)
        print("U_NEW:", u_new)

        return u_new

    def improve_from_transcripts(self, u, transcripts):
        print("U:", u, "\t, TRANSCRIPTS:", transcripts)
        u_new = improve(u, transcripts, rps_prompt)
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

    print(f"Player 1 \n{p1}")
    print(f"Player 2 \n{p2}")
    p1, _, ev = improve(p1, p2, 10, rps_prompt)
    print(f"Player 1 ev = {ev}")


    print(f"NEW Player 1 \n{p1}")
    p2, _, nev = improve(p2, p1, 10, rps_prompt)
    print(f"Player 1 ev = {-nev}")
    print(f"NEW Player 2 \n{p2}")
