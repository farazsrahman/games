"""
Unified RPS LLM game script for OpenAI (gpt-4o-mini) and Groq (llama-3.1-8b-instant).

Usage:
    # Install deps:
    #   pip install openai groq

    # For OpenAI:
    #   export LLM_PROVIDER=openai
    #   export OPENAI_API_KEY=sk-...
    #
    # For Groq:
    #   export LLM_PROVIDER=groq
    #   export GROQ_API_KEY=gsk_...

    python rps_unified.py
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



# ---- Prompts ----

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

game_prompt = """
GAME-PROMPT:
You are an agent playing a game called rock-paper-scissors.

In this game there are two players and each must pick one of rock, paper, or scissors.
After both players choose their item, they are revealed and the result of the game is determined:

- If the same item is chosen, the result is a tie.
- If different items are chosen:
  - rock beats scissors
  - scissors beats paper
  - paper beats rock

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


# ---- Core call wrapper ----

def call_model(user_content: str) -> str:
    """
    Call the selected LLM (OpenAI or Groq) and return the raw move string.
    """
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

def evaluate(u_prompt: str, v_prompt: str):
    """
    Evaluate two agents by calling the chosen provider with their respective strategy prompts.
    Returns the two raw moves ('R', 'P', or 'S').
    """
    full_u = f"{game_prompt}\n\n{u_prompt}"
    full_v = f"{game_prompt}\n\n{v_prompt}"

    move_u = call_model(full_u)
    move_v = call_model(full_v)

    payout = calculate_rps_payout(move_u, move_v)

    return move_u, move_v, payout

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

def get_opt_prompt(u_prompt: str, v_prompt: str, transcript: str) -> str:
    opt_prompt = f"""
    {game_prompt}\n\n

    Player 1 {u_prompt}\n\n

    Player 2 {v_prompt}\n\n

    Transcript {transcript}\n\n
    """.strip()

    return opt_prompt


def improve(u_prompt: str, v_prompt: str, n_games: int):
    """
    Evaluate two agents in a game n_games times, then collect information and have a big reasoning LLM
    upgrate the prompt.
    """

    full_u = f"{game_prompt}\n\n{u_prompt}"
    full_v = f"{game_prompt}\n\n{v_prompt}"

    transcript = []
    for _ in trange(n_games):
        transcript.append(evaluate(u_prompt, v_prompt))

    opt_prompt=get_opt_prompt(u_prompt, v_prompt, transcript)

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

# ---- Main ----

if __name__ == "__main__":
    p1 = rock_prompt
    p2 = paper_scissors_prompt

    print(f"Player 1 \n{p1}")
    print(f"Player 2 \n{p2}")
    p1, _, ev = improve(p1, p2, 10)
    print(f"Player 1 ev = {ev}")


    print(f"NEW Player 1 \n{p1}")
    p2, _, nev = improve(p2, p1, 10)
    print(f"Player 1 ev = {-nev}")
    print(f"NEW Player 2 \n{p2}")
