"""
Quick test script to run 1 round of the LLM competition game.
This uses minimal tokens to test the game mechanics.
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from dotenv import load_dotenv
    project_root = Path(__file__).parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()
except ImportError:
    pass

from games.llms.llm_competition import (
    LLMCompetition,
    UserPreferences,
    evaluate
)

# Get API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY must be set")

print("=" * 80)
print("LLM COMPETITION GAME - TEST ROUND")
print("=" * 80)
print()

# Initialize game with fixed seed for reproducibility
game = LLMCompetition(seed=42)

# Display user preferences
print("📊 USER PREFERENCES (Hidden)")
print("-" * 80)
print(f"  Sociability:      {game.user_prefs.sociability:.2f}")
print(f"  Knowledge Depth:   {game.user_prefs.knowledge_depth:.2f}")
print(f"  Conciseness:       {game.user_prefs.conciseness:.2f}")
print(f"  Formality:         {game.user_prefs.formality:.2f}")
print(f"  Creativity:        {game.user_prefs.creativity:.2f}")
print(f"  Accuracy:          {game.user_prefs.accuracy:.2f}")
print(f"  Empathy:           {game.user_prefs.empathy:.2f}")
print()

# Get initial strategies
p1, p2 = game.get_default_strategies()

print("🎯 INITIAL STRATEGIES")
print("-" * 80)
print(f"\nPlayer 1 Strategy:\n{p1}")
print(f"\nPlayer 2 Strategy:\n{p2}")
print()

# Run 1 round
print("=" * 80)
print("🔄 RUNNING 1 ROUND")
print("=" * 80)
print()

# Pick a question
question = game.questions[0]  # Use first question
print(f"Question: {question}")
print()

# Evaluate
print("Getting answers from both players...")
answer_a, answer_b, payout = evaluate(
    p1, 
    p2, 
    question, 
    game.user_prefs,
    game_prompt="GAME-PROMPT:\nYou are an agent playing a game where you must answer questions.\nAnother large language model is also answering the same questions.\nA user will evaluate both answers and choose which one they prefer.\n\nYour goal is to provide answers that the user will prefer over your opponent's answers.\n\nWhen given a question, provide a helpful, accurate, and well-reasoned answer.\nFollow your STRATEGY-PROMPT to guide how you approach answering."
)

print()
print("=" * 80)
print("📝 RESULTS")
print("=" * 80)
print()

print(f"Player 1 Answer ({len(answer_a)} chars):")
print("-" * 80)
print(answer_a)
print()

print(f"Player 2 Answer ({len(answer_b)} chars):")
print("-" * 80)
print(answer_b)
print()

# Determine winner
winner = "Player 1" if payout > 0 else "Player 2" if payout < 0 else "TIE"
print(f"Winner: {winner} (Payout: {payout:+d})")
print()

print("=" * 80)
print("Test Complete!")
print("=" * 80)
