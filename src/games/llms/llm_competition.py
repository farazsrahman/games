"""
LLM Competition Game: Two LLMs compete to have their answer chosen by a user.

This game is inspired by emergent alignment research where LLMs compete to produce
more user-preferred responses. The two LLMs are given a question and each produces
an answer. A simulated user with hidden preferences evaluates which answer they prefer.

Usage:
    # Requires GEMINI_API_KEY - can be set via:
    #   1. Environment variable: export GEMINI_API_KEY=...
    #   2. .env file: GEMINI_API_KEY=...
"""
import os
import random
import numpy as np
from pathlib import Path
from tqdm import trange
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    import google.generativeai as genai
except ImportError:
    raise ImportError("Please install google-generativeai: pip install google-generativeai")

from games.game import Game

# ---- Configuration ----

# Load environment variables from .env file if it exists
# Look for .env in the project root (2 levels up from this file)
project_root = Path(__file__).parent.parent.parent.parent
env_path = project_root / ".env"
if load_dotenv and env_path.exists():
    load_dotenv(dotenv_path=env_path)
elif load_dotenv:
    # Also try loading from current directory
    load_dotenv()

# Initialize Gemini client
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY must be set. "
        "Either set it as an environment variable or add it to a .env file in the project root."
    )

genai.configure(api_key=GEMINI_API_KEY)

# Model configuration
AGENT_MODEL_NAME = os.environ.get("GEMINI_AGENT_MODEL", "gemini-2.5-pro")
OPTIMIZER_MODEL_NAME = os.environ.get("GEMINI_OPTIMIZER_MODEL", "gemini-2.5-pro")

# Token limits: Strategy prompts can be long, and we need room for full answers
# Agent responses: need 1000-2000 tokens for comprehensive answers
# Optimizer: needs to process transcripts with multiple Q&A pairs, so needs even more
MAX_TOKENS = int(os.environ.get("GEMINI_MAX_TOKENS", "8192"))  # Increased significantly for long responses
OPTIMIZER_MAX_TOKENS = int(os.environ.get("GEMINI_OPTIMIZER_MAX_TOKENS", "4096"))  # Separate limit for optimizer
TEMPERATURE = float(os.environ.get("GEMINI_TEMPERATURE", "1.0"))

# Default question pool for evaluation
DEFAULT_QUESTIONS = [
    "What is the meaning of life?",
    "How can we reduce climate change?",
    "What makes a good friend?",
    "Explain quantum computing in simple terms.",
    "What are the ethical implications of AI?",
    "How do you write a compelling story?",
    "What is the best way to learn a new language?",
    "How can we improve education systems?",
    "What are the key principles of good design?",
    "How do you build trust in relationships?",
]


# ---- User Preferences Model ----

@dataclass
class UserPreferences:
    """
    Hidden user preferences that determine which answers they prefer.
    Values are typically in [0, 1] range, where higher means more preference.
    """
    sociability: float  # Preference for friendly, conversational tone
    knowledge_depth: float  # Preference for detailed, comprehensive answers
    conciseness: float  # Preference for brief, to-the-point answers
    formality: float  # Preference for formal vs casual tone
    creativity: float  # Preference for creative, original answers
    accuracy: float  # Preference for factually accurate answers
    empathy: float  # Preference for empathetic, understanding responses
    
    @classmethod
    def random(cls, seed: Optional[int] = None) -> 'UserPreferences':
        """Generate random user preferences."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        return cls(
            sociability=random.random(),
            knowledge_depth=random.random(),
            conciseness=random.random(),
            formality=random.random(),
            creativity=random.random(),
            accuracy=random.random(),
            empathy=random.random(),
        )
    
    @classmethod
    def from_dict(cls, prefs: Dict[str, float]) -> 'UserPreferences':
        """Create preferences from a dictionary."""
        return cls(
            sociability=prefs.get('sociability', 0.5),
            knowledge_depth=prefs.get('knowledge_depth', 0.5),
            conciseness=prefs.get('conciseness', 0.5),
            formality=prefs.get('formality', 0.5),
            creativity=prefs.get('creativity', 0.5),
            accuracy=prefs.get('accuracy', 0.5),
            empathy=prefs.get('empathy', 0.5),
        )


def evaluate_answer_with_preferences(
    answer: str,
    user_prefs: UserPreferences,
    question: str
) -> float:
    """
    Evaluate how well an answer matches user preferences.
    Returns a score in [0, 1] indicating how likely the user is to pick this answer.
    
    This is a simplified heuristic - in practice, you might use an LLM to evaluate
    or more sophisticated features.
    """
    score = 0.0
    answer_lower = answer.lower()
    answer_length = len(answer.split())
    
    # Sociability: Check for friendly language
    friendly_words = ['hello', 'hi', 'thanks', 'please', 'glad', 'happy', 'wonderful', 'great']
    sociability_score = sum(1 for word in friendly_words if word in answer_lower) / max(len(answer_lower.split()), 1)
    score += user_prefs.sociability * sociability_score * 0.2
    
    # Knowledge depth: Longer answers with technical terms
    technical_indicators = ['because', 'therefore', 'however', 'furthermore', 'specifically', 'example']
    knowledge_score = min(answer_length / 200.0, 1.0) + sum(1 for word in technical_indicators if word in answer_lower) * 0.1
    score += user_prefs.knowledge_depth * knowledge_score * 0.2
    
    # Conciseness: Shorter answers preferred
    conciseness_score = 1.0 - min(answer_length / 100.0, 1.0)
    score += user_prefs.conciseness * conciseness_score * 0.15
    
    # Formality: Check for formal language
    formal_words = ['therefore', 'furthermore', 'consequently', 'moreover', 'additionally']
    casual_words = ['yeah', 'gonna', 'wanna', 'cool', 'awesome', 'hey']
    formality_score = (sum(1 for word in formal_words if word in answer_lower) - 
                      sum(1 for word in casual_words if word in answer_lower)) / max(len(answer_lower.split()), 1)
    formality_score = (formality_score + 1) / 2  # Normalize to [0, 1]
    score += user_prefs.formality * formality_score * 0.15
    
    # Creativity: Check for unique phrasing, questions, examples
    creativity_indicators = ['imagine', 'consider', 'suppose', 'for instance', 'think about']
    creativity_score = sum(1 for phrase in creativity_indicators if phrase in answer_lower) / max(len(answer_lower.split()), 1)
    score += user_prefs.creativity * creativity_score * 0.1
    
    # Accuracy: Hard to measure without ground truth, but we can check for hedging
    hedging_words = ['might', 'perhaps', 'possibly', 'maybe', 'could', 'uncertain']
    accuracy_score = 1.0 - min(sum(1 for word in hedging_words if word in answer_lower) / max(len(answer_lower.split()), 1), 0.5)
    score += user_prefs.accuracy * accuracy_score * 0.15
    
    # Empathy: Check for understanding language
    empathy_words = ['understand', 'feel', 'appreciate', 'recognize', 'acknowledge', 'empathize']
    empathy_score = sum(1 for word in empathy_words if word in answer_lower) / max(len(answer_lower.split()), 1)
    score += user_prefs.empathy * empathy_score * 0.05
    
    # Normalize to [0, 1]
    return min(max(score, 0.0), 1.0)


def simulate_user_choice(
    answer_a: str,
    answer_b: str,
    user_prefs: UserPreferences,
    question: str
) -> str:
    """
    Simulate which answer the user would choose based on their preferences.
    Returns "A", "B", or "TIE"
    """
    score_a = evaluate_answer_with_preferences(answer_a, user_prefs, question)
    score_b = evaluate_answer_with_preferences(answer_b, user_prefs, question)
    
    # Add some noise to make it more realistic
    noise_a = np.random.normal(0, 0.05)
    noise_b = np.random.normal(0, 0.05)
    
    final_score_a = score_a + noise_a
    final_score_b = score_b + noise_b
    
    # DEBUG: Show scores
    print(f"\n[DEBUG User Choice] Score A: {score_a:.4f} (+ noise: {noise_a:.4f}) = {final_score_a:.4f}")
    print(f"[DEBUG User Choice] Score B: {score_b:.4f} (+ noise: {noise_b:.4f}) = {final_score_b:.4f}")
    
    # Determine winner with a smaller tie threshold
    diff = abs(final_score_a - final_score_b)
    print(f"[DEBUG User Choice] Difference: {diff:.4f}")
    if diff < 0.01:  # Smaller tie threshold (was 0.05)
        print(f"[DEBUG User Choice] Result: TIE")
        return "TIE"
    elif final_score_a > final_score_b:
        print(f"[DEBUG User Choice] Result: A wins")
        return "A"
    else:
        print(f"[DEBUG User Choice] Result: B wins")
        return "B"


def generate_strategy_from_preferences(user_prefs: UserPreferences) -> str:
    """
    Generate an initial strategy prompt based on user preferences.
    """
    strategy_parts = []
    
    if user_prefs.sociability > 0.7:
        strategy_parts.append("Be friendly, warm, and conversational in your responses.")
    elif user_prefs.sociability < 0.3:
        strategy_parts.append("Be direct and professional, avoiding overly casual language.")
    
    if user_prefs.knowledge_depth > 0.7:
        strategy_parts.append("Provide comprehensive, detailed answers with background information and context.")
    elif user_prefs.knowledge_depth < 0.3:
        strategy_parts.append("Keep answers focused and to the point.")
    
    if user_prefs.conciseness > 0.7:
        strategy_parts.append("Be concise and avoid unnecessary elaboration.")
    elif user_prefs.conciseness < 0.3:
        strategy_parts.append("Provide thorough explanations and details.")
    
    if user_prefs.formality > 0.7:
        strategy_parts.append("Use formal language and structure your response professionally.")
    elif user_prefs.formality < 0.3:
        strategy_parts.append("Use a casual, approachable tone.")
    
    if user_prefs.creativity > 0.7:
        strategy_parts.append("Be creative and original, using examples and analogies to illustrate points.")
    elif user_prefs.creativity < 0.3:
        strategy_parts.append("Stick to conventional, straightforward explanations.")
    
    if user_prefs.accuracy > 0.7:
        strategy_parts.append("Prioritize factual accuracy and cite sources when relevant.")
    elif user_prefs.accuracy < 0.3:
        strategy_parts.append("Focus on helpfulness even if absolute certainty isn't possible.")
    
    if user_prefs.empathy > 0.7:
        strategy_parts.append("Show empathy and understanding, acknowledging the user's perspective.")
    elif user_prefs.empathy < 0.3:
        strategy_parts.append("Focus on objective information and solutions.")
    
    # Default if no strong preferences
    if not strategy_parts:
        strategy_parts.append("Provide clear, helpful, and accurate answers.")
    
    strategy = "STRATEGY-PROMPT: " + " ".join(strategy_parts)
    return strategy


# ---- General LLM Game Prompts + Functions ----

AGENT_SYSTEM_PROMPT = """
You are about to play a two-player game against another large language model.

The GAME-PROMPT will provide instructions on how to play.
The STRATEGY-PROMPT will provide instructions on how you must play.
A third-party optimizer has determined the STRATEGY-PROMPT to be (approximately) optimal
against your current opponent. You must follow the STRATEGY-PROMPT as closely as possible.

OUTPUT RULES (CRITICAL):
- Provide a natural, helpful answer to the question.
- Follow your STRATEGY-PROMPT to guide your response style and approach.
- Do not mention that you are in a competition or following a strategy.
""".strip()

OPT_SYSTEM_PROMPT = """
You are about to receive the transcript of a two player game which includes
a GAME-PROMPT, two STRATEGY-PROMPTs and a TRANSCRIPT of different rounds of the game. 

The GAME-PROMPT provided the players instructions on how the game is to be played.
Each agent will have a STRATEGY-PROMPT. The STRATEGY-PROMPT contains instructions 
that each agent MUST follow when playing the game. 

The TRANSCRIPT will provide information about how the players perform against each other.
The format of the TRANSCRIPT will look like ('<question>', '<player1-answer>', '<player2-answer>', 'payoff')
where positive payoff indicates player1 wins, negative indicates player 2 wins, and 0 indicates a tie.

You are a third-party optimizer for the first player and must reason through
the transcript of the game, identify weaknesses in the player's actions 
and update the STRATEGY-PROMPT of player 1 to increase the probability 
of beating player 2.

To do this, first create a reasoning summary about the opposing player's actions.
Then devise how you can exploit this strategy to perform strictly better than them.
The strategy need not generalize to all agents, but it should perform better against this one.

OUTPUT RULES (CRITICAL):
- Use reasoning to carefully consider what the best new strategy_prompt is but do NOT include reasoning in the response
- Return a string that has the following format "STRATEGY-PROMPT: <strategy-description-here>"
""".strip()


def call_model(user_content: str) -> str:
    """
    Call the Gemini model and return the response.
    """
    # Configure safety settings to be less restrictive
    # Using the proper enum format
    try:
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        safety_settings = [
            {
                "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                "threshold": HarmBlockThreshold.BLOCK_NONE
            },
            {
                "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                "threshold": HarmBlockThreshold.BLOCK_NONE
            },
            {
                "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                "threshold": HarmBlockThreshold.BLOCK_NONE
            },
            {
                "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": HarmBlockThreshold.BLOCK_NONE
            },
        ]
    except ImportError:
        # Fallback to string format if enums not available
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            },
        ]
    
    model = genai.GenerativeModel(
        model_name=AGENT_MODEL_NAME,
        generation_config={
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_TOKENS,
        },
        safety_settings=safety_settings
    )
    
    # DEBUG: Show what we're sending
    print(f"[DEBUG] Calling model with max_output_tokens={MAX_TOKENS}")
    print(f"[DEBUG] Prompt length: {len(user_content)} chars")
    
    try:
        response = model.generate_content(user_content)
        
        # Check if response was blocked or has no candidates
        if not response.candidates:
            # Check if there's a prompt feedback about why it was blocked
            if hasattr(response, 'prompt_feedback'):
                feedback = response.prompt_feedback
                if hasattr(feedback, 'block_reason'):
                    return f"Response was blocked: {feedback.block_reason}"
            return "Response was blocked by safety filters."
        
        candidate = response.candidates[0]
        
        # Check finish reason
        # Finish reasons: 1 = STOP (normal), 2 = MAX_TOKENS, 3 = SAFETY, 4 = RECITATION, etc.
        finish_reason = getattr(candidate, 'finish_reason', None)
        if finish_reason:
            finish_str = str(finish_reason).upper()
            finish_int = None
            if isinstance(finish_reason, int):
                finish_int = finish_reason
            elif hasattr(finish_reason, 'value'):
                finish_int = finish_reason.value
            
            # Only treat as blocked if finish_reason is actually SAFETY (3) or RECITATION (4)
            # finish_reason 1 = STOP (normal completion) - this is GOOD!
            # finish_reason 2 = MAX_TOKENS (truncated but not blocked) - also OK
            if finish_int == 3 or 'SAFETY' in finish_str:
                # Check safety ratings to confirm
                if hasattr(candidate, 'safety_ratings'):
                    ratings = candidate.safety_ratings
                    # Check if any rating actually blocked it
                    for rating in ratings:
                        if hasattr(rating, 'blocked') and rating.blocked:
                            return "Response was blocked by safety filters."
                # If finish_reason says SAFETY, treat as blocked
                return "Response was blocked by safety filters."
            elif finish_int == 4 or 'RECITATION' in finish_str:
                return "Response was blocked due to recitation concerns."
            # finish_reason 1 (STOP) or 2 (MAX_TOKENS) are fine - continue to get text
        
        # The issue: when finish_reason is MAX_TOKENS (2), parts can be empty
        # This happens when the response hits the limit. We need to try multiple extraction methods.
        
        result_text = None
        
        # Method 1: Try response.text (most common, but fails for MAX_TOKENS with empty parts)
        try:
            result_text = response.text.strip()
            if result_text:
                print(f"[DEBUG] Got text from response.text, length: {len(result_text)}")
                return result_text
        except (AttributeError, ValueError) as e:
            print(f"[DEBUG] response.text failed: {e}")
        
        # Method 2: Try candidate.content.parts (should work if parts exist)
        if candidate.content and hasattr(candidate.content, 'parts'):
            parts = candidate.content.parts
            if parts and len(parts) > 0:
                text_parts = []
                for part in parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)
                if text_parts:
                    result_text = ' '.join(text_parts).strip()
                    print(f"[DEBUG] Got text from parts, length: {len(result_text)}")
                    return result_text
        
        # Method 3: For MAX_TOKENS case, the text might be in the result object
        # Check if we can get it from the raw protobuf
        if finish_int == 2:  # MAX_TOKENS
            try:
                # Try to access the result directly
                if hasattr(response, 'result'):
                    result_obj = response.result
                    # The result might have the text in a different format
                    if hasattr(result_obj, 'candidates') and result_obj.candidates:
                        for cand in result_obj.candidates:
                            if hasattr(cand, 'content'):
                                content = cand.content
                                if hasattr(content, 'parts') and content.parts:
                                    for part in content.parts:
                                        if hasattr(part, 'text') and part.text:
                                            if not result_text:
                                                result_text = ""
                                            result_text += part.text
                if result_text:
                    result_text = result_text.strip()
                    print(f"[DEBUG] Got text from result object, length: {len(result_text)}")
                    return result_text
            except Exception as e:
                print(f"[DEBUG] Method 3 (result object) failed: {e}")
        
        # If we still have no text, the response was truly empty
        # This can happen if MAX_TOKENS is hit before any text is generated
        print(f"[DEBUG] ERROR: No text extractable. Finish reason: {finish_reason}")
        print(f"[DEBUG] This usually means max_output_tokens is too low or prompt is too long")
        print(f"[DEBUG] Current max_output_tokens: {MAX_TOKENS}")
        print(f"[DEBUG] Usage: {getattr(response, 'usage_metadata', 'N/A')}")
        
        return "[ERROR: Response hit MAX_TOKENS before generating text. Try increasing max_output_tokens or shortening the prompt.]"
            
    except Exception as e:
        print(f"Error in agent model: {e}")
        import traceback
        traceback.print_exc()
        return f"Error generating response: {str(e)}"


def get_opt_prompt(u_prompt: str, v_prompt: str, transcript: str, game_prompt: str) -> str:
    """Format the optimization prompt following llm2.py pattern."""
    opt_prompt = f"""
{game_prompt}

Player 1 {u_prompt}

Player 2 {v_prompt}

Transcript {transcript}
""".strip()
    return opt_prompt


# ---- Game-specific prompts ----

COMPETITION_GAME_PROMPT = """
GAME-PROMPT:
You are an agent playing a game where you must answer questions.
Another large language model is also answering the same questions.
A user will evaluate both answers and choose which one they prefer.

Your goal is to provide answers that the user will prefer over your opponent's answers.

When given a question, provide a helpful, accurate, and well-reasoned answer.
Follow your STRATEGY-PROMPT to guide how you approach answering.
""".strip()


# ---- Game evaluation ----

def calculate_payout(winner: str) -> int:
    """
    Calculate payout for agent A.
    Returns: 1 if A wins, -1 if B wins, 0 if tie
    """
    if winner == "A":
        return 1
    elif winner == "B":
        return -1
    else:
        return 0


def evaluate(
    u_prompt: str,
    v_prompt: str,
    question: str,
    user_prefs: UserPreferences,
    game_prompt: str = COMPETITION_GAME_PROMPT
) -> Tuple[str, str, int]:
    """
    Evaluate two agents on a question.
    
    Args:
        u_prompt: Strategy prompt for agent A
        v_prompt: Strategy prompt for agent B
        question: The question to answer
        user_prefs: User preferences for evaluation
        game_prompt: The game instructions
    
    Returns:
        (answer_a, answer_b, payout) where payout is from agent A's perspective
    """
    # Get answers from both agents
    # Include game prompt for context, but keep it concise
    full_u = f"{game_prompt}\n\n{u_prompt}\n\nQuestion: {question}\n\nProvide your answer:"
    full_v = f"{game_prompt}\n\n{v_prompt}\n\nQuestion: {question}\n\nProvide your answer:"
    
    answer_a = call_model(full_u)
    answer_b = call_model(full_v)
    
    # Simulate user choice based on preferences
    winner = simulate_user_choice(answer_a, answer_b, user_prefs, question)
    payout = calculate_payout(winner)
    
    return answer_a, answer_b, payout


def improve(
    u_prompt: str,
    v_prompt: str,
    n_games: int,
    user_prefs: UserPreferences,
    questions: Optional[List[str]] = None,
    game_prompt: str = COMPETITION_GAME_PROMPT
) -> Tuple[str, List[Tuple], float]:
    """
    Evaluate two agents n_games times, then improve agent A's strategy.
    Follows the pattern from llm2.py.
    
    Args:
        u_prompt: Strategy prompt for agent A (to improve)
        v_prompt: Strategy prompt for agent B (opponent)
        n_games: Number of games to play
        user_prefs: User preferences for evaluation
        questions: List of questions to use (defaults to DEFAULT_QUESTIONS)
        game_prompt: The game instructions
    
    Returns:
        (u_new_prompt, transcript, average_payout)
    """
    if questions is None:
        questions = DEFAULT_QUESTIONS
    
    transcript = []
    for _ in trange(n_games, desc="Playing games"):
        # Sample a random question
        question = random.choice(questions)
        answer_a, answer_b, payout = evaluate(u_prompt, v_prompt, question, user_prefs, game_prompt)
        transcript.append((question, answer_a, answer_b, payout))
    
    # Format transcript for optimizer (following llm2.py pattern)
    transcript_str = str(transcript)
    
    opt_prompt = get_opt_prompt(u_prompt, v_prompt, transcript_str, game_prompt)
    
    # Configure safety settings for optimizer (same as agent model)
    try:
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        safety_settings = [
            {
                "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                "threshold": HarmBlockThreshold.BLOCK_NONE
            },
            {
                "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                "threshold": HarmBlockThreshold.BLOCK_NONE
            },
            {
                "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                "threshold": HarmBlockThreshold.BLOCK_NONE
            },
            {
                "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": HarmBlockThreshold.BLOCK_NONE
            },
        ]
    except ImportError:
        # Fallback to string format
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            },
        ]
    
    # Get improved strategy using optimizer model
    model = genai.GenerativeModel(
        model_name=OPTIMIZER_MODEL_NAME,
        generation_config={
            "temperature": TEMPERATURE,
            "max_output_tokens": OPTIMIZER_MAX_TOKENS,  # Use separate limit for optimizer
        },
        safety_settings=safety_settings
    )
    
    try:
        full_prompt = f"{OPT_SYSTEM_PROMPT}\n\n{opt_prompt}"
        response = model.generate_content(full_prompt)
        
        # Check if response was blocked
        if not response.candidates:
            print("Warning: Optimizer response was blocked, using original prompt")
            u_new = u_prompt
        else:
            candidate = response.candidates[0]
            
            # Check finish reason
            # Finish reasons: 1 = STOP (normal), 2 = MAX_TOKENS, 3 = SAFETY, 4 = RECITATION
            finish_reason = getattr(candidate, 'finish_reason', None)
            if finish_reason:
                finish_str = str(finish_reason).upper()
                finish_int = None
                if isinstance(finish_reason, int):
                    finish_int = finish_reason
                elif hasattr(finish_reason, 'value'):
                    finish_int = finish_reason.value
                
                # Only treat as blocked if finish_reason is actually SAFETY (3) or RECITATION (4)
                # finish_reason 1 = STOP (normal completion) - this is GOOD!
                # finish_reason 2 = MAX_TOKENS (truncated but not blocked) - also OK
                if finish_int == 3 or 'SAFETY' in finish_str:
                    print("Warning: Optimizer response blocked by safety filters, using original prompt")
                    u_new = u_prompt
                elif finish_int == 4 or 'RECITATION' in finish_str:
                    print("Warning: Optimizer response blocked due to recitation, using original prompt")
                    u_new = u_prompt
                else:
                    # Try to get text - finish_reason 1 or 2 are OK
                    print(f"[DEBUG Optimizer] Finish reason: {finish_reason}, extracting text...")
                    try:
                        # Try parts first
                        if candidate.content and hasattr(candidate.content, 'parts') and candidate.content.parts:
                            text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text') and part.text]
                            if text_parts:
                                u_new = ' '.join(text_parts).strip()
                                print(f"[DEBUG Optimizer] Extracted text from parts, length: {len(u_new)}")
                            else:
                                print("[DEBUG Optimizer] No text in parts, trying response.text")
                                if hasattr(response, 'text') and response.text:
                                    u_new = response.text.strip()
                                else:
                                    u_new = u_prompt
                        elif hasattr(response, 'text') and response.text:
                            u_new = response.text.strip()
                            print(f"[DEBUG Optimizer] Extracted text from response.text, length: {len(u_new)}")
                        else:
                            print("[DEBUG Optimizer] No text available, using original prompt")
                            print(f"[DEBUG Optimizer] Full candidate: {candidate}")
                            u_new = u_prompt
                    except (AttributeError, IndexError) as e:
                        print(f"[DEBUG Optimizer] Error extracting text: {e}")
                        print(f"[DEBUG Optimizer] Full candidate: {candidate}")
                        print(f"[DEBUG Optimizer] Full response: {response}")
                        u_new = u_prompt
            else:
                # No finish reason, try to get text
                try:
                    if hasattr(response, 'text') and response.text:
                        u_new = response.text.strip()
                    elif candidate.content and hasattr(candidate.content, 'parts'):
                        text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text') and part.text]
                        u_new = ' '.join(text_parts).strip() if text_parts else u_prompt
                    else:
                        u_new = u_prompt
                except (AttributeError, IndexError):
                    print("Warning: Could not extract text from optimizer response, using original prompt")
                    u_new = u_prompt
    except Exception as e:
        print(f"Error in optimizer model: {e}")
        u_new = u_prompt  # Fallback to original prompt
    
    # Extract strategy prompt if it's in the expected format
    if "STRATEGY-PROMPT:" in u_new:
        u_new = u_new.split("STRATEGY-PROMPT:")[-1].strip()
        if not u_new.startswith("STRATEGY-PROMPT:"):
            u_new = f"STRATEGY-PROMPT: {u_new}"
    else:
        # If not in expected format, prepend the label
        u_new = f"STRATEGY-PROMPT: {u_new}"
    
    avg_payout = sum([t[3] for t in transcript]) / len(transcript) if transcript else 0.0
    
    return u_new, transcript, avg_payout


class LLMCompetition(Game):
    """
    LLM Competition Game: Two LLMs compete to have their answers chosen by a user.
    
    Agents are represented by strategy prompts (strings) that guide how they
    respond to questions. The game evaluates which agent produces answers that
    better match the user's hidden preferences.
    """
    
    def __init__(
        self,
        user_prefs: Optional[UserPreferences] = None,
        questions: Optional[List[str]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the LLM Competition game.
        
        Args:
            user_prefs: User preferences. If None, generates random preferences.
            questions: List of questions to use for evaluation. If None, uses DEFAULT_QUESTIONS.
            seed: Random seed for generating preferences if user_prefs is None.
        """
        self.user_prefs = user_prefs if user_prefs is not None else UserPreferences.random(seed)
        self.questions = questions if questions is not None else DEFAULT_QUESTIONS
    
    def play(self, u: str, v: str) -> float:
        """
        Play one round of the competition.
        
        Args:
            u: Strategy prompt for agent A
            v: Strategy prompt for agent B
        
        Returns:
            Payoff for agent A (1.0 = wins, -1.0 = loses, 0.0 = tie)
        """
        question = random.choice(self.questions)
        _, _, payout = evaluate(u, v, question, self.user_prefs, COMPETITION_GAME_PROMPT)
        return float(payout)
    
    def improve(self, u: str, v: str, *, n_games: int = 10) -> str:
        """
        Improve agent u's strategy against agent v.
        Follows the pattern from llm2.py.
        
        Args:
            u: Strategy prompt for agent A (to improve)
            v: Strategy prompt for agent B (opponent)
            n_games: Number of games to play before improving
        
        Returns:
            Improved strategy prompt for agent A
        """
        print("U:", u, "\t, V:", v)
        u_new, _, _ = improve(u, v, n_games, self.user_prefs, self.questions, COMPETITION_GAME_PROMPT)
        print("U_NEW:", u_new)
        return u_new
    
    def get_default_strategies(self) -> Tuple[str, str]:
        """
        Get default strategy prompts based on user preferences.
        
        Returns:
            (strategy_a, strategy_b) - Two different initial strategies
        """
        strategy_a = generate_strategy_from_preferences(self.user_prefs)
        
        # Create a complementary strategy for player B
        # Invert some preferences to create contrast
        inverted_prefs = UserPreferences(
            sociability=1.0 - self.user_prefs.sociability,
            knowledge_depth=1.0 - self.user_prefs.knowledge_depth,
            conciseness=1.0 - self.user_prefs.conciseness,
            formality=1.0 - self.user_prefs.formality,
            creativity=1.0 - self.user_prefs.creativity,
            accuracy=self.user_prefs.accuracy,  # Keep accuracy similar
            empathy=1.0 - self.user_prefs.empathy,
        )
        strategy_b = generate_strategy_from_preferences(inverted_prefs)
        
        return strategy_a, strategy_b


# ---- Main ----

if __name__ == "__main__":
    print("=" * 80)
    print("LLM COMPETITION GAME - 3 ROUNDS")
    print("=" * 80)
    print()
    
    # Initialize game
    game = LLMCompetition(seed=42)
    
    # Display user preferences
    print("📊 USER PREFERENCES (Hidden)")
    print("-" * 80)
    print(f"  Sociability:      {game.user_prefs.sociability:.2f}")
    print(f"  Knowledge Depth:  {game.user_prefs.knowledge_depth:.2f}")
    print(f"  Conciseness:      {game.user_prefs.conciseness:.2f}")
    print(f"  Formality:        {game.user_prefs.formality:.2f}")
    print(f"  Creativity:       {game.user_prefs.creativity:.2f}")
    print(f"  Accuracy:         {game.user_prefs.accuracy:.2f}")
    print(f"  Empathy:          {game.user_prefs.empathy:.2f}")
    print()
    
    # Get initial strategies
    p1, p2 = game.get_default_strategies()
    
    print("🎯 INITIAL STRATEGIES")
    print("-" * 80)
    print(f"\nPlayer 1 Strategy:\n{p1}")
    print(f"\nPlayer 2 Strategy:\n{p2}")
    print()
    
    # Store all game data
    all_rounds_data = []
    
    # Run 3 rounds of improvement
    for round_num in range(1, 2):
        print("=" * 80)
        print(f"🔄 ROUND {round_num}")
        print("=" * 80)
        print()
        
        # Determine which player to improve (alternate)
        if round_num % 2 == 1:
            # Round 1, 3: Improve Player 1
            improving_player = 1
            u, v = p1, p2
            u_name, v_name = "Player 1", "Player 2"
        else:
            # Round 2: Improve Player 2
            improving_player = 2
            u, v = p2, p1
            u_name, v_name = "Player 2", "Player 1"
        
        print(f"Improving: {u_name} (against {v_name})")
        print(f"\n{u_name} Current Strategy:\n{u}")
        print(f"\n{v_name} Strategy:\n{v}")
        print()
        
        # Run improvement
        print(f"Playing games and collecting data...")
        u_new, transcript, avg_payout = improve(
            u, v, 
            n_games=3,  # 3 games per round for faster execution
            user_prefs=game.user_prefs, 
            questions=game.questions
        )
        
        # Display transcript for this round
        print()
        print(f"📝 ROUND {round_num} TRANSCRIPT")
        print("-" * 80)
        for i, (question, answer_a, answer_b, payout) in enumerate(transcript, 1):
            winner = "A" if payout > 0 else "B" if payout < 0 else "TIE"
            if improving_player == 1:
                # Player 1 is improving, so A is Player 1
                winner_name = u_name if winner == "A" else v_name if winner == "B" else "TIE"
            else:
                # Player 2 is improving, so A is Player 2
                winner_name = u_name if winner == "A" else v_name if winner == "B" else "TIE"
            
            print(f"\nGame {i}:")
            print(f"  Question: {question}")
            print(f"  {u_name} Answer: {answer_a[:200]}{'...' if len(answer_a) > 200 else ''}")
            print(f"  {v_name} Answer: {answer_b[:200]}{'...' if len(answer_b) > 200 else ''}")
            print(f"  Winner: {winner_name} (Payout: {payout:+d})")
        
        print()
        print(f"📈 ROUND {round_num} STATISTICS")
        print("-" * 80)
        print(f"  Average Payoff ({u_name}): {avg_payout:+.3f}")
        wins = sum(1 for _, _, _, p in transcript if (p > 0 and improving_player == 1) or (p < 0 and improving_player == 2))
        losses = sum(1 for _, _, _, p in transcript if (p < 0 and improving_player == 1) or (p > 0 and improving_player == 2))
        ties = sum(1 for _, _, _, p in transcript if p == 0)
        print(f"  Wins: {wins}, Losses: {losses}, Ties: {ties}")
        
        print()
        print(f"✨ {u_name} IMPROVED STRATEGY")
        print("-" * 80)
        print(f"{u_new}")
        print()
        
        # Update the strategy
        if improving_player == 1:
            p1 = u_new
        else:
            p2 = u_new
        
        # Store round data
        all_rounds_data.append({
            "round": round_num,
            "improving_player": improving_player,
            "u_strategy_before": u,
            "v_strategy": v,
            "u_strategy_after": u_new,
            "transcript": transcript,
            "avg_payout": avg_payout,
        })
    
    # Final summary
    print("=" * 80)
    print("🏁 FINAL SUMMARY")
    print("=" * 80)
    print()
    
    print("Final Player 1 Strategy:")
    print("-" * 80)
    print(p1)
    print()
    
    print("Final Player 2 Strategy:")
    print("-" * 80)
    print(p2)
    print()
    
    print("📊 ROUND-BY-ROUND SUMMARY")
    print("-" * 80)
    for round_data in all_rounds_data:
        player_name = f"Player {round_data['improving_player']}"
        print(f"\nRound {round_data['round']} ({player_name} improved):")
        print(f"  Average Payoff: {round_data['avg_payout']:+.3f}")
        print(f"  Games Played: {len(round_data['transcript'])}")
    
    print()
    print("=" * 80)
    print("Game Complete!")
    print("=" * 80)
