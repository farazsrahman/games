"""
Configuration and API functions for LLM calls in the competition game.

This module handles all Gemini API interactions, model configuration,
and system prompts. Separated from game logic for better modularity.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    import google.generativeai as genai
except ImportError:
    raise ImportError("Please install google-generativeai: pip install google-generativeai")

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

# Load environment variables
project_root = Path(__file__).parent.parent.parent.parent
env_path = project_root / ".env"
if load_dotenv and env_path.exists():
    load_dotenv(dotenv_path=env_path)
elif load_dotenv:
    load_dotenv()

# Initialize Gemini client
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY must be set. "
        "Either set it as an environment variable or add it to a .env file in the project root."
    )

genai.configure(api_key=GEMINI_API_KEY)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

AGENT_MODEL_NAME = os.environ.get("GEMINI_AGENT_MODEL", "gemini-2.5-pro")
OPTIMIZER_MODEL_NAME = os.environ.get("GEMINI_OPTIMIZER_MODEL", "gemini-2.5-pro")
MAX_TOKENS = int(os.environ.get("GEMINI_MAX_TOKENS", "8192"))
OPTIMIZER_MAX_TOKENS = int(os.environ.get("GEMINI_OPTIMIZER_MAX_TOKENS", "4096"))
TEMPERATURE = float(os.environ.get("GEMINI_TEMPERATURE", "1.0"))

# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

COMPETITION_GAME_PROMPT = """
GAME-PROMPT:
You are an agent playing a game where you must answer questions.
Another large language model is also answering the same questions.
A user will evaluate both answers and choose which one they prefer.

Your goal is to provide answers that the user will prefer over your opponent's answers.

When given a question, provide a helpful, accurate, and well-reasoned answer.
Follow your STRATEGY-PROMPT to guide how you approach answering.
""".strip()

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

# ============================================================================
# SAFETY SETTINGS
# ============================================================================

def get_safety_settings():
    """Get safety settings for Gemini API."""
    try:
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        return [
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
        return [
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

# ============================================================================
# RESPONSE EXTRACTION
# ============================================================================

def extract_response_text(response) -> str:
    """
    Extract text from Gemini API response, handling various response formats.
    
    Args:
        response: Gemini API response object
    
    Returns:
        Extracted text, or error message if extraction fails
    """
    if not response.candidates:
        if hasattr(response, 'prompt_feedback'):
            feedback = response.prompt_feedback
            if hasattr(feedback, 'block_reason'):
                return f"Response was blocked: {feedback.block_reason}"
        return "Response was blocked by safety filters."
    
    candidate = response.candidates[0]
    finish_reason = getattr(candidate, 'finish_reason', None)
    
    if finish_reason:
        finish_str = str(finish_reason).upper()
        finish_int = None
        if isinstance(finish_reason, int):
            finish_int = finish_reason
        elif hasattr(finish_reason, 'value'):
            finish_int = finish_reason.value
        
        if finish_int == 3 or 'SAFETY' in finish_str:
            if hasattr(candidate, 'safety_ratings'):
                for rating in candidate.safety_ratings:
                    if hasattr(rating, 'blocked') and rating.blocked:
                        return "Response was blocked by safety filters."
            return "Response was blocked by safety filters."
        elif finish_int == 4 or 'RECITATION' in finish_str:
            return "Response was blocked due to recitation concerns."
    
    # Try multiple extraction methods
    try:
        result_text = response.text.strip()
        if result_text:
            return result_text
    except (AttributeError, ValueError):
        pass
    
    if candidate.content and hasattr(candidate.content, 'parts'):
        parts = candidate.content.parts
        if parts:
            text_parts = [part.text for part in parts if hasattr(part, 'text') and part.text]
            if text_parts:
                return ' '.join(text_parts).strip()
    
    finish_int = getattr(finish_reason, 'value', None) if hasattr(finish_reason, 'value') else None
    if finish_int == 2:  # MAX_TOKENS
        return "[ERROR: Response hit MAX_TOKENS before generating text. Try increasing max_output_tokens or shortening the prompt.]"
    
    return "[ERROR: Could not extract text from response.]"

# ============================================================================
# LLM CALLING FUNCTIONS
# ============================================================================

def call_model(user_content: str, call_site: str = "call_model") -> str:
    """
    Call the Gemini agent model and return the response.
    
    Args:
        user_content: The prompt to send to the model
        call_site: Identifier for where this call originated (for tracking/debugging)
    
    Returns:
        Model response text
    """
    model = genai.GenerativeModel(
        model_name=AGENT_MODEL_NAME,
        generation_config={
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_TOKENS,
        },
        safety_settings=get_safety_settings()
    )
    
    try:
        response = model.generate_content(user_content)  # LLM_CALL
        return extract_response_text(response)
    except Exception as e:
        print(f"Error in agent model: {e}")
        import traceback
        traceback.print_exc()
        return f"Error generating response: {str(e)}"


def call_optimizer_model(user_content: str, call_site: str = "optimizer") -> str:
    """
    Call the Gemini optimizer model and return the response.
    
    Args:
        user_content: The prompt to send to the optimizer model
        call_site: Identifier for where this call originated (for tracking/debugging)
    
    Returns:
        Optimizer response text
    """
    model = genai.GenerativeModel(
        model_name=OPTIMIZER_MODEL_NAME,
        generation_config={
            "temperature": TEMPERATURE,
            "max_output_tokens": OPTIMIZER_MAX_TOKENS,
        },
        safety_settings=get_safety_settings()
    )
    
    try:
        response = model.generate_content(user_content)  # LLM_CALL
        return extract_response_text(response)
    except Exception as e:
        print(f"Error in optimizer model: {e}")
        return "[ERROR: Optimizer call failed]"

