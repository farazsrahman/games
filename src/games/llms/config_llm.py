"""
Configuration and API functions for LLM calls in the competition game.

This module handles all Gemini API interactions, model configuration,
and system prompts. Separated from game logic for better modularity.
"""
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

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

# Parallel execution configuration
# For I/O-bound tasks (API calls), we can use more threads than CPU cores
# Default: 20 concurrent requests (conservative to respect API rate limits)
# Can be overridden via GEMINI_MAX_CONCURRENT_REQUESTS environment variable
# Note: Gemini API typically allows 60+ requests per minute, but concurrent requests
# should be limited to avoid overwhelming the API and hitting rate limits
DEFAULT_MAX_WORKERS = int(os.environ.get("GEMINI_MAX_CONCURRENT_REQUESTS", "20"))

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
# PROMPT FORMATTING FUNCTIONS
# ============================================================================

def format_answer_generation_prompt(game_prompt: str, strategy: str, question: str) -> str:
    """
    Format the prompt for generating an agent's answer to a question.
    
    Args:
        game_prompt: The game instructions prompt
        strategy: The agent's strategy prompt
        question: The question to answer
    
    Returns:
        Formatted prompt string
    """
    return f"{game_prompt}\n\n{strategy}\n\nQuestion: {question}\n\nProvide your answer:"


def format_user_preference_prompt(
    user_persona: str,
    question: str,
    answer_a: str,
    answer_b: str
) -> str:
    """
    Format the prompt for LLM-based user preference evaluation.
    
    Args:
        user_persona: Description of the user persona
        question: The question being answered
        answer_a: First answer to compare
        answer_b: Second answer to compare
    
    Returns:
        Formatted prompt string
    """
    return f"""{user_persona}

You are evaluating two answers to the following question:

Question: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Which answer do you prefer? You must respond with exactly one of the following:
- "A" if you prefer Answer A
- "B" if you prefer Answer B
- "TIE" if you have no preference or both answers are equally good

Your response (A, B, or TIE):"""

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
    from datetime import datetime
    import time
    
    print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] call_model ({call_site}): Initializing model {AGENT_MODEL_NAME}...")
    
    model = genai.GenerativeModel(
        model_name=AGENT_MODEL_NAME,
        generation_config={
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_TOKENS,
        },
        safety_settings=get_safety_settings()
    )
    
    print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] call_model ({call_site}): Model initialized. Making API call to Gemini...")
    start_time = time.time()
    
    try:
        response = model.generate_content(user_content)  # LLM_CALL
        elapsed = time.time() - start_time
        print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] call_model ({call_site}): ✅ API call completed in {elapsed:.2f} seconds")
        
        result = extract_response_text(response)
        print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] call_model ({call_site}): Response extracted ({len(result)} chars)")
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] call_model ({call_site}): ❌ Error after {elapsed:.2f} seconds: {e}")
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
    from datetime import datetime
    import time
    
    print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] call_optimizer_model ({call_site}): Initializing model {OPTIMIZER_MODEL_NAME}...")
    
    model = genai.GenerativeModel(
        model_name=OPTIMIZER_MODEL_NAME,
        generation_config={
            "temperature": TEMPERATURE,
            "max_output_tokens": OPTIMIZER_MAX_TOKENS,
        },
        safety_settings=get_safety_settings()
    )
    
    print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] call_optimizer_model ({call_site}): Model initialized. Making API call to Gemini...")
    start_time = time.time()
    
    try:
        response = model.generate_content(user_content)  # LLM_CALL
        elapsed = time.time() - start_time
        print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] call_optimizer_model ({call_site}): ✅ API call completed in {elapsed:.2f} seconds")
        
        result = extract_response_text(response)
        print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] call_optimizer_model ({call_site}): Response extracted ({len(result)} chars)")
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] call_optimizer_model ({call_site}): ❌ Error after {elapsed:.2f} seconds: {e}")
        return "[ERROR: Optimizer call failed]"


# ============================================================================
# BATCH LLM CALLING FUNCTIONS (PARALLEL)
# ============================================================================

def _call_model_worker(args):
    """
    Worker function for parallel LLM calls.
    
    Args:
        args: Tuple of (user_content, call_site, model_type)
              where model_type is "agent" or "optimizer"
    
    Returns:
        Tuple of (call_site, result_text) or (call_site, error_message)
    """
    user_content, call_site, model_type = args
    
    try:
        if model_type == "optimizer":
            result = call_optimizer_model(user_content, call_site)
        else:
            result = call_model(user_content, call_site)
        return (call_site, result)
    except Exception as e:
        return (call_site, f"Error in parallel call: {str(e)}")


def _call_model_worker_silent(args):
    """
    Worker function for parallel LLM calls with reduced logging (to avoid spam).
    Only logs start/completion, not intermediate steps.
    
    Args:
        args: Tuple of (user_content, call_site, model_type)
              where model_type is "agent" or "optimizer"
    
    Returns:
        Tuple of (call_site, result_text) or (call_site, error_message)
    """
    from datetime import datetime
    import time
    import google.generativeai as genai
    
    user_content, call_site, model_type = args
    
    try:
        if model_type == "optimizer":
            model_name = OPTIMIZER_MODEL_NAME
            max_tokens = OPTIMIZER_MAX_TOKENS
        else:
            model_name = AGENT_MODEL_NAME
            max_tokens = MAX_TOKENS
        
        print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] [PARALLEL] call_model ({call_site}): Starting API call...")
        start_time = time.time()
        
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": TEMPERATURE,
                "max_output_tokens": max_tokens,
            },
            safety_settings=get_safety_settings()
        )
        
        response = model.generate_content(user_content)
        elapsed = time.time() - start_time
        
        result = extract_response_text(response)
        print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] [PARALLEL] call_model ({call_site}): ✅ Completed in {elapsed:.2f}s ({len(result)} chars)")
        
        return (call_site, result)
    except Exception as e:
        elapsed = time.time() - start_time if 'start_time' in locals() else 0
        print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] [PARALLEL] call_model ({call_site}): ❌ Error after {elapsed:.2f}s: {e}")
        return (call_site, f"Error in parallel call: {str(e)}")


def call_model_batch(
    prompts: List[Tuple[str, str]],
    max_workers: Optional[int] = None,
    model_type: str = "agent"
) -> Dict[str, str]:
    """
    Make multiple LLM calls in parallel using threads.
    
    Args:
        prompts: List of (user_content, call_site) tuples
        max_workers: Maximum number of parallel threads. If None, uses min(len(prompts), 10)
        model_type: "agent" or "optimizer" to determine which model to use
    
    Returns:
        Dictionary mapping call_site -> response_text
    
    Example:
        >>> prompts = [
        ...     ("What is 2+2?", "math_q1"),
        ...     ("What is 3+3?", "math_q2"),
        ... ]
        >>> results = call_model_batch(prompts)
        >>> print(results["math_q1"])  # Response for first prompt
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datetime import datetime
    
    if not prompts:
        return {}
    
    if max_workers is None:
        # Default to min of prompt count and DEFAULT_MAX_WORKERS
        # This balances performance with API rate limit considerations
        # For I/O-bound tasks (API calls), we can use more threads than CPU cores
        # but should respect API rate limits (typically 60+ requests/min for Gemini)
        max_workers = min(len(prompts), DEFAULT_MAX_WORKERS)
    
    print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] ⚡ call_model_batch: Starting {len(prompts)} PARALLEL {model_type} calls with {max_workers} workers...")
    start_time = datetime.now()
    
    # Prepare arguments for workers
    worker_args = [(user_content, call_site, model_type) for user_content, call_site in prompts]
    
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] ⚡ call_model_batch: Submitting {len(prompts)} tasks to thread pool...")
        future_to_site = {
            executor.submit(_call_model_worker_silent, args): args[1] 
            for args in worker_args
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_site):
            call_site = future_to_site[future]
            try:
                result_site, result_text = future.result()
                results[result_site] = result_text
                completed += 1
                if completed % 5 == 0 or completed == len(prompts):
                    print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] ⚡ call_model_batch: Progress: {completed}/{len(prompts)} calls completed...")
            except Exception as e:
                print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] ⚡ call_model_batch: Error for {call_site}: {e}")
                results[call_site] = f"Error: {str(e)}"
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] ⚡ call_model_batch: ✅ All {len(prompts)} PARALLEL calls completed in {elapsed:.2f} seconds (avg: {elapsed/len(prompts):.2f}s per call)")
    
    return results


def call_optimizer_model_batch(
    prompts: List[Tuple[str, str]],
    max_workers: Optional[int] = None
) -> Dict[str, str]:
    """
    Make multiple optimizer model calls in parallel.
    
    Args:
        prompts: List of (user_content, call_site) tuples
        max_workers: Maximum number of parallel threads. If None, uses min(len(prompts), 10)
    
    Returns:
        Dictionary mapping call_site -> response_text
    """
    return call_model_batch(prompts, max_workers=max_workers, model_type="optimizer")
