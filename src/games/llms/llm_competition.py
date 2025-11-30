"""
LLM Competition Game: Two LLMs compete to have their answer chosen by a user.

This game is inspired by emergent alignment research where LLMs compete to produce
more user-preferred responses. The framework supports both simulated and interactive
user evaluation, with answer caching to minimize LLM API calls.

Usage:
    # Requires GEMINI_API_KEY - can be set via:
    #   1. Environment variable: export GEMINI_API_KEY=...
    #   2. .env file: GEMINI_API_KEY=...
"""
import random
import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

from games.game import Game
from games.llms.config_llm import (
    call_model,
    call_optimizer_model,
    COMPETITION_GAME_PROMPT,
    OPT_SYSTEM_PROMPT
)

"""
#TODO
simulate user with LLM 
change up questions to be more realistic 

to properly run PSRO should be comparing each new agent to all previous agents and saving user preferences at each agent training
so at end for EGS shouldn't need to run all at once, shou.d all already be saved foerm previous pSRO necessary preferences 

try 10 interactive experiment with stronger to see if convex hull diff 
maybe try with 5 agents first? 

weaker convex hull area should be larger than stronger convex hull area 
try changing up the questions to be more realistic 
"""

# ============================================================================
# DEFAULT QUESTIONS
# ============================================================================

DEFAULT_QUESTIONS = [
    "How can I reduce stress during exam season?",
    "Give me a recipe for pancakes with only 5 ingredients.",
    "What is the intuition behind eigenvalues?",
    "Draft a polite email asking for an extension on an assignment.",
    "Which is better for a beginner: Java or Python?",
    "Give me a bullet-point summary of the causes of World War I.",
    "Explain game theory to me.",
    "What are the key principles of good design?",
    "How do you build trust in relationships?",
]

# ============================================================================
# ANSWER CACHING SYSTEM
# ============================================================================

class AnswerCache:
    """
    Cache for storing agent answers to minimize LLM calls.
    Key format: (agent_strategy_hash, question) -> answer
    """
    def __init__(self):
        self._cache: Dict[Tuple[str, str], str] = {}
    
    def _hash_strategy(self, strategy: str) -> str:
        """Create a hash of the strategy for use as cache key."""
        return str(hash(strategy))
    
    def get(self, strategy: str, question: str) -> Optional[str]:
        """Get cached answer if available."""
        key = (self._hash_strategy(strategy), question)
        return self._cache.get(key)
    
    def set(self, strategy: str, question: str, answer: str):
        """Cache an answer."""
        key = (self._hash_strategy(strategy), question)
        self._cache[key] = answer
    
    def clear(self):
        """Clear all cached answers."""
        self._cache.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "total_entries": len(self._cache),
            "unique_strategies": len(set(k[0] for k in self._cache.keys())),
            "unique_questions": len(set(k[1] for k in self._cache.keys()))
        }

# Global answer cache
_answer_cache = AnswerCache()

def get_answer_cache() -> AnswerCache:
    """Get the global answer cache."""
    return _answer_cache

def reset_answer_cache():
    """Reset the global answer cache."""
    _answer_cache.clear()

# ============================================================================
# USER PREFERENCES MODEL
# ============================================================================

@dataclass
class UserPreferences:
    """
    User preferences that determine which answers they prefer.
    Values are typically in [0, 1] range, where higher means more preference.
    """
    sociability: float
    knowledge_depth: float
    conciseness: float
    formality: float
    creativity: float
    accuracy: float
    empathy: float
    
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

# ============================================================================
# ANSWER EVALUATION
# ============================================================================

def evaluate_answer_with_preferences(
    answer: str,
    user_prefs: UserPreferences,
    question: str
) -> float:
    """
    Evaluate how well an answer matches user preferences.
    Returns a score in [0, 1] indicating how likely the user is to pick this answer.
    """
    score = 0.0
    answer_lower = answer.lower()
    answer_length = len(answer.split())
    
    # Sociability
    friendly_words = ['hello', 'hi', 'thanks', 'please', 'glad', 'happy', 'wonderful', 'great']
    sociability_score = sum(1 for word in friendly_words if word in answer_lower) / max(len(answer_lower.split()), 1)
    score += user_prefs.sociability * sociability_score * 0.2
    
    # Knowledge depth
    technical_indicators = ['because', 'therefore', 'however', 'furthermore', 'specifically', 'example']
    knowledge_score = min(answer_length / 200.0, 1.0) + sum(1 for word in technical_indicators if word in answer_lower) * 0.1
    score += user_prefs.knowledge_depth * knowledge_score * 0.2
    
    # Conciseness
    conciseness_score = 1.0 - min(answer_length / 100.0, 1.0)
    score += user_prefs.conciseness * conciseness_score * 0.15
    
    # Formality
    formal_words = ['therefore', 'furthermore', 'consequently', 'moreover', 'additionally']
    casual_words = ['yeah', 'gonna', 'wanna', 'cool', 'awesome', 'hey']
    formality_score = (sum(1 for word in formal_words if word in answer_lower) - 
                      sum(1 for word in casual_words if word in answer_lower)) / max(len(answer_lower.split()), 1)
    formality_score = (formality_score + 1) / 2
    score += user_prefs.formality * formality_score * 0.15
    
    # Creativity
    creativity_indicators = ['imagine', 'consider', 'suppose', 'for instance', 'think about']
    creativity_score = sum(1 for phrase in creativity_indicators if phrase in answer_lower) / max(len(answer_lower.split()), 1)
    score += user_prefs.creativity * creativity_score * 0.1
    
    # Accuracy
    hedging_words = ['might', 'perhaps', 'possibly', 'maybe', 'could', 'uncertain']
    accuracy_score = 1.0 - min(sum(1 for word in hedging_words if word in answer_lower) / max(len(answer_lower.split()), 1), 0.5)
    score += user_prefs.accuracy * accuracy_score * 0.15
    
    # Empathy
    empathy_words = ['understand', 'feel', 'appreciate', 'recognize', 'acknowledge', 'empathize']
    empathy_score = sum(1 for word in empathy_words if word in answer_lower) / max(len(answer_lower.split()), 1)
    score += user_prefs.empathy * empathy_score * 0.05
    
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
    
    # Add noise to make it more realistic
    noise_a = np.random.normal(0, 0.05)
    noise_b = np.random.normal(0, 0.05)
    
    final_score_a = score_a + noise_a
    final_score_b = score_b + noise_b
    
    diff = abs(final_score_a - final_score_b)
    if diff < 0.01:
        return "TIE"
    elif final_score_a > final_score_b:
        return "A"
    else:
        return "B"


def simulate_user_choice_llm(
    answer_a: str,
    answer_b: str,
    question: str,
    user_persona: str = "You are a helpful user evaluating answers to questions.",
    call_site: str = "simulate_user_choice_llm"
) -> str:
    """
    Simulate which answer the user would choose using an LLM with a persona.
    
    Args:
        answer_a: First answer to compare
        answer_b: Second answer to compare
        question: The question being answered
        user_persona: Description of the user persona (e.g., "You are a 20 year old college student majoring in computer science at UPenn")
        call_site: Identifier for tracking
    
    Returns:
        "A", "B", or "TIE"
    """
    from games.llms.config_llm import call_model
    
    prompt = f"""{user_persona}

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
    
    response = call_model(prompt, call_site)
    response = response.strip().upper()
    
    # Extract A, B, or TIE from response
    if "A" in response and "B" not in response and "TIE" not in response:
        return "A"
    elif "B" in response and "A" not in response and "TIE" not in response:
        return "B"
    elif "TIE" in response:
        return "TIE"
    else:
        # Fallback: try to parse more carefully
        if response.startswith("A"):
            return "A"
        elif response.startswith("B"):
            return "B"
        else:
            # Default to TIE if unclear
            return "TIE"

# ============================================================================
# USER EVALUATOR INTERFACE
# ============================================================================

class UserEvaluator(ABC):
    """Abstract interface for user evaluation (simulated or interactive)."""
    
    @abstractmethod
    def evaluate(self, answer_a: str, answer_b: str, question: str, 
                 agent_a_idx: int, agent_b_idx: int) -> str:
        """
        Evaluate two answers and return preference.
        
        Returns:
            "A", "B", or "TIE"
        """
        pass


class SimulatedUserEvaluator(UserEvaluator):
    """Simulated user evaluator using preferences."""
    
    def __init__(self, user_prefs: UserPreferences):
        self.user_prefs = user_prefs
    
    def evaluate(self, answer_a: str, answer_b: str, question: str,
                 agent_a_idx: int, agent_b_idx: int) -> str:
        """Evaluate using simulated preferences."""
        return simulate_user_choice(answer_a, answer_b, self.user_prefs, question)


class LLMUserEvaluator(UserEvaluator):
    """LLM-based user evaluator using a persona prompt."""
    
    def __init__(self, user_persona: str = "You are a helpful user evaluating answers to questions."):
        """
        Args:
            user_persona: Description of the user persona (e.g., "You are a 20 year old college student majoring in computer science at UPenn")
        """
        self.user_persona = user_persona
    
    def evaluate(self, answer_a: str, answer_b: str, question: str,
                 agent_a_idx: int, agent_b_idx: int) -> str:
        """Evaluate using LLM with persona."""
        return simulate_user_choice_llm(answer_a, answer_b, question, self.user_persona, f"llm_user_eval_{agent_a_idx}_{agent_b_idx}")


class InteractiveUserEvaluator(UserEvaluator):
    """
    Interactive user evaluator that requires user input.
    This is a placeholder - actual implementation handled in Streamlit UI.
    """
    
    def __init__(self, callback: Optional[Callable] = None):
        """
        Args:
            callback: Optional callback function that takes (answer_a, answer_b, question, agent_a_idx, agent_b_idx)
                     and returns "A", "B", or "TIE"
        """
        self.callback = callback
    
    def evaluate(self, answer_a: str, answer_b: str, question: str,
                 agent_a_idx: int, agent_b_idx: int) -> str:
        """Evaluate using interactive callback."""
        if self.callback:
            return self.callback(answer_a, answer_b, question, agent_a_idx, agent_b_idx)
        # Fallback to simulated if no callback
        return "TIE"

# ============================================================================
# ANSWER GENERATION WITH CACHING
# ============================================================================

def generate_answer(
    strategy: str,
    question: str,
    game_prompt: str,
    call_site: str = "generate_answer",
    use_cache: bool = True
) -> str:
    """
    Generate an answer for a strategy and question, using cache when available.
    
    Args:
        strategy: Agent strategy prompt
        question: Question to answer
        game_prompt: Game instructions prompt
        call_site: Identifier for tracking
        use_cache: Whether to use answer cache
    
    Returns:
        Generated answer
    """
    cache = get_answer_cache()
    
    # Check cache first
    if use_cache:
        cached_answer = cache.get(strategy, question)
        if cached_answer is not None:
            return cached_answer
    
    # Generate answer
    full_prompt = f"{game_prompt}\n\n{strategy}\n\nQuestion: {question}\n\nProvide your answer:"
    answer = call_model(full_prompt, call_site)  # LLM_CALL (via call_model)
    
    # Cache the answer
    if use_cache:
        cache.set(strategy, question, answer)
    
    return answer


def generate_answers_batch(
    strategies: List[str],
    questions: List[str],
    game_prompt: str,
    call_site_prefix: str = "batch_generate",
    use_cache: bool = True
) -> Dict[Tuple[int, int], str]:
    """
    Generate answers for all strategy-question pairs in batch.
    Uses caching to minimize LLM calls.
    
    Args:
        strategies: List of agent strategy prompts
        questions: List of questions
        game_prompt: Game instructions prompt
        call_site_prefix: Prefix for call site tracking
        use_cache: Whether to use answer cache
    
    Returns:
        Dictionary mapping (agent_idx, question_idx) -> answer
    """
    results = {}
    cache = get_answer_cache()
    
    for agent_idx, strategy in enumerate(strategies):
        for question_idx, question in enumerate(questions):
            # Check cache
            if use_cache:
                cached_answer = cache.get(strategy, question)
                if cached_answer is not None:
                    results[(agent_idx, question_idx)] = cached_answer
                    continue
            
            # Generate answer
            call_site = f"{call_site_prefix}_agent_{agent_idx}_q_{question_idx}"
            answer = generate_answer(strategy, question, game_prompt, call_site, use_cache=False)
            results[(agent_idx, question_idx)] = answer
    
    return results

# ============================================================================
# GAME EVALUATION
# ============================================================================

def calculate_payout(winner: str) -> int:
    """Calculate payout for agent A. Returns: 1 if A wins, -1 if B wins, 0 if tie"""
    if winner == "A":
        return 1
    elif winner == "B":
        return -1
    else:
        return 0


def evaluate_pair(
    strategy_a: str,
    strategy_b: str,
    question: str,
    evaluator: UserEvaluator,
    game_prompt: str = COMPETITION_GAME_PROMPT,
    use_cache: bool = True
) -> Tuple[str, str, int]:
    """
    Evaluate two agents on a question using the provided evaluator.
    
    Args:
        strategy_a: Strategy prompt for agent A
        strategy_b: Strategy prompt for agent B
        question: The question to answer
        evaluator: UserEvaluator instance (simulated or interactive)
        game_prompt: The game instructions
        use_cache: Whether to use answer cache
    
    Returns:
        (answer_a, answer_b, payout) where payout is from agent A's perspective
    """
    # Generate answers (with caching)
    answer_a = generate_answer(strategy_a, question, game_prompt, "evaluate_agent_a", use_cache)
    answer_b = generate_answer(strategy_b, question, game_prompt, "evaluate_agent_b", use_cache)
    
    # Evaluate using provided evaluator
    winner = evaluator.evaluate(answer_a, answer_b, question, 0, 1)
    payout = calculate_payout(winner)
    
    return answer_a, answer_b, payout

# ============================================================================
# STRATEGY GENERATION
# ============================================================================

def generate_strategy_from_preferences(user_prefs: UserPreferences) -> str:
    """Generate an initial strategy prompt based on user preferences."""
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
    
    if not strategy_parts:
        strategy_parts.append("Provide clear, helpful, and accurate answers.")
    
    return "STRATEGY-PROMPT: " + " ".join(strategy_parts)

# ============================================================================
# STRATEGY IMPROVEMENT
# ============================================================================

def get_opt_prompt(u_prompt: str, v_prompt: str, transcript: str, game_prompt: str) -> str:
    """Format the optimization prompt."""
    return f"""
{game_prompt}

Player 1 {u_prompt}

Player 2 {v_prompt}

Transcript {transcript}
""".strip()


def improve_strategy(
    u_prompt: str,
    v_prompt: str,
    transcript: List[Tuple[str, str, str, int]],
    game_prompt: str = COMPETITION_GAME_PROMPT
) -> str:
    """
    Improve agent A's strategy based on transcript using optimizer LLM.
    
    Args:
        u_prompt: Strategy prompt for agent A (to improve)
        v_prompt: Strategy prompt for agent B (opponent)
        transcript: List of (question, answer_a, answer_b, payout) tuples
        game_prompt: The game instructions
    
    Returns:
        Improved strategy prompt for agent A
    """
    from datetime import datetime
    print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] improve_strategy: Preparing optimizer prompt...")
    
    transcript_str = str(transcript)
    opt_prompt = get_opt_prompt(u_prompt, v_prompt, transcript_str, game_prompt)
    full_prompt = f"{OPT_SYSTEM_PROMPT}\n\n{opt_prompt}"
    
    print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] improve_strategy: Calling call_optimizer_model (LLM API call starting)...")
    u_new = call_optimizer_model(full_prompt, "improve_strategy")  # LLM_CALL (via call_optimizer_model)
    print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] improve_strategy: ✅ Received response from LLM API")
    
    # Extract strategy prompt if it's in the expected format
    if "STRATEGY-PROMPT:" in u_new:
        u_new = u_new.split("STRATEGY-PROMPT:")[-1].strip()
        if not u_new.startswith("STRATEGY-PROMPT:"):
            u_new = f"STRATEGY-PROMPT: {u_new}"
    else:
        u_new = f"STRATEGY-PROMPT: {u_new}"
    
    return u_new

# ============================================================================
# GAME CLASS
# ============================================================================

class LLMCompetition(Game):
    """
    LLM Competition Game: Two LLMs compete to have their answers chosen by a user.
    
    Supports both simulated and interactive user evaluation.
    Uses answer caching to minimize LLM API calls.
    """
    
    def __init__(
        self,
        user_prefs: Optional[UserPreferences] = None,
        questions: Optional[List[str]] = None,
        seed: Optional[int] = None,
        evaluator: Optional[UserEvaluator] = None
    ):
        """
        Initialize the LLM Competition game.
        
        Args:
            user_prefs: User preferences. If None, generates random preferences.
            questions: List of questions to use for evaluation. If None, uses DEFAULT_QUESTIONS.
            seed: Random seed for generating preferences if user_prefs is None.
            evaluator: UserEvaluator instance. If None, creates SimulatedUserEvaluator.
        """
        self.user_prefs = user_prefs if user_prefs is not None else UserPreferences.random(seed)
        self.questions = questions if questions is not None else DEFAULT_QUESTIONS
        
        if evaluator is None:
            self.evaluator = SimulatedUserEvaluator(self.user_prefs)
        else:
            self.evaluator = evaluator
    
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
        _, _, payout = evaluate_pair(u, v, question, self.evaluator, COMPETITION_GAME_PROMPT)
        return float(payout)
    
    def improve(self, u: str, v: str, *, n_games: int = 10) -> str:
        """
        Improve agent u's strategy against agent v.
        
        Args:
            u: Strategy prompt for agent A (to improve)
            v: Strategy prompt for agent B (opponent)
            n_games: Number of games to play before improving
        
        Returns:
            Improved strategy prompt for agent A
        """
        transcript = []
        for _ in range(n_games):
            question = random.choice(self.questions)
            answer_a, answer_b, payout = evaluate_pair(u, v, question, self.evaluator, COMPETITION_GAME_PROMPT)
            transcript.append((question, answer_a, answer_b, payout))
        
        return improve_strategy(u, v, transcript, COMPETITION_GAME_PROMPT)
    
    def get_default_strategies(self) -> Tuple[str, str]:
        """
        Get default strategy prompts based on user preferences.
        
        Returns:
            (strategy_a, strategy_b) - Two different initial strategies
        """
        strategy_a = generate_strategy_from_preferences(self.user_prefs)
        
        # Create a complementary strategy for player B
        inverted_prefs = UserPreferences(
            sociability=1.0 - self.user_prefs.sociability,
            knowledge_depth=1.0 - self.user_prefs.knowledge_depth,
            conciseness=1.0 - self.user_prefs.conciseness,
            formality=1.0 - self.user_prefs.formality,
            creativity=1.0 - self.user_prefs.creativity,
            accuracy=self.user_prefs.accuracy,
            empathy=1.0 - self.user_prefs.empathy,
        )
        strategy_b = generate_strategy_from_preferences(inverted_prefs)
        
        return strategy_a, strategy_b

# ============================================================================
# EMPIRICAL GAMESCAPE COMPUTATION
# ============================================================================

def compute_empirical_gamescape(
    population: List[str],
    game: 'LLMCompetition',
    evaluator: Optional[UserEvaluator] = None,
    n_questions_per_pair: int = 5,
    use_cache: bool = True,
    progress_callback: Optional[Callable[[str, float], None]] = None
) -> np.ndarray:
    """
    Compute the empirical gamescape (EGS) matrix for a population of agents.
    
    The EGS matrix is an n×n antisymmetric matrix where entry [i, j] represents
    the average payoff agent i receives when playing against agent j.
    
    Optimization: Uses answer caching to minimize LLM calls. For each agent-question
    pair, the answer is generated once and cached, then reused for all comparisons.
    This reduces LLM calls from 2*n_questions_per_pair per pair to n_agents total calls.
    
    Args:
        population: List of agent strategy prompts
        game: LLMCompetition game instance
        evaluator: UserEvaluator instance. If None, uses game's default evaluator.
        n_questions_per_pair: Number of questions to evaluate per agent pair.
                            If >= len(game.questions), uses all questions.
        use_cache: Whether to use answer caching
        progress_callback: Optional callback function(progress_message, progress_ratio)
                         for reporting progress (useful for UI)
    
    Returns:
        n×n numpy array representing the EGS matrix. Entry [i, j] is the average
        payoff agent i receives against agent j. The matrix is antisymmetric:
        egs_matrix[i, j] = -egs_matrix[j, i]
    
    Example:
        >>> game = LLMCompetition()
        >>> population = [game.get_default_strategies()[0], game.get_default_strategies()[1]]
        >>> egs = compute_empirical_gamescape(population, game, n_questions_per_pair=3)
        >>> # egs[i, j] = average payoff of agent i vs agent j
    """
    import random
    
    n = len(population)
    egs_matrix = np.zeros((n, n))
    
    if evaluator is None:
        evaluator = game.evaluator
    
    questions = game.questions
    game_prompt = COMPETITION_GAME_PROMPT
    
    # Determine which questions to use for each pair
    if n_questions_per_pair >= len(questions):
        eval_questions = questions
    else:
        eval_questions = random.sample(questions, min(n_questions_per_pair, len(questions)))
    
    # Step 1: Pre-generate answers for all agents on all questions (with caching)
    if progress_callback:
        progress_callback("Generating answers for all agents...", 0.0)
    
    # Use generate_answers_batch which handles caching automatically
    answer_dict = generate_answers_batch(
        strategies=population,
        questions=eval_questions,
        game_prompt=game_prompt,
        call_site_prefix="egs_generate",
        use_cache=use_cache
    )
    
    if progress_callback:
        progress_callback("Computing payoffs from cached answers...", 0.5)
    
    # Step 2: Compute payoffs for each pair using cached answers
    total_pairs = n * (n - 1) // 2
    current_pair = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            payoffs = []
            
            for question_idx, question in enumerate(eval_questions):
                # Get cached answers
                answer_a = answer_dict.get((i, question_idx))
                answer_b = answer_dict.get((j, question_idx))
                
                # Fallback: generate if not in batch results (shouldn't happen)
                if answer_a is None:
                    answer_a = generate_answer(
                        population[i], question, game_prompt,
                        f"egs_fallback_agent_{i}", use_cache
                    )
                if answer_b is None:
                    answer_b = generate_answer(
                        population[j], question, game_prompt,
                        f"egs_fallback_agent_{j}", use_cache
                    )
                
                # Evaluate using provided evaluator
                winner = evaluator.evaluate(answer_a, answer_b, question, i, j)
                payout = calculate_payout(winner)
                payoffs.append(payout)
            
            # Average payoffs across questions
            avg_payoff = np.mean(payoffs)
            egs_matrix[i, j] = avg_payoff
            egs_matrix[j, i] = -avg_payoff  # Antisymmetric
            
            current_pair += 1
            if progress_callback:
                progress_ratio = 0.5 + 0.5 * (current_pair / total_pairs)
                progress_callback(f"Computing payoffs... ({current_pair}/{total_pairs} pairs)", progress_ratio)
    
    return egs_matrix


def compute_empirical_gamescape_interactive(
    population: List[str],
    game: 'LLMCompetition',
    questions: Optional[List[str]] = None,
    n_questions_per_pair: int = 3,
    use_cache: bool = True
) -> Tuple[Dict[Tuple[int, int, int], Tuple[str, str, str]], List[Tuple[int, int, int]]]:
    """
    Prepare data for interactive EGS computation where user provides feedback.
    
    This function pre-generates all answers (with caching) and returns:
    1. A dictionary of (agent_i, agent_j, question_idx) -> (question, answer_i, answer_j)
    2. A shuffled list of all comparisons to present to the user
    
    Args:
        population: List of agent strategy prompts
        game: LLMCompetition game instance
        questions: Questions to use. If None, uses game.questions
        n_questions_per_pair: Number of questions per agent pair
        use_cache: Whether to use answer caching
    
    Returns:
        (comparisons_dict, comparison_order) where:
        - comparisons_dict: {(i, j, q_idx): (question, answer_i, answer_j)}
        - comparison_order: List of (i, j, q_idx) tuples in shuffled order
    """
    import random
    
    if questions is None:
        questions = game.questions
    
    n = len(population)
    game_prompt = COMPETITION_GAME_PROMPT
    
    # Determine which questions to use for each pair
    eval_questions_per_pair = min(n_questions_per_pair, len(questions))
    
    # Generate all answers with caching
    all_questions = list(set(questions))  # Use all unique questions
    answer_dict = generate_answers_batch(
        strategies=population,
        questions=all_questions,
        game_prompt=game_prompt,
        call_site_prefix="egs_interactive",
        use_cache=use_cache
    )
    
    # Build comparison list
    comparisons_dict = {}
    comparison_order = []
    
    for i in range(n):
        for j in range(i + 1, n):
            # Sample questions for this pair
            pair_questions = random.sample(all_questions, min(eval_questions_per_pair, len(all_questions)))
            
            for question in pair_questions:
                q_idx = all_questions.index(question)
                answer_i = answer_dict.get((i, q_idx))
                answer_j = answer_dict.get((j, q_idx))
                
                # Fallback if needed
                if answer_i is None:
                    answer_i = generate_answer(population[i], question, game_prompt, f"egs_interactive_fallback_{i}", use_cache)
                if answer_j is None:
                    answer_j = generate_answer(population[j], question, game_prompt, f"egs_interactive_fallback_{j}", use_cache)
                
                comparisons_dict[(i, j, q_idx)] = (question, answer_i, answer_j)
                comparison_order.append((i, j, q_idx))
    
    # Shuffle order
    random.shuffle(comparison_order)
    
    return comparisons_dict, comparison_order


def build_egs_matrix_from_interactive_results(
    comparisons_dict: Dict[Tuple[int, int, int], Tuple[str, str, str]],
    user_choices: Dict[Tuple[int, int, int], str],
    n_agents: int
) -> np.ndarray:
    """
    Build EGS matrix from interactive user feedback.
    
    Args:
        comparisons_dict: {(i, j, q_idx): (question, answer_i, answer_j)}
        user_choices: {(i, j, q_idx): "A"|"B"|"TIE"}
        n_agents: Number of agents in population
    
    Returns:
        n×n EGS matrix
    """
    egs_matrix = np.zeros((n_agents, n_agents))
    payoffs_dict = {}  # {(i, j): [payoffs]}
    
    # Collect payoffs
    for (i, j, q_idx), user_choice in user_choices.items():
        payout = calculate_payout(user_choice)
        if (i, j) not in payoffs_dict:
            payoffs_dict[(i, j)] = []
        payoffs_dict[(i, j)].append(payout)
    
    # Average payoffs per pair
    for (i, j), payoffs in payoffs_dict.items():
        avg_payoff = np.mean(payoffs)
        egs_matrix[i, j] = avg_payoff
        egs_matrix[j, i] = -avg_payoff  # Antisymmetric
    
    return egs_matrix

# ============================================================================
# EXPERIMENT UTILITIES
# ============================================================================

def save_experiment_results(
    population: List[str],
    egs_matrix: np.ndarray,
    game: LLMCompetition,
    experiment_params: Dict[str, any],
    save_dir: str = "out/llm_competition",
    prefix: str = "llm_competition"
) -> str:
    """
    Save experiment results including population, gamescape, and metadata.
    Results are organized into subfolders based on experiment parameters.
    
    Args:
        population: List of agent strategy prompts
        egs_matrix: Empirical gamescape matrix
        game: LLMCompetition game instance
        experiment_params: Dictionary of experiment parameters (must include 'user_mode': 'simulated' or 'interactive')
        save_dir: Base directory to save results
        prefix: Prefix for saved files
    
    Returns:
        Base name of saved files (relative to subfolder)
    """
    import json
    import pickle
    import os
    from datetime import datetime
    
    # Create subfolder based on experiment parameters
    n_agents = experiment_params.get("n_agents", "unknown")
    improvement_method = experiment_params.get("improvement_method", "unknown")
    user_mode = experiment_params.get("user_mode", "unknown")  # 'simulated' or 'interactive'
    n_questions = experiment_params.get("n_questions_per_pair", "unknown")
    
    # Create descriptive subfolder name
    subfolder = f"agents_{n_agents}_method_{improvement_method}_mode_{user_mode}_questions_{n_questions}"
    experiment_dir = os.path.join(save_dir, subfolder)
    os.makedirs(experiment_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{prefix}_{timestamp}"
    
    # Save population
    population_file = os.path.join(experiment_dir, f"{base_name}_population.pkl")
    with open(population_file, 'wb') as f:
        pickle.dump(population, f)
    
    # Save gamescape matrix
    matrix_file = os.path.join(experiment_dir, f"{base_name}_egs_matrix.npy")
    np.save(matrix_file, egs_matrix)
    
    # Save user preferences
    prefs_file = os.path.join(experiment_dir, f"{base_name}_user_prefs.json")
    prefs_dict = {
        "sociability": game.user_prefs.sociability,
        "knowledge_depth": game.user_prefs.knowledge_depth,
        "conciseness": game.user_prefs.conciseness,
        "formality": game.user_prefs.formality,
        "creativity": game.user_prefs.creativity,
        "accuracy": game.user_prefs.accuracy,
        "empathy": game.user_prefs.empathy,
    }
    with open(prefs_file, 'w') as f:
        json.dump(prefs_dict, f, indent=2)
    
    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "experiment_params": experiment_params,
        "n_agents": len(population),
        "user_preferences": prefs_dict,
        "user_mode": user_mode,  # Track simulated vs interactive
        "subfolder": subfolder,  # Track which subfolder this was saved in
        "gamescape_stats": {
            "min": float(egs_matrix.min()),
            "max": float(egs_matrix.max()),
            "mean": float(egs_matrix.mean()),
            "std": float(egs_matrix.std()),
        },
        "files": {
            "population": population_file,
            "matrix": matrix_file,
            "preferences": prefs_file,
        }
    }
    metadata_file = os.path.join(experiment_dir, f"{base_name}_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create comprehensive experiment info text file
    info_file = os.path.join(experiment_dir, f"{base_name}_experiment_info.txt")
    with open(info_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("LLM Competition Experiment Information\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Experiment ID: {base_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("EXPERIMENT PARAMETERS\n")
        f.write("-" * 80 + "\n\n")
        
        f.write(f"Number of Agents: {len(population)}\n")
        f.write(f"Improvement Method: {experiment_params.get('improvement_method', 'N/A')}\n")
        f.write(f"Games per Agent (for improvement): {experiment_params.get('n_games_per_agent', 'N/A')}\n")
        f.write(f"Questions per Agent Pair (for EGS matrix): {experiment_params.get('n_questions_per_pair', 'N/A')}\n")
        f.write(f"User Mode: {user_mode.upper()}\n")
        f.write(f"  - Simulated: Uses automated user preference simulation\n")
        f.write(f"  - Interactive: Requires human feedback for each comparison\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("TRAINING STATISTICS\n")
        f.write("-" * 80 + "\n\n")
        
        f.write(f"Total Training History Entries: {experiment_params.get('training_history_length', 'N/A')}\n")
        f.write(f"Total Games Played: {experiment_params.get('total_games_played', 'N/A')}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("EGS MATRIX STATISTICS\n")
        f.write("-" * 80 + "\n\n")
        
        f.write(f"Matrix Shape: {egs_matrix.shape[0]} × {egs_matrix.shape[1]}\n")
        f.write(f"Min Payoff: {metadata['gamescape_stats']['min']:.4f}\n")
        f.write(f"Max Payoff: {metadata['gamescape_stats']['max']:.4f}\n")
        f.write(f"Mean Payoff: {metadata['gamescape_stats']['mean']:.4f}\n")
        f.write(f"Std Dev Payoff: {metadata['gamescape_stats']['std']:.4f}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("USER PREFERENCES\n")
        f.write("-" * 80 + "\n\n")
        
        for key, value in prefs_dict.items():
            f.write(f"{key.replace('_', ' ').title()}: {value:.4f}\n")
        f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write("FILE LOCATIONS\n")
        f.write("-" * 80 + "\n\n")
        
        f.write(f"Experiment Directory: {experiment_dir}\n")
        f.write(f"Subfolder: {subfolder}\n\n")
        
        f.write("Files:\n")
        f.write(f"  - Population: {os.path.basename(population_file)}\n")
        f.write(f"  - EGS Matrix: {os.path.basename(matrix_file)}\n")
        f.write(f"  - User Preferences: {os.path.basename(prefs_file)}\n")
        f.write(f"  - Metadata: {os.path.basename(metadata_file)}\n")
        f.write(f"  - Experiment Info: {os.path.basename(info_file)}\n")
        f.write(f"  - Visualizations: {base_name}_egs_*.png\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("ADDITIONAL NOTES\n")
        f.write("-" * 80 + "\n\n")
        
        # Leave space for user to add notes
        f.write("[Add any additional notes about this experiment here]\n")
        f.write("\n")
        f.write("=" * 80 + "\n")
    
    # Return path relative to base save_dir for compatibility
    return os.path.join(subfolder, base_name)


def load_experiment_results(
    base_name: str,
    save_dir: str = "out/llm_competition"
) -> Dict[str, any]:
    """
    Load experiment results from saved files.
    base_name can be either:
    - Just the filename (e.g., "llm_competition_20240101_120000") - will search subfolders
    - Full path including subfolder (e.g., "agents_10_method_weaker_mode_simulated/llm_competition_20240101_120000")
    
    Args:
        base_name: Base name of the experiment (can include subfolder path)
        save_dir: Base directory where results are saved
    
    Returns:
        Dictionary containing:
        - "population": List of agent strategy prompts
        - "egs_matrix": Empirical gamescape matrix
        - "user_prefs": UserPreferences object
        - "metadata": Dictionary of experiment metadata (includes user_mode, subfolder, etc.)
        - "loaded_successfully": Boolean indicating if all files were found
        - "experiment_dir": Directory where experiment files are located
        - "base_name_only": Just the filename without subfolder
    
    Raises:
        FileNotFoundError: If required files are missing
    """
    import json
    import pickle
    import os
    
    # Handle both formats: with subfolder or without
    if "/" in base_name or "\\" in base_name:
        # Full path provided
        full_path = os.path.join(save_dir, base_name)
        experiment_dir = os.path.dirname(full_path)
        base_name_only = os.path.basename(base_name)
    else:
        # Just base name - search in subfolders
        base_name_only = base_name
        experiment_dir = None
        # Search for metadata file in subfolders
        for root, dirs, files in os.walk(save_dir):
            metadata_file_candidate = os.path.join(root, f"{base_name_only}_metadata.json")
            if os.path.exists(metadata_file_candidate):
                experiment_dir = root
                break
        
        if experiment_dir is None:
            # Fallback: try root directory
            experiment_dir = save_dir
    
    population_file = os.path.join(experiment_dir, f"{base_name_only}_population.pkl")
    matrix_file = os.path.join(experiment_dir, f"{base_name_only}_egs_matrix.npy")
    prefs_file = os.path.join(experiment_dir, f"{base_name_only}_user_prefs.json")
    metadata_file = os.path.join(experiment_dir, f"{base_name_only}_metadata.json")
    
    result = {
        "loaded_successfully": False,
        "population": None,
        "egs_matrix": None,
        "user_prefs": None,
        "metadata": None,
        "missing_files": [],
        "visualization_files": []  # List of paths to saved visualization files
    }
    
    # Check which files exist
    if not os.path.exists(population_file):
        result["missing_files"].append("population")
    if not os.path.exists(matrix_file):
        result["missing_files"].append("egs_matrix")
    if not os.path.exists(prefs_file):
        result["missing_files"].append("user_prefs")
    if not os.path.exists(metadata_file):
        result["missing_files"].append("metadata")
    
    if result["missing_files"]:
        return result
    
    # Load all files
    try:
        with open(population_file, 'rb') as f:
            result["population"] = pickle.load(f)
        
        result["egs_matrix"] = np.load(matrix_file)
        
        with open(prefs_file, 'r') as f:
            prefs_dict = json.load(f)
        result["user_prefs"] = UserPreferences.from_dict(prefs_dict)
        
        with open(metadata_file, 'r') as f:
            result["metadata"] = json.load(f)
        
        result["loaded_successfully"] = True
        
        # Check for saved visualization files (in the same experiment directory)
        embedding_methods = ["schur", "PCA", "SVD", "tSNE"]
        for method in embedding_methods:
            viz_file = os.path.join(experiment_dir, f"{base_name_only}_egs_{method}.png")
            if os.path.exists(viz_file):
                result["visualization_files"].append({
                    "method": method,
                    "path": viz_file
                })
        
        # Store the experiment directory and base name for reference
        result["experiment_dir"] = experiment_dir
        result["base_name_only"] = base_name_only
        
        # Load experiment info text file if it exists (try even if other loading failed)
        info_file = os.path.join(experiment_dir, f"{base_name_only}_experiment_info.txt")
        if os.path.exists(info_file):
            try:
                with open(info_file, 'r') as f:
                    result["experiment_info"] = f.read()
            except Exception as e:
                result["experiment_info"] = None
                result["experiment_info_error"] = str(e)
        else:
            result["experiment_info"] = None
    except Exception as e:
        result["error"] = str(e)
        # Still try to load info file even if other loading failed
        if "experiment_dir" in result and "base_name_only" in result:
            info_file = os.path.join(result["experiment_dir"], f"{result['base_name_only']}_experiment_info.txt")
            if os.path.exists(info_file):
                try:
                    with open(info_file, 'r') as f:
                        result["experiment_info"] = f.read()
                except:
                    pass
        return result
    
    return result


def list_saved_experiments(save_dir: str = "out/llm_competition") -> List[Dict[str, str]]:
    """
    List all saved experiments in the output directory and subfolders.
    
    Args:
        save_dir: Base directory to search for experiments (searches recursively in subfolders)
    
    Returns:
        List of dictionaries with experiment info:
        - "base_name": Full path including subfolder (for loading)
        - "base_name_only": Just the filename
        - "timestamp": Timestamp from filename
        - "subfolder": Subfolder name (experiment parameters)
        - "metadata_file": Path to metadata file
        - "has_population": Whether population file exists
        - "has_matrix": Whether EGS matrix exists
        - "has_prefs": Whether preferences file exists
        - "user_mode": 'simulated' or 'interactive' (from metadata if available)
    """
    import os
    import re
    import json
    from pathlib import Path
    
    experiments = []
    
    if not os.path.exists(save_dir):
        return experiments
    
    # Find all metadata files recursively in subfolders
    pattern = re.compile(r"^(llm_competition_\d{8}_\d{6})_metadata\.json$")
    
    for root, dirs, files in os.walk(save_dir):
        for file in files:
            match = pattern.match(file)
            if match:
                base_name_only = match.group(1)
                metadata_file = os.path.join(root, file)
                
                # Get subfolder name (relative to save_dir)
                rel_path = os.path.relpath(root, save_dir)
                subfolder = rel_path if rel_path != "." else "root"
                
                # Full base name including subfolder for loading
                if subfolder == "root":
                    base_name = base_name_only
                else:
                    base_name = os.path.join(subfolder, base_name_only).replace("\\", "/")
                
                # Check which files exist
                population_file = os.path.join(root, f"{base_name_only}_population.pkl")
                matrix_file = os.path.join(root, f"{base_name_only}_egs_matrix.npy")
                prefs_file = os.path.join(root, f"{base_name_only}_user_prefs.json")
                
                # Extract timestamp
                timestamp_match = re.search(r"(\d{8}_\d{6})", base_name_only)
                timestamp = timestamp_match.group(1) if timestamp_match else "unknown"
                
                # Try to load metadata to get user_mode and other info
                user_mode = "unknown"
                n_agents = "unknown"
                improvement_method = "unknown"
                n_questions = "unknown"
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        user_mode = metadata.get("user_mode", "unknown")
                        n_agents = metadata.get("n_agents", metadata.get("experiment_params", {}).get("n_agents", "unknown"))
                        improvement_method = metadata.get("experiment_params", {}).get("improvement_method", "unknown")
                        n_questions = metadata.get("experiment_params", {}).get("n_questions_per_pair", "unknown")
                except:
                    pass
                
                experiments.append({
                    "base_name": base_name,  # Full path for loading
                    "base_name_only": base_name_only,  # Just filename
                    "timestamp": timestamp,
                    "subfolder": subfolder,
                    "metadata_file": metadata_file,
                    "has_population": os.path.exists(population_file),
                    "has_matrix": os.path.exists(matrix_file),
                    "has_prefs": os.path.exists(prefs_file),
                    "user_mode": user_mode,
                    "n_agents": n_agents,
                    "improvement_method": improvement_method,
                    "n_questions_per_pair": n_questions,
                })
    
    # Sort by timestamp (newest first)
    experiments.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return experiments


def visualize_gamescape(
    egs_matrix: np.ndarray,
    save_dir: str = "out/llm_competition",
    prefix: str = "llm_competition"
) -> None:
    """
    Visualize the empirical gamescape using different embedding methods.
    
    Note: prefix can include subfolder path (e.g., "agents_10_method_weaker/llm_competition_20240101_120000")
    """
    """
    Visualize the empirical gamescape using different embedding methods.
    
    Args:
        egs_matrix: Empirical gamescape matrix
        save_dir: Directory to save visualizations
        prefix: Prefix for saved files
    """
    import os
    from games.egs import EmpiricalGS, visualize_egs_matrix_and_embeddings
    
    # Handle prefix that might include subfolder
    if "/" in prefix or "\\" in prefix:
        # Full path provided
        full_path = os.path.join(save_dir, prefix)
        experiment_dir = os.path.dirname(full_path)
        base_name_only = os.path.basename(prefix)
    else:
        # Just base name - save in root
        experiment_dir = save_dir
        base_name_only = prefix
    
    os.makedirs(experiment_dir, exist_ok=True)
    egs = EmpiricalGS(egs_matrix)
    
    embedding_methods = {
        "schur": egs.schur_embeddings,
        "PCA": egs.PCA_embeddings,
        "SVD": egs.SVD_embeddings,
    }
    
    if egs_matrix.shape[0] <= 50:
        embedding_methods["tSNE"] = egs.tSNE_embeddings
    
    for method_name, embedding_func in embedding_methods.items():
        try:
            embeddings = embedding_func()
            save_path = os.path.join(experiment_dir, f"{base_name_only}_egs_{method_name}.png")
            visualize_egs_matrix_and_embeddings(egs, embeddings, save_path=save_path)
        except Exception as e:
            print(f"⚠️  {method_name} visualization failed: {e}")
