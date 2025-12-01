"""
Streamlit UI components for LLM Competition game.
This module contains all the UI logic and state management for the LLM competition tab.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Import game logic
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from games.llms.llm_competition import (
    LLMCompetition, 
    COMPETITION_GAME_PROMPT,
    call_model,
    get_opt_prompt,
    OPT_SYSTEM_PROMPT,
    simulate_user_choice,
    simulate_user_choice_llm,
    compute_empirical_gamescape
)
from games.egs import EmpiricalGS, visualize_egs_matrix_and_embeddings


def initialize_training_state(state: Dict[str, Any]) -> None:
    """Initialize all state variables for LLM competition training."""
    state.setdefault("initialized", False)
    state.setdefault("current_game_idx", 0)
    state.setdefault("current_question_idx", 0)
    state.setdefault("training_history", [])
    state.setdefault("waiting_for_feedback", False)
    state.setdefault("current_transcript", [])
    state.setdefault("egs_matrix", None)
    state.setdefault("generating_answers", False)
    state.setdefault("improving_agent", False)
    state.setdefault("computing_egs", False)
    state.setdefault("feedback_given", False)
    state.setdefault("initializing", False)
    state.setdefault("processing_feedback", False)
    state.setdefault("user_choice", None)
    state.setdefault("egs_answer_cache", {})
    state.setdefault("egs_interactive_mode", False)
    state.setdefault("answer_cache", {})
    state.setdefault("preference_cache", {})
    state.setdefault("collecting_preferences", False)
    state.setdefault("preference_collection_queue", [])
    state.setdefault("current_preference_comparison", None)
    state.setdefault("egs_viz_generated", False)
    state.setdefault("egs_viz_data", None)
    state.setdefault("egs_viz_method", "schur")


def generate_answers_for_agent(state: Dict[str, Any], agent_idx: int) -> None:
    """Generate answers for an agent on all fixed questions and cache them."""
    from datetime import datetime
    fixed_questions = state.get("fixed_questions", [])
    answer_cache = state.get("answer_cache", {})
    population = state["population"]
    
    print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] generate_answers_for_agent: Starting for agent {agent_idx} ({len(fixed_questions)} questions)")
    
    for i, question in enumerate(fixed_questions):
        cache_key = (agent_idx, question)
        if cache_key not in answer_cache:
            print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] generate_answers_for_agent: Generating answer {i+1}/{len(fixed_questions)} for agent {agent_idx}...")
            full_prompt = f"{COMPETITION_GAME_PROMPT}\n\n{population[agent_idx]}\n\nQuestion: {question}\n\nProvide your answer:"
            answer = call_model(full_prompt, f"agent_{agent_idx}_q_{i}")
            answer_cache[cache_key] = answer
            print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] generate_answers_for_agent: ✅ Cached answer {i+1}/{len(fixed_questions)}")
        else:
            print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] generate_answers_for_agent: Answer {i+1}/{len(fixed_questions)} already cached, skipping")
    
    print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] generate_answers_for_agent: ✅ Completed for agent {agent_idx}")
    state["answer_cache"] = answer_cache


def compute_agent_scores(state: Dict[str, Any], agent_i: int, agent_j: int) -> float:
    """
    Compute average score of agent_i against agent_j using cached preferences.
    Returns average payoff (1 = i wins, -1 = j wins, 0 = tie).
    """
    fixed_questions = state.get("fixed_questions", [])
    preference_cache = state.get("preference_cache", {})
    scores = []
    
    for question in fixed_questions:
        # Try both (i, j) and (j, i) since cache might have either order
        cache_key_ij = (agent_i, agent_j, question)
        cache_key_ji = (agent_j, agent_i, question)
        
        user_choice = None
        if cache_key_ij in preference_cache:
            user_choice = preference_cache[cache_key_ij]
        elif cache_key_ji in preference_cache:
            # If stored as (j, i), need to flip the choice
            cached_choice = preference_cache[cache_key_ji]
            if cached_choice == "A":
                user_choice = "B"  # j wins means i loses
            elif cached_choice == "B":
                user_choice = "A"  # j loses means i wins
            else:
                user_choice = "TIE"
        
        if user_choice is not None:
            if user_choice == "A":  # agent_i wins
                scores.append(1)
            elif user_choice == "B":  # agent_j wins
                scores.append(-1)
            else:  # TIE
                scores.append(0)
    
    if not scores:
        return 0.0
    
    return sum(scores) / len(scores)


def select_opponent_psro(state: Dict[str, Any], agent_idx: int) -> int:
    """
    Select opponent for agent_idx based on PSRO method (uniform, weaker, or stronger).
    Uses cached preferences to determine weaker/stronger agents.
    Falls back to uniform if no preferences are cached yet.
    """
    import random
    
    population = state["population"]
    improvement_method = state.get("improvement_method", "uniform")
    available_indices = [i for i in range(len(population)) if i != agent_idx]
    
    if not available_indices:
        return agent_idx  # Fallback to self-play
    
    if improvement_method == "uniform":
        return random.choice(available_indices)
    
    # For weaker/stronger, compute scores against all available agents
    agent_scores = {}
    has_any_preferences = False
    for opponent_idx in available_indices:
        score = compute_agent_scores(state, agent_idx, opponent_idx)
        agent_scores[opponent_idx] = score
        # Check if we have any preferences for this pair
        fixed_questions = state.get("fixed_questions", [])
        for question in fixed_questions:
            if (agent_idx, opponent_idx, question) in state.get("preference_cache", {}):
                has_any_preferences = True
                break
    
    # If no preferences cached yet, fall back to uniform
    if not has_any_preferences:
        return random.choice(available_indices)
    
    if improvement_method == "weaker":
        # Select from agents that agent_idx beats (score > 0)
        weaker_indices = [idx for idx, score in agent_scores.items() if score > 0]
        if weaker_indices:
            return random.choice(weaker_indices)
    elif improvement_method == "stronger":
        # Select from agents that beat agent_idx (score < 0)
        stronger_indices = [idx for idx, score in agent_scores.items() if score < 0]
        if stronger_indices:
            return random.choice(stronger_indices)
    
    # Fallback to uniform if no suitable opponents found
    return random.choice(available_indices)


def start_next_game(state: Dict[str, Any]) -> None:
    """Start the next game - uses fixed questions and cached answers when available."""
    if not state["initialized"] or not state["game"]:
        return
    
    population = state["population"]
    if len(population) < 2:
        st.error("Not enough agents!")
        return
    
    # Select agent to train (last one) and opponent using PSRO method
    agent_idx = len(population) - 1
    opponent_idx = select_opponent_psro(state, agent_idx)
    
    # Use fixed questions - cycle through them systematically
    fixed_questions = state.get("fixed_questions", state["game"].questions)
    if not fixed_questions:
        st.error("No fixed questions available!")
        return
    
    # Cycle through questions systematically
    current_question_idx = state.get("current_question_idx", 0)
    question = fixed_questions[current_question_idx % len(fixed_questions)]
    
    # Move to next question for next game
    state["current_question_idx"] = (current_question_idx + 1) % len(fixed_questions)
    
    # Check cache first
    answer_cache = state.get("answer_cache", {})
    answer_a = answer_cache.get((agent_idx, question))
    answer_b = answer_cache.get((opponent_idx, question))
    
    # Generate answers if not cached
    if answer_a is None:
        u_prompt = population[agent_idx]
        full_u = f"{COMPETITION_GAME_PROMPT}\n\n{u_prompt}\n\nQuestion: {question}\n\nProvide your answer:"
        answer_a = call_model(full_u, f"agent_{agent_idx}_q_{fixed_questions.index(question)}")
        answer_cache[(agent_idx, question)] = answer_a
        state["answer_cache"] = answer_cache
    
    if answer_b is None:
        v_prompt = population[opponent_idx]
        full_v = f"{COMPETITION_GAME_PROMPT}\n\n{v_prompt}\n\nQuestion: {question}\n\nProvide your answer:"
        answer_b = call_model(full_v, f"agent_{opponent_idx}_q_{fixed_questions.index(question)}")
        answer_cache[(opponent_idx, question)] = answer_b
        state["answer_cache"] = answer_cache
    
    # Set state after both answers are ready
    state["current_question"] = question
    state["current_answer_a"] = answer_a
    state["current_answer_b"] = answer_b
    state["current_agent_a_idx"] = agent_idx
    state["current_agent_b_idx"] = opponent_idx
    state["waiting_for_feedback"] = True
    state["feedback_given"] = False
    state["current_game_idx"] += 1


def setup_next_preference_comparison(state: Dict[str, Any]) -> None:
    """Set up the next comparison from the preference queue for display."""
    preference_queue = state.get("preference_collection_queue", [])
    answer_cache = state.get("answer_cache", {})
    preference_cache = state.get("preference_cache", {})
    
    if not preference_queue:
        # All preferences collected, continue with next agent
        state["collecting_preferences"] = False
        state["preference_collection_queue"] = []
        state["waiting_for_feedback"] = False
        
        # Continue with next agent or finish
        if len(state["population"]) < state["n_agents"]:
            new_agent = state["population"][-1]
            state["population"].append(new_agent)
            agent_idx = len(state["population"]) - 1
            opponent_idx = select_opponent_psro(state, agent_idx)
            state["current_agent_a_idx"] = agent_idx
            state["current_agent_b_idx"] = opponent_idx
            state["generating_answers"] = True
        else:
            if state["egs_matrix"] is None:
                state["computing_egs"] = True
        st.rerun()
        return
    
    # Find the first unanswered comparison
    current_comparison = None
    for comparison in preference_queue:
        agent_i, agent_j, question = comparison
        cache_key = (agent_i, agent_j, question)
        if cache_key not in preference_cache:
            current_comparison = comparison
            break
    
    if not current_comparison:
        # All preferences collected - continue with next agent
        state["collecting_preferences"] = False
        state["preference_collection_queue"] = []
        state["waiting_for_feedback"] = False
        # Continue to next agent (will be handled by calling function)
        return
    
    # Set up state for this comparison (reusing the same state variables as training)
    agent_i, agent_j, question = current_comparison
    answer_i = answer_cache.get((agent_i, question))
    answer_j = answer_cache.get((agent_j, question))
    
    if not answer_i or not answer_j:
        st.error("Missing answers for this comparison. Please refresh.")
        return
    
    # Use the same state variables as training feedback
    state["current_question"] = question
    state["current_answer_a"] = answer_i
    state["current_answer_b"] = answer_j
    state["current_agent_a_idx"] = agent_i
    state["current_agent_b_idx"] = agent_j
    state["waiting_for_feedback"] = True


def handle_user_choice(state: Dict[str, Any], user_choice: str) -> None:
    """Handle user choice - unified for both training and preference collection."""
    # Store user choice and process it
    state["user_choice"] = user_choice
    state["phase"] = "processing"
    st.rerun()


def compute_egs_from_cache(state: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Compute EGS matrix from cached preferences.
    Returns None if some preferences are missing.
    """
    from datetime import datetime
    print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] compute_egs_from_cache: Starting...")
    
    population = state["population"]
    n = len(population)
    fixed_questions = state.get("fixed_questions", [])
    preference_cache = state.get("preference_cache", {})
    
    print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] compute_egs_from_cache: n={n}, fixed_questions={len(fixed_questions)}, cache_size={len(preference_cache)}")
    
    if not fixed_questions:
        print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] compute_egs_from_cache: No fixed questions, returning None")
        return None
    
    egs_matrix = np.zeros((n, n))
    missing_pairs = []
    
    # Check all pairs
    total_pairs = n * (n - 1) // 2
    pairs_checked = 0
    for i in range(n):
        for j in range(i + 1, n):
            pairs_checked += 1
            payoffs = []
            all_cached = True
            
            for question in fixed_questions:
                # Try both (i, j) and (j, i) since cache might have either order
                cache_key_ij = (i, j, question)
                cache_key_ji = (j, i, question)
                
                user_choice = None
                if cache_key_ij in preference_cache:
                    user_choice = preference_cache[cache_key_ij]
                elif cache_key_ji in preference_cache:
                    # If stored as (j, i), need to flip the choice
                    cached_choice = preference_cache[cache_key_ji]
                    if cached_choice == "A":
                        user_choice = "B"  # j wins means i loses
                    elif cached_choice == "B":
                        user_choice = "A"  # j loses means i wins
                    else:
                        user_choice = "TIE"
                
                if user_choice is not None:
                    # Convert to payout: "A" (i wins) = 1, "B" (j wins) = -1, "TIE" = 0
                    if user_choice == "A":
                        payoffs.append(1)
                    elif user_choice == "B":
                        payoffs.append(-1)
                    else:  # TIE
                        payoffs.append(0)
                else:
                    all_cached = False
                    missing_pairs.append((i, j, question))
            
            if all_cached and payoffs:
                avg_payoff = np.mean(payoffs)
                egs_matrix[i, j] = avg_payoff
                egs_matrix[j, i] = -avg_payoff
                print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] compute_egs_from_cache: Pair ({i}, {j}): avg_payoff={avg_payoff:.3f}")
            elif not all_cached:
                print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] compute_egs_from_cache: Pair ({i}, {j}): missing preferences")
    
    print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] compute_egs_from_cache: Checked {pairs_checked}/{total_pairs} pairs, missing: {len(missing_pairs)}")
    
    # If any preferences are missing, return None
    if missing_pairs:
        print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] compute_egs_from_cache: Missing {len(missing_pairs)} preferences, returning None")
        print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] compute_egs_from_cache: First 10 missing: {missing_pairs[:10]}")
        return None
    
    # Ensure perfect antisymmetry (fix any floating point errors)
    # Make it perfectly antisymmetric: M = (M - M^T) / 2
    egs_matrix = (egs_matrix - egs_matrix.T) / 2
    
    return egs_matrix


def generate_egs_visualization(state: Dict[str, Any], method: str = "schur") -> Optional[str]:
    """
    Generate EGS visualization for the given method.
    Returns base64-encoded image data, or None on error.
    """
    if state.get("egs_matrix") is None:
        return None
    
    try:
        egs = EmpiricalGS(state["egs_matrix"])
        
        if method == "schur":
            embeddings = egs.schur_embeddings()
        elif method == "PCA":
            embeddings = egs.PCA_embeddings()
        elif method == "SVD":
            embeddings = egs.SVD_embeddings()
        elif method == "tSNE":
            if state["egs_matrix"].shape[0] > 50:
                embeddings = egs.PCA_embeddings()  # Fallback for large populations
            else:
                embeddings = egs.tSNE_embeddings()
        else:
            return None
        
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        visualize_egs_matrix_and_embeddings(egs, embeddings, save_path=tmp_path)
        
        with open(tmp_path, 'rb') as f:
            img_data = base64.b64encode(f.read()).decode()
        
        os.unlink(tmp_path)
        return img_data
    except Exception as e:
        st.error(f"Error generating {method} visualization: {str(e)}")
        return None

