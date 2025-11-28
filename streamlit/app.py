"""
Main Streamlit application for visualizing and running game theory simulations.
"""
import streamlit as st
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import time
import base64

# Add streamlit directory to path for imports
streamlit_dir = Path(__file__).parent
sys.path.insert(0, str(streamlit_dir))

from utils import ensure_demo_dir, list_demo_files, get_latest_demo_file, format_file_size
from game_runners import (
    run_disc_game_demo,
    run_blotto_game_demo,
    run_differentiable_lotto_demo,
    run_penneys_game_demo
)

# Page configuration
st.set_page_config(
    page_title="Open Ended Zero Sum Game Theory Simulations",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .game-section {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
    }
    /* Constrain static image sizes - but exclude GIFs to allow animation */
    .stImage img:not([src*=".gif"]) {
        max-height: 400px !important;
        max-width: 500px !important;
        width: auto !important;
        height: auto !important;
        object-fit: contain;
    }
    /* Allow GIFs to animate freely - use max-width only */
    .stImage img[src*=".gif"] {
        max-width: 900px !important;
        max-height: 600px !important;
        width: auto !important;
        height: auto !important;
    }
    /* Scrollable answer boxes for LLM Competition */
    .answer-box {
        height: 400px;
        overflow-y: auto;
        padding: 15px;
        border-radius: 5px;
        background-color: #f0f2f6;
        margin-bottom: 15px;
        line-height: 1.6;
    }
    .answer-box-a {
        border: 2px solid #1f77b4;
    }
    .answer-box-b {
        border: 2px solid #ff7f0e;
    }
    /* Style markdown content inside answer boxes */
    .answer-box h1, .answer-box h2, .answer-box h3 {
        margin-top: 1em;
        margin-bottom: 0.5em;
    }
    .answer-box p {
        margin-bottom: 1em;
    }
    .answer-box code {
        background-color: #e8e8e8;
        padding: 2px 4px;
        border-radius: 3px;
        font-family: 'Courier New', monospace;
    }
    .answer-box pre {
        background-color: #e8e8e8;
        padding: 10px;
        border-radius: 5px;
        overflow-x: auto;
    }
    .answer-box pre code {
        background-color: transparent;
        padding: 0;
    }
    .answer-box ul, .answer-box ol {
        margin-left: 20px;
        margin-bottom: 1em;
    }
    .answer-box li {
        margin-bottom: 0.5em;
    }
    .answer-box strong {
        font-weight: bold;
    }
    .answer-box em {
        font-style: italic;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if "runs" not in st.session_state:
        st.session_state.runs = {}
    if "comparison_mode" not in st.session_state:
        st.session_state.comparison_mode = False
    if "selected_runs" not in st.session_state:
        st.session_state.selected_runs = []


def render_disc_game_tab():
    """Render the Disc Game tab."""
    st.header("🎯 Disc Game")
    st.markdown("""
    The Disc Game demonstrates population diversity vs. convergence in symmetric zero-sum games.
    - **PSRO Uniform Weaker**: Leads to diverse population distributions
    - **PSRO Uniform Stronger**: Leads to convergence to same distribution
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Configuration")
        improvement_type = st.selectbox(
            "Improvement Strategy",
            ["weaker", "stronger"],
            help="Choose whether to improve against weaker or stronger opponents"
        )
        num_iterations = st.slider(
            "Number of Iterations",
            min_value=100,
            max_value=1000,
            value=500,
            step=50,
            help="More iterations = longer simulation but smoother visualization"
        )
        learning_rate = st.slider(
            "Learning Rate",
            min_value=0.001,
            max_value=0.1,
            value=0.01,
            step=0.001,
            format="%.3f"
        )
        
        col_viz1, col_viz2 = st.columns(2)
        with col_viz1:
            fps = st.slider("FPS", min_value=10, max_value=30, value=20)
        with col_viz2:
            dpi = st.slider("DPI", min_value=80, max_value=200, value=120)
    
    with col2:
        st.subheader("Run Simulation")
        run_button = st.button("Run Demo", type="primary", use_container_width=True)
        run_all_button = st.button("🔄 Run All Variations", use_container_width=True, 
                                   help="Run both 'weaker' and 'stronger' variations")
        
        if run_button:
            with st.spinner("Running simulation... This may take a while."):
                try:
                    result = run_disc_game_demo(
                        improvement_type=improvement_type,
                        num_iterations=num_iterations,
                        learning_rate=learning_rate,
                        fps=fps,
                        dpi=dpi
                    )
                    
                    # Store result in session state
                    run_id = f"disc_{improvement_type}_{int(time.time())}"
                    st.session_state.runs[run_id] = {
                        "game": "disc",
                        "type": improvement_type,
                        "result": result,
                        "timestamp": time.time()
                    }
                    st.success("✅ Simulation completed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error running simulation: {str(e)}")
        
        if run_all_button:
            variations = ["weaker", "stronger"]
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, var_type in enumerate(variations):
                status_text.text(f"Running {var_type} variation ({idx + 1}/{len(variations)})...")
                try:
                    result = run_disc_game_demo(
                        improvement_type=var_type,
                        num_iterations=num_iterations,
                        learning_rate=learning_rate,
                        fps=fps,
                        dpi=dpi
                    )
                    
                    # Store result in session state
                    run_id = f"disc_{var_type}_{int(time.time())}"
                    st.session_state.runs[run_id] = {
                        "game": "disc",
                        "type": var_type,
                        "result": result,
                        "timestamp": time.time()
                    }
                    progress_bar.progress((idx + 1) / len(variations))
                except Exception as e:
                    st.error(f"Error running {var_type} variation: {str(e)}")
            
            progress_bar.empty()
            status_text.empty()
            st.success(f"✅ All {len(variations)} variations completed!")
            st.rerun()
    
    # Display results
    st.subheader("Results")
    
    # List available demos
    demo_files = list_demo_files("disc", ".gif")
    if demo_files:
        selected_file = st.selectbox(
            "Select visualization to display",
            demo_files,
            key="disc_file_selector"
        )
        
        if selected_file:
            gif_path = Path("demos/disc") / selected_file
            if gif_path.exists():
                st.image(str(gif_path))
                
                # Download button
                with open(gif_path, "rb") as f:
                    st.download_button(
                        label="📥 Download GIF",
                        data=f.read(),
                        file_name=selected_file,
                        mime="image/gif"
                    )
                
                # File info
                file_size = gif_path.stat().st_size
                st.caption(f"File size: {format_file_size(file_size)}")
    else:
        st.info("No visualizations available. Run a simulation to generate one.")


def render_blotto_game_tab():
    """Render the Blotto Game tab."""
    st.header("⚔️ Colonel Blotto Game")
    st.markdown("""
    Discrete Colonel Blotto game where agents allocate resources across battlefields.
    The visualization shows the win rate evolution over training iterations.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Configuration")
        improvement_type = st.selectbox(
            "Improvement Strategy",
            ["uniform", "weaker", "stronger"],
            help="Choose PSRO improvement strategy"
        )
        num_iterations = st.slider(
            "Number of Iterations",
            min_value=100,
            max_value=2000,
            value=1000,
            step=100
        )
        n_rounds = st.slider(
            "Evaluation Rounds",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            help="Number of rounds per evaluation"
        )
        n_battlefields = st.number_input(
            "Number of Battlefields",
            min_value=2,
            max_value=5,
            value=3,
            step=1,
            help="Number of battlefields to allocate resources across"
        )
        budget = st.number_input(
            "Budget",
            min_value=5,
            max_value=20,
            value=10,
            step=1,
            help="Total budget to allocate across battlefields"
        )
        
        # Calculate and display number of possible allocations
        from math import comb
        num_allocations = comb(budget + n_battlefields - 1, n_battlefields - 1)
        if num_allocations > 1000:
            st.warning(f"⚠️ Large action space: {num_allocations:,} possible allocations. This may slow down training.")
        else:
            st.info(f"ℹ️ Action space size: {num_allocations:,} possible allocations")
        num_agents = st.slider(
            "Number of Agents",
            min_value=2,
            max_value=10,
            value=3,
            step=1,
            help="Number of agents in the population"
        )
    
    with col2:
        st.subheader("Run Simulation")
        run_button = st.button("Run Demo", type="primary", use_container_width=True, key="blotto_run")
        run_all_button = st.button("🔄 Run All Variations", use_container_width=True, key="blotto_run_all",
                                   help="Run 'uniform', 'weaker', and 'stronger' variations")
        
        if run_button:
            with st.spinner("Running simulation... This may take a while."):
                try:
                    result = run_blotto_game_demo(
                        improvement_type=improvement_type,
                        num_iterations=num_iterations,
                        n_rounds=n_rounds,
                        n_battlefields=n_battlefields,
                        budget=budget,
                        num_agents=num_agents
                    )
                    
                    run_id = f"blotto_{improvement_type}_{int(time.time())}"
                    st.session_state.runs[run_id] = {
                        "game": "blotto",
                        "type": improvement_type,
                        "result": result,
                        "timestamp": time.time()
                    }
                    st.success("✅ Simulation completed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error running simulation: {str(e)}")
        
        if run_all_button:
            variations = ["uniform", "weaker", "stronger"]
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, var_type in enumerate(variations):
                status_text.text(f"Running {var_type} variation ({idx + 1}/{len(variations)})...")
                try:
                    result = run_blotto_game_demo(
                        improvement_type=var_type,
                        num_iterations=num_iterations,
                        n_rounds=n_rounds,
                        n_battlefields=n_battlefields,
                        budget=budget,
                        num_agents=num_agents
                    )
                    
                    run_id = f"blotto_{var_type}_{int(time.time())}"
                    st.session_state.runs[run_id] = {
                        "game": "blotto",
                        "type": var_type,
                        "result": result,
                        "timestamp": time.time()
                    }
                    progress_bar.progress((idx + 1) / len(variations))
                except Exception as e:
                    st.error(f"Error running {var_type} variation: {str(e)}")
            
            progress_bar.empty()
            status_text.empty()
            st.success(f"✅ All {len(variations)} variations completed!")
            st.rerun()
    
    # Display results
    st.subheader("Results")
    
    # Helper function to load results from files for a given improvement type
    def load_results_from_files(improvement_type: str) -> Dict[str, Any]:
        """Load visualization files from directory for a given improvement type."""
        base_dir = Path("demos/blotto")
        base_name = f"blotto_PSRO_{improvement_type}"
        
        results = {
            "egs_visualization_paths": {},
            "gif_path_population": None,
            "gif_path_matchups": None,
            "plot_path": None,
        }
        
        # Load EGS visualizations
        methods = ["PCA", "SVD", "schur", "tSNE"]
        for method in methods:
            egs_path = base_dir / f"{base_name}_egs_{method.lower()}.png"
            if egs_path.exists():
                results["egs_visualization_paths"][method] = str(egs_path.absolute())
        
        # Load GIFs
        pop_gif = base_dir / f"{base_name}_population.gif"
        if pop_gif.exists():
            results["gif_path_population"] = str(pop_gif.absolute())
        
        match_gif = base_dir / f"{base_name}_matchups.gif"
        if match_gif.exists():
            results["gif_path_matchups"] = str(match_gif.absolute())
        
        # Load training plot
        training_plot = base_dir / f"{base_name}.png"
        if training_plot.exists():
            results["plot_path"] = str(training_plot.absolute())
        
        return results
    
    # Helper function to display results for a given improvement type
    def display_results_for_type(improvement_type: str):
        """Display all results for a given improvement type."""
        # First, try to get from session_state (most recent run of this type)
        result = None
        runs_of_type = [
            r for r in st.session_state.runs.values() 
            if r.get("game") == "blotto" and r.get("type") == improvement_type
        ]
        
        if runs_of_type:
            latest_run = max(runs_of_type, key=lambda x: x.get("timestamp", 0))
            if "result" in latest_run:
                result = latest_run["result"]
        
        # If no result in session_state, load from files
        if not result:
            result = load_results_from_files(improvement_type)
        
        # If still no results, show message
        if not result or (not result.get("egs_visualization_paths") and 
                         not result.get("gif_path_population") and 
                         not result.get("plot_path")):
            st.info(f"ℹ️ No results available for '{improvement_type}' type. Run a simulation to generate results.")
            return
        
        # Get paths from result
        egs_viz_paths = result.get("egs_visualization_paths", {})
        egs_viz_path = result.get("egs_visualization_path")
        gamescape_path = result.get("gamescape_matrix_path")
        embeddings_path = result.get("embeddings_2d_path")
        gif_pop_path = result.get("gif_path_population")
        gif_match_path = result.get("gif_path_matchups")
        training_plot_path = result.get("plot_path")
        
        # Display all EGS visualizations (Matrix + PCA, Schur, SVD, t-SNE)
        if egs_viz_paths and len(egs_viz_paths) > 0:
            st.markdown("#### Empirical Gamescape Visualizations")
            st.markdown("**Gamescape Matrix & 2D Embeddings (All Methods)**")
            
            # Display in a grid: 2 columns
            methods_order = ["PCA", "SVD", "schur", "tSNE"]
            cols = st.columns(2)
            
            displayed_count = 0
            for idx, method in enumerate(methods_order):
                if method in egs_viz_paths and Path(egs_viz_paths[method]).exists():
                    with cols[idx % 2]:
                        st.markdown(f"**{method.upper()}**")
                        st.image(str(egs_viz_paths[method]), use_container_width=True)
                        with open(egs_viz_paths[method], "rb") as f:
                            st.download_button(
                                label=f"📥 Download {method.upper()}",
                                data=f.read(),
                                file_name=Path(egs_viz_paths[method]).name,
                                mime="image/png",
                                key=f"blotto_{improvement_type}_egs_{method.lower()}_dl"
                            )
                    displayed_count += 1
            
            if displayed_count == 0:
                st.warning(f"⚠️ EGS visualizations were generated but files not found. Expected paths: {list(egs_viz_paths.values())}")
        elif egs_viz_path and Path(egs_viz_path).exists():
            # Fallback: single combined visualization (backward compatibility)
            st.markdown("#### Empirical Gamescape Visualization")
            st.markdown("**Gamescape Matrix & 2D Embeddings**")
            st.info("ℹ️ This is an older visualization format. Run a new simulation to see all EGS methods (PCA, SVD, Schur, t-SNE).")
            st.image(str(egs_viz_path), use_container_width=True)
            with open(egs_viz_path, "rb") as f:
                st.download_button(
                    label="📥 Download EGS Visualization",
                    data=f.read(),
                    file_name=Path(egs_viz_path).name,
                    mime="image/png",
                    key=f"blotto_{improvement_type}_egs_viz_dl"
                )
        elif gamescape_path or embeddings_path:
            # Fallback: show separate visualizations for backward compatibility
            st.markdown("#### Visualizations")
            col1, col2 = st.columns(2)
            
            with col1:
                if gamescape_path and Path(gamescape_path).exists():
                    st.markdown("**Gamescape Matrix**")
                    st.image(str(gamescape_path), use_container_width=True)
                    with open(gamescape_path, "rb") as f:
                        st.download_button(
                            label="📥 Download",
                            data=f.read(),
                            file_name=Path(gamescape_path).name,
                            mime="image/png",
                            key=f"blotto_{improvement_type}_gamescape_dl"
                        )
                else:
                    st.info("Gamescape matrix not available")
            
            with col2:
                if embeddings_path and Path(embeddings_path).exists():
                    st.markdown("**2D Policy Embeddings**")
                    st.image(str(embeddings_path), use_container_width=True)
                    with open(embeddings_path, "rb") as f:
                        st.download_button(
                            label="📥 Download",
                            data=f.read(),
                            file_name=Path(embeddings_path).name,
                            mime="image/png",
                            key=f"blotto_{improvement_type}_embeddings_dl"
                        )
                else:
                    st.info("2D embeddings not available")
        
        # Population GIF
        if gif_pop_path and Path(gif_pop_path).exists():
            st.markdown("#### Population Evolution (Allocations & Entropy)")
            with open(gif_pop_path, "rb") as f:
                gif_bytes = f.read()
                gif_data = base64.b64encode(gif_bytes).decode()
            
            st.markdown(
                f'<img src="data:image/gif;base64,{gif_data}" style="max-width: 100%; height: auto;" />',
                unsafe_allow_html=True
            )
            st.download_button(
                label="📥 Download Population GIF",
                data=gif_bytes,
                file_name=Path(gif_pop_path).name,
                mime="image/gif",
                key=f"blotto_{improvement_type}_pop_gif_dl"
            )
        
        # Matchups GIF
        if gif_match_path and Path(gif_match_path).exists():
            st.markdown("#### Matchups Evolution (Win Rates)")
            with open(gif_match_path, "rb") as f:
                gif_bytes = f.read()
                gif_data = base64.b64encode(gif_bytes).decode()
            
            st.markdown(
                f'<img src="data:image/gif;base64,{gif_data}" style="max-width: 100%; height: auto;" />',
                unsafe_allow_html=True
            )
            st.download_button(
                label="📥 Download Matchups GIF",
                data=gif_bytes,
                file_name=Path(gif_match_path).name,
                mime="image/gif",
                key=f"blotto_{improvement_type}_match_gif_dl"
            )
        
        # Training plot
        if training_plot_path and Path(training_plot_path).exists():
            with st.expander("📈 Training Progress Plot", expanded=False):
                st.image(str(training_plot_path), use_container_width=True)
                with open(training_plot_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Training Plot",
                        data=f.read(),
                        file_name=Path(training_plot_path).name,
                        mime="image/png",
                        key=f"blotto_{improvement_type}_training_dl"
                    )
        
        # Final statistics if available
        if "final_values" in result:
            st.markdown("---")
            st.markdown("#### Final Statistics")
            final_vals = result["final_values"]
            num_agents_display = result.get("num_agents", 3)
            
            num_cols = min(3, len(final_vals))
            cols = st.columns(num_cols)
            for idx, (key, value) in enumerate(final_vals.items()):
                with cols[idx % num_cols]:
                    display_key = key.replace('agent_', 'Agent ').replace('_vs_', ' vs ')
                    st.metric(display_key, f"{value:.4f}")
    
    # Create tabs for each improvement type
    tab1, tab2, tab3 = st.tabs(["Uniform", "Weaker", "Stronger"])
    
    with tab1:
        display_results_for_type("uniform")
    
    with tab2:
        display_results_for_type("weaker")
    
    with tab3:
        display_results_for_type("stronger")


def render_differentiable_lotto_tab():
    """Render the Differentiable Lotto tab."""
    st.header("🎲 Differentiable Lotto")
    st.markdown("""
    Continuous variant of the Lotto game where agents distribute mass over servers in R².
    Customers are softly assigned to nearest servers using softmax.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Configuration")
        improvement_type = st.selectbox(
            "Improvement Strategy",
            ["weaker", "stronger", "uniform"],
            key="diff_lotto_improvement"
        )
        num_iterations = st.slider(
            "Number of Iterations",
            min_value=50,
            max_value=500,
            value=100,
            step=50,
            key="diff_lotto_iterations"
        )
        num_customers = st.slider(
            "Number of Customers",
            min_value=5,
            max_value=20,
            value=9,
            step=1,
            key="diff_lotto_customers"
        )
        num_servers = st.slider(
            "Number of Servers",
            min_value=2,
            max_value=5,
            value=3,
            step=1,
            key="diff_lotto_servers"
        )
        
        st.markdown("---")
        st.subheader("Advanced Options")
        optimize_server_positions = st.checkbox(
            "Optimize Server Positions",
            value=True,
            key="diff_lotto_optimize"
        )
        enforce_width_constraint = st.checkbox(
            "Enforce Width Constraint",
            value=True,
            key="diff_lotto_width"
        )
        if enforce_width_constraint:
            width_penalty_lambda = st.slider(
                "Width Penalty λ",
                min_value=0.1,
                max_value=10.0,
                value=1.0,
                step=0.1,
                key="diff_lotto_lambda"
            )
        else:
            width_penalty_lambda = 0.0
        
        col_viz1, col_viz2 = st.columns(2)
        with col_viz1:
            fps = st.slider("FPS", min_value=10, max_value=30, value=20, key="diff_lotto_fps")
        with col_viz2:
            dpi = st.slider("DPI", min_value=80, max_value=200, value=120, key="diff_lotto_dpi")
    
    with col2:
        st.subheader("Run Simulation")
        run_button = st.button("Run Demo", type="primary", use_container_width=True, key="diff_lotto_run")
        run_all_button = st.button("🔄 Run All Variations", use_container_width=True, key="diff_lotto_run_all",
                                   help="Run 'weaker', 'stronger', and 'uniform' variations")
        
        if run_button:
            with st.spinner("Running simulation... This may take a while."):
                try:
                    result = run_differentiable_lotto_demo(
                        improvement_type=improvement_type,
                        num_iterations=num_iterations,
                        num_customers=num_customers,
                        num_servers=num_servers,
                        optimize_server_positions=optimize_server_positions,
                        enforce_width_constraint=enforce_width_constraint,
                        width_penalty_lambda=width_penalty_lambda if enforce_width_constraint else 0.0,
                        fps=fps,
                        dpi=dpi
                    )
                    
                    run_id = f"diff_lotto_{improvement_type}_{int(time.time())}"
                    st.session_state.runs[run_id] = {
                        "game": "differentiable_lotto",
                        "type": improvement_type,
                        "result": result,
                        "timestamp": time.time()
                    }
                    st.success("✅ Simulation completed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error running simulation: {str(e)}")
        
        if run_all_button:
            variations = ["weaker", "stronger", "uniform"]
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, var_type in enumerate(variations):
                status_text.text(f"Running {var_type} variation ({idx + 1}/{len(variations)})...")
                try:
                    result = run_differentiable_lotto_demo(
                        improvement_type=var_type,
                        num_iterations=num_iterations,
                        num_customers=num_customers,
                        num_servers=num_servers,
                        optimize_server_positions=optimize_server_positions,
                        enforce_width_constraint=enforce_width_constraint,
                        width_penalty_lambda=width_penalty_lambda if enforce_width_constraint else 0.0,
                        fps=fps,
                        dpi=dpi
                    )
                    
                    run_id = f"diff_lotto_{var_type}_{int(time.time())}"
                    st.session_state.runs[run_id] = {
                        "game": "differentiable_lotto",
                        "type": var_type,
                        "result": result,
                        "timestamp": time.time()
                    }
                    progress_bar.progress((idx + 1) / len(variations))
                except Exception as e:
                    st.error(f"Error running {var_type} variation: {str(e)}")
            
            progress_bar.empty()
            status_text.empty()
            st.success(f"✅ All {len(variations)} variations completed!")
            st.rerun()
    
    # Display results
    st.subheader("Results")
    
    # List available demos
    demo_files = list_demo_files("blotto", ".gif")
    diff_lotto_files = [f for f in demo_files if "demo_PSRO" in f]
    
    if diff_lotto_files:
        selected_file = st.selectbox(
            "Select visualization to display",
            diff_lotto_files,
            key="diff_lotto_file_selector"
        )
        
        if selected_file:
            gif_path = Path("demos/blotto") / selected_file
            if gif_path.exists():
                st.image(str(gif_path))
                
                with open(gif_path, "rb") as f:
                    st.download_button(
                        label="📥 Download GIF",
                        data=f.read(),
                        file_name=selected_file,
                        mime="image/gif",
                        key="diff_lotto_download"
                    )
                
                file_size = gif_path.stat().st_size
                st.caption(f"File size: {format_file_size(file_size)}")
                
                # Show statistics if available
                if "differentiable_lotto" in [r["game"] for r in st.session_state.runs.values()]:
                    latest_run = max(
                        [r for r in st.session_state.runs.values() if r["game"] == "differentiable_lotto"],
                        key=lambda x: x["timestamp"]
                    )
                    if "result" in latest_run:
                        res = latest_run["result"]
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Initial Payoff (1v2)", f"{res['initial_payoffs'][0]:.4f}")
                        with col2:
                            st.metric("Initial Payoff (1v3)", f"{res['initial_payoffs'][1]:.4f}")
                        with col3:
                            st.metric("Initial Payoff (2v3)", f"{res['initial_payoffs'][2]:.4f}")
    else:
        st.info("No visualizations available. Run a simulation to generate one.")


def render_penneys_game_tab():
    """Render the Penney's Game tab."""
    st.header("🪙 Penney's Game")
    st.markdown("""
    **Penney's Game** is a non-transitive zero-sum game where two players choose sequences of H/T.
    A coin is flipped repeatedly, and the first player whose sequence appears wins.
    This game demonstrates non-transitivity - for any sequence, there's a sequence that beats it with probability > 0.5.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Configuration")
        
        improvement_type = st.selectbox(
            "PSRO Strategy",
            ["uniform", "weaker", "stronger"],
            index=0,
            help="PSRO improvement strategy: uniform (sample uniformly), weaker (sample from weaker opponents), stronger (sample from stronger opponents)"
        )
        
        num_iterations = st.slider(
            "Number of Iterations",
            min_value=100,
            max_value=2000,
            value=500,
            step=100,
            help="Number of training iterations"
        )
        
        sequence_length = st.slider(
            "Sequence Length",
            min_value=2,
            max_value=4,
            value=3,
            step=1,
            help="Length of H/T sequences (2^length possible sequences)"
        )
        
        n_rounds = st.slider(
            "Evaluation Rounds",
            min_value=100,
            max_value=2000,
            value=500,
            step=100,
            help="Number of rounds to evaluate win rates"
        )
    
    with col2:
        st.subheader("Run Simulation")
        run_button = st.button("Run Demo", type="primary", use_container_width=True, key="penneys_run")
        run_all_button = st.button("🔄 Run All Variations", use_container_width=True, key="penneys_run_all",
                                   help="Run all three PSRO variants (uniform, weaker, stronger)")
    
    if run_button:
        with st.spinner("Running Penney's Game simulation..."):
            try:
                result = run_penneys_game_demo(
                    improvement_type=improvement_type,
                    num_iterations=num_iterations,
                    sequence_length=sequence_length,
                    n_rounds=n_rounds
                )
                
                # Store result
                run_id = f"penneys_{improvement_type}_{int(time.time())}"
                st.session_state.runs[run_id] = {
                    "game": "penneys",
                    "type": improvement_type,
                    "timestamp": time.time(),
                    "result": result
                }
                
                st.success("Simulation completed!")
                st.rerun()
            except Exception as e:
                st.error(f"Error running simulation: {e}")
    
    if run_all_button:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        variants = ["uniform", "weaker", "stronger"]
        variant_descriptions = {
            "uniform": "Uniform: Randomly selects any opponent",
            "weaker": "Weaker: Only improves against opponents it beats",
            "stronger": "Stronger: Only improves against opponents that beat it"
        }
        
        try:
            for idx, variant in enumerate(variants):
                status_text.text(f"Running PSRO {variant.capitalize()}... ({variant_descriptions[variant]})")
                progress_bar.progress((idx) / len(variants))
                
                result = run_penneys_game_demo(
                    improvement_type=variant,
                    num_iterations=num_iterations,
                    sequence_length=sequence_length,
                    n_rounds=n_rounds
                )
                run_id = f"penneys_{variant}_{int(time.time())}"
                st.session_state.runs[run_id] = {
                    "game": "penneys",
                    "type": variant,
                    "timestamp": time.time(),
                    "result": result
                }
            
            progress_bar.progress(1.0)
            status_text.text("All simulations completed!")
            st.success(f"✅ Completed all 3 PSRO variants! Generated plots and GIFs for: {', '.join(variants)}")
            st.rerun()
        except Exception as e:
            st.error(f"Error running simulations: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    # Display results
    st.subheader("Results")
    
    # List available demo files
    demo_files_png = list_demo_files("penneys", ".png")
    demo_files_gif = list_demo_files("penneys", ".gif")
    penneys_plots = [f for f in demo_files_png if "penneys_PSRO" in f]
    penneys_gifs = [f for f in demo_files_gif if "penneys_PSRO" in f and ("population" in f or "matchups" in f)]
    
    # Show GIFs if available
    if penneys_gifs:
        st.markdown("#### Animated Visualizations")
        gif_type = st.radio(
            "Select GIF type",
            ["Population (Probabilities & Entropy)", "Matchups (Win Rates)"],
            key="penneys_gif_type"
        )
        
        # Filter GIFs by type
        if "Population" in gif_type:
            gif_files = [f for f in penneys_gifs if "population" in f]
        else:
            gif_files = [f for f in penneys_gifs if "matchups" in f]
        
        if gif_files:
            selected_gif = st.selectbox(
                "Select GIF to display",
                gif_files,
                key="penneys_gif_selector"
            )
            
            if selected_gif:
                gif_path = Path("demos/penneys") / selected_gif
                if gif_path.exists():
                    # Read GIF file once for both display and download
                    with open(gif_path, "rb") as f:
                        gif_bytes = f.read()
                        gif_data = base64.b64encode(gif_bytes).decode()
                    
                    # Determine max width based on GIF type
                    max_width = "900px" if "population" in selected_gif else "600px"
                    
                    # Use base64 encoding with HTML img tag to ensure GIF animates properly
                    st.markdown(
                        f'<img src="data:image/gif;base64,{gif_data}" style="max-width: {max_width}; height: auto;" />',
                        unsafe_allow_html=True
                    )
                    
                    # Download button
                    st.download_button(
                        label="📥 Download GIF",
                        data=gif_bytes,
                        file_name=selected_gif,
                        mime="image/gif",
                        key="penneys_gif_download"
                    )
    
    # Show static plots
    if penneys_plots:
        st.markdown("#### Static Plots")
        selected_file = st.selectbox(
            "Select plot to display",
            penneys_plots,
            key="penneys_file_selector"
        )
        
        if selected_file:
            plot_path = Path("demos/penneys") / selected_file
            if plot_path.exists():
                st.image(str(plot_path))
                
                with open(plot_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Plot",
                        data=f.read(),
                        file_name=selected_file,
                        mime="image/png",
                        key="penneys_download"
                    )
    
    # Show final statistics if available
    if "penneys" in [r["game"] for r in st.session_state.runs.values()]:
        latest_run = max(
            [r for r in st.session_state.runs.values() if r["game"] == "penneys"],
            key=lambda x: x["timestamp"]
        )
        if "result" in latest_run and "final_values" in latest_run["result"]:
            st.markdown("#### Final Statistics")
            final_vals = latest_run["result"]["final_values"]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Agent 1 vs 2", f"{final_vals['agent_1_vs_2']:.4f}")
            with col2:
                st.metric("Agent 1 vs 3", f"{final_vals['agent_1_vs_3']:.4f}")
            with col3:
                st.metric("Agent 2 vs 3", f"{final_vals['agent_2_vs_3']:.4f}")
    
    if not penneys_plots and not penneys_gifs:
        st.info("No visualizations available. Run a simulation to generate them.")


def render_llm_competition_tab():
    """Render the LLM Competition tab with interactive training."""
    import sys
    from pathlib import Path
    import random
    
    # Add src to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    
    try:
        from games.llms.llm_competition import (
            LLMCompetition, UserPreferences, DEFAULT_QUESTIONS,
            COMPETITION_GAME_PROMPT,
            get_opt_prompt, OPT_SYSTEM_PROMPT,
            save_experiment_results, visualize_gamescape,
            compute_empirical_gamescape, compute_empirical_gamescape_interactive,
            build_egs_matrix_from_interactive_results,
            load_experiment_results, list_saved_experiments,
            load_experiment_results, list_saved_experiments
        )
        from games.llms.config_llm import call_model
        from games.egs import EmpiricalGS
        import google.generativeai as genai
        import numpy as np
        import matplotlib.pyplot as plt
        import io
        import base64
        import os
    except ImportError as e:
        error_msg = str(e)
        st.error(f"Failed to import required modules: {error_msg}")
        
        if "GEMINI_API_KEY" in error_msg or "gemini" in error_msg.lower():
            st.warning("⚠️ GEMINI_API_KEY is not set!")
            st.info("""
            To use the LLM Competition tab, you need to set your Gemini API key:
            
            1. Get your API key from: https://makersuite.google.com/app/apikey
            2. Set it as an environment variable:
               ```bash
               export GEMINI_API_KEY=your_api_key_here
               ```
            3. Or create a `.env` file in the project root with:
               ```
               GEMINI_API_KEY=your_api_key_here
               ```
            """)
        else:
            st.info("Please ensure all dependencies are installed:")
            st.code("pip install -r requirements.txt")
        
        import traceback
        with st.expander("Full error details"):
            st.code(traceback.format_exc())
        return
    
    st.header("🤖 LLM Competition")
    st.markdown("""
    Train a population of LLM agents that compete to produce user-preferred answers.
    You'll provide feedback by choosing which answer you prefer for each comparison.
    """)
    
    # Initialize session state for LLM competition
    if "llm_training_state" not in st.session_state:
        st.session_state.llm_training_state = {
            "initialized": False,
            "population": [],
            "game": None,
            "current_agent_idx": 0,
            "current_game_idx": 0,
            "current_question": None,
            "current_answer_a": None,
            "current_answer_b": None,
            "current_agent_a_idx": None,
            "current_agent_b_idx": None,
            "training_history": [],
            "egs_matrix": None,
            "waiting_for_feedback": False,
            "current_transcript": [],
            "generating_answers": False,
            "improving_agent": False,
            "computing_egs": False,
            "feedback_given": False,
            "processing_feedback": False,
            "user_choice": None,
            "n_agents": 10,
            "n_games_per_agent": 1,
            "improvement_method": "weaker",
            "user_mode": "interactive",
            "llm_user_persona": "You are a 20 year old college student majoring in computer science at UPenn.",
            "egs_answer_cache": {}  # Cache for EGS computation: {(agent_idx, question): answer}
        }
    
    state = st.session_state.llm_training_state
    
    # Load Previous Experiment Section
    st.markdown("---")
    st.subheader("📂 Load Previous Experiment")
    
    with st.expander("Load a previously saved experiment", expanded=False):
        # Use absolute path relative to project root (where streamlit is typically run from)
        import os
        from pathlib import Path
        # Get project root (parent of streamlit directory)
        project_root = Path(__file__).parent.parent
        save_dir = str(project_root / "out" / "llm_competition")
        saved_experiments = list_saved_experiments(save_dir=save_dir)
        
        if not saved_experiments:
            st.info("No saved experiments found in the `out/llm_competition/` directory.")
        else:
            st.info(f"Found {len(saved_experiments)} saved experiment(s).")
            
            # Create selection dropdown with experiment parameters
            experiment_options = {}
            for exp in saved_experiments:
                timestamp = exp["timestamp"]
                # Format timestamp nicely
                try:
                    from datetime import datetime
                    dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                    formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    formatted_time = timestamp
                
                # Build descriptive label with parameters
                parts = []
                n_agents = exp.get("n_agents", "unknown")
                if n_agents != "unknown" and n_agents is not None:
                    parts.append(f"{n_agents} agents")
                improvement_method = exp.get("improvement_method", "unknown")
                if improvement_method != "unknown" and improvement_method is not None:
                    parts.append(improvement_method)
                user_mode = exp.get("user_mode", "unknown")
                if user_mode != "unknown" and user_mode is not None:
                    mode_display = "🤖 Simulated" if user_mode == "simulated" else "👤 Interactive"
                    parts.append(mode_display)
                n_questions = exp.get("n_questions_per_pair", "unknown")
                if n_questions != "unknown" and n_questions is not None:
                    parts.append(f"{n_questions} questions")
                
                # Show which files are available
                files_status = []
                if exp["has_population"]:
                    files_status.append("Population")
                if exp["has_matrix"]:
                    files_status.append("EGS Matrix")
                if exp["has_prefs"]:
                    files_status.append("Preferences")
                
                params_str = " | ".join(parts) if parts else "Unknown params"
                label = f"{formatted_time} | {params_str} | ({', '.join(files_status)})"
                experiment_options[label] = exp["base_name"]
            
            selected_label = st.selectbox(
                "Select an experiment to load:",
                options=list(experiment_options.keys()),
                key="load_experiment_select"
            )
            
            col_load1, col_load2 = st.columns([3, 1])
            with col_load1:
                if selected_label:
                    selected_base_name = experiment_options[selected_label]
                    st.caption(f"Base name: `{selected_base_name}`")
            
            with col_load2:
                if st.button("📥 Load Experiment", type="primary", use_container_width=True, key="load_experiment_btn"):
                    with st.spinner("Loading experiment..."):
                        try:
                            # Use absolute path
                            import os
                            from pathlib import Path
                            project_root = Path(__file__).parent.parent
                            save_dir = str(project_root / "out" / "llm_competition")
                            result = load_experiment_results(selected_base_name, save_dir=save_dir)
                            
                            if result["loaded_successfully"]:
                                # Populate state with loaded data
                                state["population"] = result["population"]
                                state["egs_matrix"] = result["egs_matrix"]
                                
                                # Recreate game with loaded preferences
                                loaded_prefs = result["user_prefs"]
                                game = LLMCompetition(user_prefs=loaded_prefs)
                                state["game"] = game
                                
                                # Store metadata, visualization files, and experiment info
                                state["loaded_experiment_metadata"] = result["metadata"]
                                state["loaded_experiment_name"] = selected_base_name
                                state["loaded_experiment_viz_files"] = result.get("visualization_files", [])
                                state["loaded_experiment_info"] = result.get("experiment_info")
                                
                                # Mark as loaded (not initialized for training, but ready for viewing)
                                state["experiment_loaded"] = True
                                state["initialized"] = False  # Don't allow training on loaded experiment
                                
                                st.success(f"✅ **Experiment loaded successfully!**\n\n"
                                          f"**Loaded:** {len(result['population'])} agents, "
                                          f"EGS matrix {result['egs_matrix'].shape[0]}×{result['egs_matrix'].shape[1]}")
                                st.rerun()
                            else:
                                missing = ", ".join(result.get("missing_files", []))
                                error_msg = result.get("error", "Unknown error")
                                st.error(f"❌ **Failed to load experiment:**\n\n"
                                        f"Missing files: {missing}\n"
                                        f"Error: {error_msg}")
                        except Exception as e:
                            st.error(f"❌ **Error loading experiment:** {str(e)}")
                            import traceback
                            with st.expander("Error Details"):
                                st.code(traceback.format_exc())
    
    # Show loaded experiment info if one is loaded
    if state.get("experiment_loaded", False):
        st.markdown("---")
        st.info(f"📂 **Viewing loaded experiment:** `{state.get('loaded_experiment_name', 'unknown')}`")
        
        if state.get("loaded_experiment_metadata"):
            metadata = state["loaded_experiment_metadata"]
            col_meta1, col_meta2, col_meta3 = st.columns(3)
            with col_meta1:
                st.metric("Number of Agents", metadata.get("n_agents", "N/A"))
            with col_meta2:
                st.metric("Timestamp", metadata.get("timestamp", "N/A"))
            with col_meta3:
                if "gamescape_stats" in metadata:
                    st.metric("EGS Mean", f"{metadata['gamescape_stats'].get('mean', 0):.3f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Configuration")
        n_agents = st.slider(
            "Number of Agents",
            min_value=2,
            max_value=20,
            value=state.get("n_agents", 10),
            help="Total number of agents to train",
            key="n_agents_slider",
            disabled=state.get("initialized", False)
        )
        state["n_agents"] = n_agents
        
        # Allow changing games per agent even during training (affects future improvements)
        n_games_per_agent = st.slider(
            "Games per Agent Improvement",
            min_value=1,
            max_value=5,
            value=state.get("n_games_per_agent", 1),
            help="Number of games to play before improving an agent",
            key="n_games_per_agent_slider"
        )
        state["n_games_per_agent"] = n_games_per_agent
        
        improvement_method = st.selectbox(
            "PSRO Improvement Method",
            ["weaker", "stronger", "uniform"],
            index=["weaker", "stronger", "uniform"].index(state.get("improvement_method", "weaker")),
            help="Method for selecting opponents during training"
        )
        state["improvement_method"] = improvement_method
        
        # User preference mode
        user_mode_options = {
            "interactive": "Interactive (You provide feedback)",
            "simulated_feature": "Simulated (Feature Vector)",
            "simulated_llm": "Simulated (LLM with Persona)"
        }
        
        current_mode = state.get("user_mode", "interactive")
        user_mode = st.selectbox(
            "User Evaluation Mode",
            options=list(user_mode_options.keys()),
            format_func=lambda x: user_mode_options[x],
            index=list(user_mode_options.keys()).index(current_mode) if current_mode in user_mode_options else 0,
            help="Choose how user preferences are determined: Interactive (you provide feedback), Simulated Feature Vector (uses preference scores), or Simulated LLM (uses an LLM with a persona)"
        )
        state["user_mode"] = user_mode
        
        # LLM persona configuration (only shown if LLM mode is selected)
        if user_mode == "simulated_llm":
            default_persona = state.get("llm_user_persona", "You are a 20 year old college student majoring in computer science at UPenn.")
            llm_persona = st.text_area(
                "LLM User Persona",
                value=default_persona,
                height=100,
                help="Describe the persona of the simulated user. This will be used as a prompt for the LLM to evaluate answers."
            )
            state["llm_user_persona"] = llm_persona
        
        # Number of questions per pair for EGS computation
        n_questions_per_pair = st.slider(
            "Questions per Agent Pair (EGS Computation)",
            min_value=1,
            max_value=10,
            value=state.get("n_questions_per_pair", 5),
            help="Number of questions to evaluate per agent pair when computing the empirical gamescape (EGS) matrix. This setting only affects EGS computation, not the training phase. More questions = more accurate EGS but slower computation."
        )
        state["n_questions_per_pair"] = n_questions_per_pair
    
    with col2:
        st.subheader("Run Training")
        
        # Initialize Training Button
        init_button_disabled = (
            state.get("initializing", False) or 
            state.get("waiting_for_feedback", False) or 
            state.get("initialized", False) or
            state.get("generating_answers", False) or
            state.get("improving_agent", False)
        )
        
        if st.button("🚀 Initialize Training", type="primary", use_container_width=True, 
                    disabled=init_button_disabled, key="init_training_btn"):
            if not state["initialized"]:
                # Set flag and immediately show feedback
                state["initializing"] = True
                st.rerun()
            else:
                st.warning("⚠️ Training already initialized. Use 'Reset Training' to start over.")
        
        # Reset Training Button
        reset_button_disabled = (
            state.get("initializing", False) or 
            state.get("waiting_for_feedback", False) or 
            state.get("generating_answers", False) or
            state.get("improving_agent", False) or
            state.get("processing_feedback", False) or
            not state.get("initialized", False)
        )
        
        if st.button("🔄 Reset Training", use_container_width=True, 
                    disabled=reset_button_disabled, key="reset_training_btn"):
            # Reset all state
            state["initialized"] = False
            state["population"] = []
            state["game"] = None
            state["current_agent_idx"] = 0
            state["current_game_idx"] = 0
            state["current_question"] = None
            state["current_answer_a"] = None
            state["current_answer_b"] = None
            state["current_agent_a_idx"] = None
            state["current_agent_b_idx"] = None
            state["training_history"] = []
            state["egs_matrix"] = None
            state["waiting_for_feedback"] = False
            state["current_transcript"] = []
            state["generating_answers"] = False
            state["improving_agent"] = False
            state["computing_egs"] = False
            state["feedback_given"] = False
            state["initializing"] = False
            state["processing_feedback"] = False
            state["user_choice"] = None
            state["egs_answer_cache"] = {}  # Clear cache on reset
            state["answer_cache"] = {}  # Clear answer cache
            state["preference_cache"] = {}  # Clear preference cache
            state["fixed_questions"] = None
            state["collecting_preferences"] = False
            state["preference_collection_queue"] = []
            state["current_preference_comparison"] = None
            state["user_mode"] = "interactive"
            state["llm_user_persona"] = "You are a 20 year old college student majoring in computer science at UPenn."
            st.success("✅ Training reset successfully!")
            st.rerun()
        
        # Handle initialization - run immediately when flag is set
        if state.get("initializing", False) and not state.get("initialized", False):
            # Show status immediately
            status_placeholder = st.empty()
            status_placeholder.info("🔄 **Initializing training...** Please wait.")
            
            with st.spinner("Setting up game environment and initializing agents..."):
                try:
                    # Initialize game
                    status_placeholder.info("🔄 **Step 1/4:** Creating game instance...")
                    game = LLMCompetition(seed=42)
                    state["game"] = game
                    
                    # Select fixed questions for the entire experiment
                    status_placeholder.info("🔄 **Step 2/4:** Selecting fixed questions for experiment...")
                    import random
                    n_questions = state.get("n_questions_per_pair", 3)
                    all_questions = game.questions
                    fixed_questions = random.sample(all_questions, min(n_questions, len(all_questions)))
                    state["fixed_questions"] = fixed_questions
                    state["question_seed"] = random.getstate()  # Save seed for reproducibility
                    
                    # Get initial strategies
                    status_placeholder.info("🔄 **Step 3/4:** Generating initial agent strategies...")
                    p1, p2 = game.get_default_strategies()
                    state["population"] = [p1, p2]
                    
                    # Initialize training state
                    status_placeholder.info("🔄 **Step 4/4:** Finalizing setup...")
                    state["initialized"] = True
                    state["current_game_idx"] = 0
                    state["current_question_idx"] = 0  # Track which question we're on in the fixed set
                    state["training_history"] = []
                    state["waiting_for_feedback"] = False
                    state["current_transcript"] = []
                    state["egs_matrix"] = None
                    state["generating_answers"] = False
                    state["improving_agent"] = False
                    state["computing_egs"] = False
                    state["feedback_given"] = False
                    state["initializing"] = False
                    state["processing_feedback"] = False
                    state["user_choice"] = None
                    state["egs_answer_cache"] = {}
                    state["egs_interactive_mode"] = False
                    
                    # Initialize caches for proper PSRO
                    state["answer_cache"] = {}  # {(agent_idx, question): answer}
                    state["preference_cache"] = {}  # {(agent_i, agent_j, question): user_choice}
                    
                    # State for collecting preferences after agent improvement
                    state["collecting_preferences"] = False
                    state["preference_collection_queue"] = []  # List of (agent_i, agent_j, question) tuples
                    state["current_preference_comparison"] = None
                    
                    # Generate answers for initial agents on all fixed questions
                    fixed_questions = state.get("fixed_questions", [])
                    if fixed_questions:
                        status_placeholder.info("🔄 **Generating answers for initial agents...**")
                        _generate_answers_for_agent(state, 0)
                        _generate_answers_for_agent(state, 1)
                    
                    # Add first new agent to train (copy the last one as starting point)
                    if len(state["population"]) < state["n_agents"]:
                        new_agent = state["population"][-1]
                        state["population"].append(new_agent)
                        agent_idx = len(state["population"]) - 1
                        # For first agent, just use uniform selection (no preferences cached yet)
                        import random
                        opponent_idx = random.choice([0, 1])  # Choose from initial 2 agents
                        state["current_agent_a_idx"] = agent_idx
                        state["current_agent_b_idx"] = opponent_idx
                        # Auto-start first game
                        state["generating_answers"] = True
                    
                    status_placeholder.success("✅ **Training initialized successfully!** Starting first game...")
                    time.sleep(0.5)  # Brief pause to show success message
                    st.rerun()
                except Exception as e:
                    state["initializing"] = False
                    status_placeholder.error(f"❌ **Error initializing training:** {str(e)}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
    
    # Training Progress
    if state["initialized"]:
        st.subheader("Training Progress")
        # Count how many agents have been fully trained (improved at least once)
        trained_agent_indices = set()
        training_history = state.get("training_history", [])
        for hist in training_history:
            agent_idx = hist.get("agent_idx")
            if agent_idx is not None:
                trained_agent_indices.add(agent_idx)
        trained_count = len(trained_agent_indices)
        
        # Total agents to train: all agents except the initial 2
        # We start with 2 agents (indices 0 and 1), then train agents starting from index 2
        # So we need to train: n_agents - 2 agents
        total_agents = state["n_agents"]
        current_population_size = len(state.get("population", []))
        total_to_train = max(0, total_agents - 2)  # Total we need to train
        
        # Calculate progress based on agents that have completed training
        if total_to_train <= 0:
            progress = 1.0  # Already complete or invalid
        else:
            progress = min(trained_count / total_to_train, 1.0)
        
        st.progress(progress)
        
        # Show which agents have been trained (for debugging/transparency)
        if trained_agent_indices:
            trained_list = sorted(list(trained_agent_indices))
            trained_str = ", ".join([f"Agent {idx + 1}" for idx in trained_list])
            st.caption(f"**Trained agents ({trained_count}/{total_to_train}):** {trained_str} | **Population:** {current_population_size}/{total_agents} agents")
        else:
            st.caption(f"**Trained agents:** {trained_count}/{total_to_train} | **Population:** {current_population_size}/{total_agents} agents")
        
        # Training Status - Show what's happening
        st.markdown("---")
        st.subheader("Current Status")
        
        # Show processing feedback spinner and process the feedback
        if state.get("processing_feedback", False):
            if state.get("user_choice"):
                # Show prominent success message
                st.success("✅ **Your preference has been recorded!** Processing feedback now...")
                # Process the stored user choice
                user_choice = state["user_choice"]
                state["user_choice"] = None  # Clear it to prevent reprocessing
                # Process feedback - this will update state and trigger rerun
                try:
                    _process_user_feedback(state, user_choice)
                except Exception as e:
                    st.error(f"Error processing feedback: {e}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
                    # Reset state to allow retry
                    state["processing_feedback"] = False
                    state["feedback_given"] = False
                    state["waiting_for_feedback"] = True
                    st.rerun()
            else:
                # Processing feedback but no user_choice - might be stuck, reset
                st.warning("⚠️ Processing feedback but no user choice found. Resetting state.")
                state["processing_feedback"] = False
                state["feedback_given"] = False
                state["waiting_for_feedback"] = True
                st.rerun()
        
        # Show improving spinner if needed
        elif state.get("improving_agent", False):
            status_container = st.container()
            with status_container:
                st.info("🔄 **Improving agent strategy...**")
                with st.spinner("Calling optimizer LLM to refine strategy (this may take 30-60 seconds)..."):
                    _improve_agent_strategy(state)
        
        # Show generating answers status
        elif state.get("generating_answers", False):
            status_container = st.container()
            with status_container:
                st.info("🔄 **Generating answers from both agents...**")
                try:
                    with st.spinner("Calling LLM API to generate responses (this may take 30-60 seconds)..."):
                        _start_next_game(state)
                    
                    state["generating_answers"] = False
                    
                    # If simulated user mode, automatically process feedback without asking user
                    user_mode = state.get("user_mode", "interactive")
                    if user_mode in ["simulated_feature", "simulated_llm"] and state.get("waiting_for_feedback", False):
                        # Automatically simulate user choice
                        question = state.get("current_question")
                        answer_a = state.get("current_answer_a")
                        answer_b = state.get("current_answer_b")
                        if question and answer_a and answer_b:
                            if user_mode == "simulated_feature":
                                from games.llms.llm_competition import simulate_user_choice
                                user_choice = simulate_user_choice(
                                    answer_a, answer_b, 
                                    state["game"].user_prefs, 
                                    question
                                )
                            elif user_mode == "simulated_llm":
                                from games.llms.llm_competition import simulate_user_choice_llm
                                llm_persona = state.get("llm_user_persona", "You are a helpful user evaluating answers to questions.")
                                user_choice = simulate_user_choice_llm(
                                    answer_a, answer_b,
                                    question,
                                    llm_persona,
                                    "streamlit_training"
                                )
                            # Process feedback automatically
                            state["user_choice"] = user_choice
                            state["processing_feedback"] = True
                            state["feedback_given"] = True
                            st.rerun()
                    else:
                        # Interactive mode - rerun to show waiting_for_feedback UI
                        st.rerun()
                except Exception as e:
                    st.error(f"Error generating answers: {e}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
                    state["generating_answers"] = False
                    state["waiting_for_feedback"] = False
                    st.rerun()
        
        elif state.get("waiting_for_feedback") and state.get("user_mode", "interactive") == "interactive":
            # Check if we're collecting preferences (after agent improvement) or training
            collecting_preferences = state.get("collecting_preferences", False)
            
            # Only set up next comparison if we don't have one ready
            if collecting_preferences:
                # Check if we need to set up a new comparison
                has_current = state.get("current_question") and state.get("current_answer_a") and state.get("current_answer_b")
                if not has_current:
                    _setup_next_preference_comparison(state)
                    # If _setup_next_preference_comparison called st.rerun(), we'll return here
                    # Otherwise continue to show the UI
            
            # Show progress if collecting preferences
            if collecting_preferences:
                preference_queue = state.get("preference_collection_queue", [])
                preference_cache = state.get("preference_cache", {})
                completed = len([c for c in preference_queue if (c[0], c[1], c[2]) in preference_cache])
                total = len(preference_queue)
                st.info(f"**Progress:** {completed} / {total} comparisons completed")
            
            st.subheader("⏳ Waiting for Your Feedback")
            
            if state.get("current_question") and state.get("current_answer_a") and state.get("current_answer_b"):
                st.markdown(f"**Question:** {state['current_question']}")
                
                # Buttons at the top
                st.markdown("---")
                col_btn1, col_btn2, col_btn3 = st.columns(3)
                button_disabled = state.get("feedback_given", False) or state.get("processing_feedback", False)                
                with col_btn1:
                    if st.button("✅ Prefer Answer A", type="primary", use_container_width=True, 
                                key="prefer_a", disabled=button_disabled):                        _handle_user_choice(state, "A")
                
                with col_btn2:
                    if st.button("🤝 Tie / No Preference", use_container_width=True, 
                                key="prefer_tie", disabled=button_disabled):                        _handle_user_choice(state, "TIE")
                
                with col_btn3:
                    if st.button("✅ Prefer Answer B", type="primary", use_container_width=True, 
                                key="prefer_b", disabled=button_disabled):                        _handle_user_choice(state, "B")                
                st.markdown("---")
                
                # Answers displayed below buttons without boxes, using native markdown
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("### Answer A")
                    st.markdown(state["current_answer_a"])
                
                with col_b:
                    st.markdown("### Answer B")
                    st.markdown(state["current_answer_b"])
        
        elif len(state["population"]) < state["n_agents"]:
            # Auto-start next game if not waiting for feedback and not already generating
            if (not state.get("waiting_for_feedback", False) and 
                not state.get("generating_answers", False) and 
                not state.get("improving_agent", False) and
                not state.get("processing_feedback", False)):
                # Automatically start the next game
                state["generating_answers"] = True
                st.rerun()
        else:
            # Training complete
            st.success("🎉 Training Complete! All agents have been trained.")
            
            # Auto-compute EGS if not done yet
            if state.get("computing_egs", False):
                state["computing_egs"] = False
                user_mode = state.get("user_mode", "interactive")
                
                # Compute EGS from cached preferences
                egs_matrix = _compute_egs_from_cache(state)
                if egs_matrix is not None:
                    state["egs_matrix"] = egs_matrix
                    st.success("✅ EGS matrix computed from cached preferences!")
                    st.rerun()
                else:
                    # Some preferences missing - compute using standard method
                    if user_mode in ["simulated_feature", "simulated_llm"]:
                        # Simulated mode: compute automatically using refactored function
                        with st.spinner("Computing empirical gamescape matrix..."):
                            try:
                                from games.llms.llm_competition import compute_empirical_gamescape
                                
                                # Progress callback for UI
                                progress_placeholder = st.empty()
                                progress_bar_placeholder = st.empty()
                                
                                def progress_callback(msg, ratio):
                                    progress_placeholder.info(f"🔄 {msg}")
                                    progress_bar_placeholder.progress(ratio)
                                
                                egs_matrix = compute_empirical_gamescape(
                                    population=state["population"],
                                    game=state["game"],
                                    evaluator=None,  # Use game's default evaluator
                                    n_questions_per_pair=len(state.get("fixed_questions", [])),
                                    use_cache=True,
                                    progress_callback=progress_callback
                                )
                                
                                progress_placeholder.empty()
                                progress_bar_placeholder.empty()
                                state["egs_matrix"] = egs_matrix
                                st.success("✅ EGS matrix computed!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error computing EGS matrix: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
                    else:
                        # Interactive mode: initialize interactive EGS computation
                        state["egs_interactive_mode"] = True
                    state["egs_comparisons"] = []  # List of (i, j, question, answer_i, answer_j, user_choice)
                    state["egs_current_comparison"] = None
                    state["egs_comparison_idx"] = 0
                    st.info("🔄 **Interactive EGS Mode:** You'll be asked to compare agents. Starting...")
                    st.rerun()
            
            # Post-training features
            st.markdown("---")
            st.subheader("Post-Training Analysis")
            
            # Save Results Section
            st.markdown("#### 💾 Save Experiment Results")
            col_save1, col_save2 = st.columns([2, 1])
            with col_save1:
                st.info("💡 Save your experiment results to disk for later analysis. Results will be saved to the `out/llm_competition/` directory.")
            with col_save2:
                save_disabled = state.get("egs_matrix") is None
                if st.button("💾 Save Results", type="primary", use_container_width=True, 
                            disabled=save_disabled, key="save_results_btn"):
                    if state.get("egs_matrix") is None:
                        st.warning("⚠️ EGS matrix must be computed before saving. Please wait for it to complete.")
                    else:
                        with st.spinner("Saving experiment results..."):
                            try:
                                # Prepare experiment parameters
                                user_mode_str = state.get("user_mode", "interactive")
                                # Convert to old format for metadata compatibility
                                if user_mode_str == "interactive":
                                    user_mode = "interactive"
                                else:
                                    user_mode = "simulated"
                                experiment_params = {
                                    "n_agents": state["n_agents"],
                                    "n_games_per_agent": state["n_games_per_agent"],
                                    "improvement_method": state["improvement_method"],
                                    "n_questions_per_pair": state.get("n_questions_per_pair", 5),
                                    "user_mode": user_mode,  # Track simulated vs interactive
                                    "training_history_length": len(state.get("training_history", [])),
                                    "total_games_played": state.get("current_game_idx", 0)
                                }
                                
                                # Save results - use absolute path
                                project_root = Path(__file__).parent.parent
                                save_dir = str(project_root / "out" / "llm_competition")
                                
                                base_name = save_experiment_results(
                                    population=state["population"],
                                    egs_matrix=state["egs_matrix"],
                                    game=state["game"],
                                    experiment_params=experiment_params,
                                    save_dir=save_dir,
                                    prefix="llm_competition"
                                )
                                
                                # Also generate and save visualizations
                                try:
                                    visualize_gamescape(
                                        egs_matrix=state["egs_matrix"],
                                        save_dir=save_dir,
                                        prefix=base_name
                                    )
                                    # Extract subfolder from base_name if present
                                    if "/" in base_name:
                                        subfolder, filename = base_name.rsplit("/", 1)
                                        save_location = f"`out/llm_competition/{subfolder}/`"
                                    else:
                                        save_location = "`out/llm_competition/`"
                                    
                                    st.success(f"✅ **Results saved successfully!**\n\n"
                                              f"**Base name:** `{base_name}`\n\n"
                                              f"**Files saved to {save_location}:**\n"
                                              f"- `{base_name.split('/')[-1]}_population.pkl` - All agent strategies\n"
                                              f"- `{base_name.split('/')[-1]}_egs_matrix.npy` - Gamescape matrix\n"
                                              f"- `{base_name.split('/')[-1]}_user_prefs.json` - User preferences\n"
                                              f"- `{base_name.split('/')[-1]}_metadata.json` - Experiment metadata\n"
                                              f"- `{base_name.split('/')[-1]}_experiment_info.txt` - Detailed experiment information\n"
                                              f"- `{base_name.split('/')[-1]}_egs_*.png` - Visualization plots")
                                except Exception as viz_error:
                                    # Extract subfolder from base_name if present
                                    if "/" in base_name:
                                        subfolder, filename = base_name.rsplit("/", 1)
                                        save_location = f"`out/llm_competition/{subfolder}/`"
                                    else:
                                        save_location = "`out/llm_competition/`"
                                    
                                    st.success(f"✅ **Results saved successfully!**\n\n"
                                             f"**Base name:** `{base_name}`\n\n"
                                             f"**Files saved to {save_location}:**\n"
                                             f"- `{base_name.split('/')[-1]}_population.pkl` - All agent strategies\n"
                                             f"- `{base_name.split('/')[-1]}_egs_matrix.npy` - Gamescape matrix\n"
                                             f"- `{base_name.split('/')[-1]}_user_prefs.json` - User preferences\n"
                                             f"- `{base_name.split('/')[-1]}_metadata.json` - Experiment metadata\n"
                                             f"- `{base_name.split('/')[-1]}_experiment_info.txt` - Detailed experiment information")
                                    st.warning(f"⚠️ Visualizations could not be generated: {str(viz_error)}")
                                
                            except Exception as e:
                                st.error(f"❌ **Error saving results:** {str(e)}")
                                import traceback
                                with st.expander("Error Details"):
                                    st.code(traceback.format_exc())
            
            st.markdown("---")
            
            # Display all trained agents
            with st.expander("📋 All Trained Agents", expanded=True):
                for idx, agent_strategy in enumerate(state["population"]):
                    st.markdown(f"**Agent {idx + 1}:**")
                    st.text(agent_strategy)
                    st.markdown("---")
            
            # Training history
            if state["training_history"]:
                with st.expander("📜 Training History", expanded=False):
                    for hist_item in state["training_history"]:
                        st.json(hist_item)
            
            # Handle interactive EGS computation
            if state.get("egs_interactive_mode", False) and state["egs_matrix"] is None:
                _handle_interactive_egs(state)
            
            # Display EGS matrix if computed (either from training or loaded experiment)
            if state.get("egs_matrix") is not None:
                # Check if we have saved visualization files from a loaded experiment
                loaded_viz_files = state.get("loaded_experiment_viz_files", [])
                show_saved_viz = False
                selected_viz = None
                
                st.markdown("#### Empirical Gamescape Matrix")
                
                # If we have saved visualization files, offer to display them
                if loaded_viz_files and state.get("experiment_loaded", False):
                    st.info(f"💡 Found {len(loaded_viz_files)} saved visualization(s) for this experiment.")
                    viz_methods = {viz["method"]: viz["path"] for viz in loaded_viz_files}
                    
                    selected_viz = st.selectbox(
                        "View saved visualization:",
                        options=["None"] + list(viz_methods.keys()),
                        key="view_saved_viz"
                    )
                    
                    if selected_viz != "None" and selected_viz in viz_methods:
                        show_saved_viz = True
                        viz_path = viz_methods[selected_viz]
                        try:
                            with open(viz_path, 'rb') as f:
                                img_data = base64.b64encode(f.read()).decode()
                            st.markdown(
                                f'<img src="data:image/png;base64,{img_data}" style="max-width: 100%;" />',
                                unsafe_allow_html=True
                            )
                        except Exception as e:
                            st.error(f"Error loading visualization: {e}")
                            show_saved_viz = False
                
                # If no saved visualizations or user wants to see matrix only, show matrix heatmap
                if not show_saved_viz:
                    # Display matrix heatmap
                    fig, ax = plt.subplots(figsize=(10, 8))
                    matrix = state["egs_matrix"]
                    vmax = np.abs(matrix).max()
                    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', 
                                  vmin=-vmax, vmax=vmax, interpolation='nearest')
                    ax.set_title('Empirical Gamescape Matrix\n(Green: Positive, Red: Negative)')
                    ax.set_xlabel('Agent j')
                    ax.set_ylabel('Agent i')
                    plt.colorbar(im, ax=ax, label='Payoff')
                    plt.tight_layout()
                    
                    # Convert to base64 for display
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                    buf.seek(0)
                    img_data = base64.b64encode(buf.read()).decode()
                    st.markdown(
                        f'<img src="data:image/png;base64,{img_data}" style="max-width: 100%;" />',
                        unsafe_allow_html=True
                    )
                    plt.close(fig)
                
                # EGS embeddings visualization
                st.markdown("#### EGS Embeddings Visualization")
                embedding_method = st.selectbox(
                    "Embedding Method",
                    ["schur", "PCA", "SVD", "tSNE"],
                    index=0
                )
                
                if st.button("📈 Generate Embedding Visualization"):
                    with st.spinner(f"Generating {embedding_method} embeddings..."):
                        try:
                            from games.egs import visualize_egs_matrix_and_embeddings
                            
                            egs = EmpiricalGS(state["egs_matrix"])
                            
                            if embedding_method == "schur":
                                embeddings = egs.schur_embeddings()
                            elif embedding_method == "PCA":
                                embeddings = egs.PCA_embeddings()
                            elif embedding_method == "SVD":
                                embeddings = egs.SVD_embeddings()
                            elif embedding_method == "tSNE":
                                if state["egs_matrix"].shape[0] > 50:
                                    st.warning("t-SNE may be slow for large populations. Using PCA instead.")
                                    embeddings = egs.PCA_embeddings()
                                else:
                                    embeddings = egs.tSNE_embeddings()
                            
                            # Use the existing visualization function directly
                            # Save to a temporary file, then read it for display
                            import tempfile
                            import os
                            
                            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                                tmp_path = tmp_file.name
                            
                            # Call the function to generate and save the visualization
                            visualize_egs_matrix_and_embeddings(egs, embeddings, save_path=tmp_path)
                            
                            # Read the saved file and convert to base64
                            with open(tmp_path, 'rb') as f:
                                img_data = base64.b64encode(f.read()).decode()
                            
                            # Clean up temp file
                            os.unlink(tmp_path)
                            
                            st.markdown(
                                f'<img src="data:image/png;base64,{img_data}" style="max-width: 100%;" />',
                                unsafe_allow_html=True
                            )
                            
                        except Exception as e:
                            st.error(f"Error generating embeddings: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
    
    # Display loaded experiment data (if experiment was loaded but not initialized for training)
    elif state.get("experiment_loaded", False) and not state.get("initialized", False):
        st.markdown("---")
        st.subheader("📂 Loaded Experiment Data")
        
        # Display experiment metadata prominently
        if state.get("loaded_experiment_metadata"):
            metadata = state["loaded_experiment_metadata"]
            with st.expander("📊 Experiment Parameters", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Number of Agents", metadata.get("n_agents", "Unknown"))
                    st.metric("User Mode", 
                             "🤖 Simulated" if metadata.get("user_mode") == "simulated" else "👤 Interactive",
                             help="Whether this experiment used simulated or interactive user preferences")
                
                with col2:
                    exp_params = metadata.get("experiment_params", {})
                    st.metric("Improvement Method", exp_params.get("improvement_method", "Unknown"))
                    st.metric("Games per Agent", exp_params.get("n_games_per_agent", "Unknown"))
                
                with col3:
                    st.metric("Questions per Pair", exp_params.get("n_questions_per_pair", "Unknown"))
                    timestamp = metadata.get("timestamp", "Unknown")
                    try:
                        from datetime import datetime
                        dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                        formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                        st.metric("Run Date", formatted_time)
                    except:
                        st.metric("Run Date", timestamp)
                
                # Show subfolder info
                if metadata.get("subfolder"):
                    st.info(f"📁 **Experiment Folder:** `{metadata['subfolder']}`")
        
        # Display experiment info file if available
        if state.get("loaded_experiment_info"):
            with st.expander("📄 Full Experiment Information", expanded=False):
                # Display the info file content in a code block for better formatting
                # This preserves the structure and formatting of the text file
                st.code(state["loaded_experiment_info"], language="text")
        elif state.get("loaded_experiment_metadata"):
            # If info file doesn't exist, show a note
            st.info("💡 Experiment info file not found. This may be an older experiment.")
        
        # Display all agents from loaded experiment
        if state.get("population"):
            with st.expander("📋 All Agents", expanded=True):
                for idx, agent_strategy in enumerate(state["population"]):
                    st.markdown(f"**Agent {idx + 1}:**")
                    st.text(agent_strategy)
                    st.markdown("---")
        
        # Display metadata if available
        if state.get("loaded_experiment_metadata"):
            metadata = state["loaded_experiment_metadata"]
            with st.expander("📊 Experiment Metadata", expanded=False):
                st.json(metadata)
        
        # Display EGS matrix if available
        if state.get("egs_matrix") is not None:
            st.markdown("---")
            # Check if we have saved visualization files from a loaded experiment
            loaded_viz_files = state.get("loaded_experiment_viz_files", [])
            show_saved_viz = False
            selected_viz = None
            
            st.markdown("#### Empirical Gamescape Matrix")
            
            # If we have saved visualization files, offer to display them
            if loaded_viz_files:
                st.info(f"💡 Found {len(loaded_viz_files)} saved visualization(s) for this experiment.")
                viz_methods = {viz["method"]: viz["path"] for viz in loaded_viz_files}
                
                selected_viz = st.selectbox(
                    "View saved visualization:",
                    options=["None"] + list(viz_methods.keys()),
                    key="view_saved_viz_loaded"
                )
                
                if selected_viz != "None" and selected_viz in viz_methods:
                    show_saved_viz = True
                    viz_path = viz_methods[selected_viz]
                    try:
                        with open(viz_path, 'rb') as f:
                            img_data = base64.b64encode(f.read()).decode()
                        st.markdown(
                            f'<img src="data:image/png;base64,{img_data}" style="max-width: 100%;" />',
                            unsafe_allow_html=True
                        )
                    except Exception as e:
                        st.error(f"Error loading visualization: {e}")
                        show_saved_viz = False
            
            # If no saved visualizations or user wants to see matrix only, show matrix heatmap
            if not show_saved_viz:
                # Display matrix heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                matrix = state["egs_matrix"]
                vmax = np.abs(matrix).max()
                im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', 
                              vmin=-vmax, vmax=vmax, interpolation='nearest')
                ax.set_title('Empirical Gamescape Matrix\n(Green: Positive, Red: Negative)')
                ax.set_xlabel('Agent j')
                ax.set_ylabel('Agent i')
                plt.colorbar(im, ax=ax, label='Payoff')
                plt.tight_layout()
                
                # Convert to base64 for display
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                img_data = base64.b64encode(buf.read()).decode()
                st.markdown(
                    f'<img src="data:image/png;base64,{img_data}" style="max-width: 100%;" />',
                    unsafe_allow_html=True
                )
                plt.close(fig)
            
            # EGS embeddings visualization
            st.markdown("#### EGS Embeddings Visualization")
            embedding_method = st.selectbox(
                "Embedding Method",
                ["schur", "PCA", "SVD", "tSNE"],
                index=0,
                key="embedding_method_loaded"
            )
            
            if st.button("📈 Generate Embedding Visualization", key="generate_embedding_loaded"):
                with st.spinner(f"Generating {embedding_method} embeddings..."):
                    try:
                        from games.egs import visualize_egs_matrix_and_embeddings
                        
                        egs = EmpiricalGS(state["egs_matrix"])
                        
                        if embedding_method == "schur":
                            embeddings = egs.schur_embeddings()
                        elif embedding_method == "PCA":
                            embeddings = egs.PCA_embeddings()
                        elif embedding_method == "SVD":
                            embeddings = egs.SVD_embeddings()
                        elif embedding_method == "tSNE":
                            if state["egs_matrix"].shape[0] > 50:
                                st.warning("t-SNE may be slow for large populations. Using PCA instead.")
                                embeddings = egs.PCA_embeddings()
                            else:
                                embeddings = egs.tSNE_embeddings()
                        
                        # Use the existing visualization function directly
                        # Save to a temporary file, then read it for display
                        import tempfile
                        import os
                        
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                            tmp_path = tmp_file.name
                        
                        # Call the function to generate and save the visualization
                        visualize_egs_matrix_and_embeddings(egs, embeddings, save_path=tmp_path)
                        
                        # Read the saved file and convert to base64
                        with open(tmp_path, 'rb') as f:
                            img_data = base64.b64encode(f.read()).decode()
                        
                        # Clean up temp file
                        os.unlink(tmp_path)
                        
                        st.markdown(
                            f'<img src="data:image/png;base64,{img_data}" style="max-width: 100%;" />',
                            unsafe_allow_html=True
                        )
                        
                    except Exception as e:
                        st.error(f"Error generating embeddings: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
    
    else:
        st.info("👆 Click 'Initialize Training' to begin.")


def _start_next_game(state):
    """Start the next game - uses fixed questions and cached answers when available."""
    import random
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from games.llms.llm_competition import COMPETITION_GAME_PROMPT, call_model
    
    if not state["initialized"] or not state["game"]:
        return
    
    population = state["population"]
    if len(population) < 2:
        st.error("Not enough agents!")
        return
    
    # Select agent to train (last one) and opponent using PSRO method
    agent_idx = len(population) - 1
    opponent_idx = _select_opponent_psro(state, agent_idx)    
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
    state["feedback_given"] = False  # Reset feedback flag for new game
    state["current_game_idx"] += 1


def _setup_next_preference_comparison(state):
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
            opponent_idx = _select_opponent_psro(state, agent_idx)
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
        # All done, clear and continue
        state["collecting_preferences"] = False
        state["preference_collection_queue"] = []
        state["waiting_for_feedback"] = False
        st.rerun()
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
    # Don't call st.rerun() here - let the UI render naturally


def _handle_user_choice(state, user_choice):
    """Handle user choice - either for training feedback or preference collection."""
    collecting_preferences = state.get("collecting_preferences", False)
    if collecting_preferences:
        # Store in preference cache
        agent_i = state.get("current_agent_a_idx")
        agent_j = state.get("current_agent_b_idx")
        question = state.get("current_question")
        cache_key = (agent_i, agent_j, question)
        
        preference_cache = state.get("preference_cache", {})
        preference_cache[cache_key] = user_choice
        state["preference_cache"] = preference_cache
        
        # Clear current comparison so _setup_next_preference_comparison will load the next one
        state["current_question"] = None
        state["current_answer_a"] = None
        state["current_answer_b"] = None
        state["feedback_given"] = False
        st.rerun()
    else:
        # Training feedback - process normally
        state["feedback_given"] = True
        state["processing_feedback"] = True
        state["user_choice"] = user_choice
        st.rerun()


def _process_user_feedback(state, user_choice):
    """Process user feedback and continue training."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from games.llms.llm_competition import get_opt_prompt, OPT_SYSTEM_PROMPT, COMPETITION_GAME_PROMPT
    import google.generativeai as genai
    import os
    
    # Check if we have the necessary state to process feedback
    # We need at least a question and answers
    if not state.get("current_question") or not state.get("current_answer_a") or not state.get("current_answer_b"):
        # Missing required data, can't process
        st.warning("⚠️ Missing required data to process feedback. Resetting state.")
        state["processing_feedback"] = False
        state["waiting_for_feedback"] = False
        state["feedback_given"] = False
        # Try to continue with next game
        if len(state.get("current_transcript", [])) < state.get("n_games_per_agent", 1):
            state["generating_answers"] = True
        st.rerun()
        return
    
    # Check if we've already processed this specific feedback by checking the transcript
    # This prevents double-processing if the function is called multiple times
    current_q = state.get("current_question")
    current_transcript = state.get("current_transcript", [])
    if current_q and current_transcript:
        # Check if this exact question-answer pair is already in transcript
        current_answer_a = state.get("current_answer_a")
        current_answer_b = state.get("current_answer_b")
        already_processed = any(
            entry[0] == current_q and entry[1] == current_answer_a and entry[2] == current_answer_b
            for entry in current_transcript
        )
        if already_processed:
            # Already processed this exact feedback, just update state and continue
            if len(current_transcript) >= state["n_games_per_agent"]:
                state["improving_agent"] = True
                state["waiting_for_feedback"] = False
                state["processing_feedback"] = False
                st.rerun()
                return
            else:
                state["waiting_for_feedback"] = False
                state["processing_feedback"] = False
                state["generating_answers"] = True
                state["feedback_given"] = False
                st.rerun()
                return
    
    # Mark feedback as given immediately to prevent double-clicks
    state["feedback_given"] = True
    
    # Calculate payout (1 = A wins, -1 = B wins, 0 = tie)
    payout = 1 if user_choice == "A" else (-1 if user_choice == "B" else 0)
    
    # Add to transcript
    transcript_entry = (
        state["current_question"],
        state["current_answer_a"],
        state["current_answer_b"],
        payout
    )
    
    if "current_transcript" not in state:
        state["current_transcript"] = []
    state["current_transcript"].append(transcript_entry)
    
    agent_idx = state["current_agent_a_idx"]
    opponent_idx = state["current_agent_b_idx"]
    
    # Also store in preference cache for PSRO and EGS computation
    question = state["current_question"]
    preference_cache = state.get("preference_cache", {})
    cache_key = (agent_idx, opponent_idx, question)
    preference_cache[cache_key] = user_choice
    state["preference_cache"] = preference_cache
    
    # Clear processing flag before deciding next step
    state["processing_feedback"] = False
    
    # Check if we've played enough games to improve the agent
    games_played = len(state["current_transcript"])
    games_needed = state["n_games_per_agent"]
    
    if games_played >= games_needed:
        # Need to improve agent - set flag and rerun to show spinner
        state["improving_agent"] = True
        state["waiting_for_feedback"] = False
        st.rerun()
    else:
        # Not enough games yet - continue with next game
        state["waiting_for_feedback"] = False
        state["generating_answers"] = True
        state["feedback_given"] = False  # Reset for next game
        st.rerun()


def _generate_answers_for_agent(state, agent_idx):
    """Generate answers for an agent on all fixed questions and cache them."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from games.llms.llm_competition import COMPETITION_GAME_PROMPT, call_model
    
    fixed_questions = state.get("fixed_questions", [])
    answer_cache = state.get("answer_cache", {})
    population = state["population"]
    
    for question in fixed_questions:
        cache_key = (agent_idx, question)
        if cache_key not in answer_cache:
            full_prompt = f"{COMPETITION_GAME_PROMPT}\n\n{population[agent_idx]}\n\nQuestion: {question}\n\nProvide your answer:"
            answer = call_model(full_prompt, f"agent_{agent_idx}_q_{fixed_questions.index(question)}")
            answer_cache[cache_key] = answer
    
    state["answer_cache"] = answer_cache


def _compute_agent_scores(state, agent_i, agent_j):
    """
    Compute average score for agent_i against agent_j across all fixed questions.
    Returns average score (positive = i beats j, negative = j beats i, 0 = tie).
    """
    fixed_questions = state.get("fixed_questions", [])
    preference_cache = state.get("preference_cache", {})
    
    scores = []
    for question in fixed_questions:
        cache_key = (agent_i, agent_j, question)
        if cache_key in preference_cache:
            user_choice = preference_cache[cache_key]
            # Convert to score: "A" (i wins) = 1, "B" (j wins) = -1, "TIE" = 0
            if user_choice == "A":
                scores.append(1)
            elif user_choice == "B":
                scores.append(-1)
            else:  # TIE
                scores.append(0)
    
    if not scores:
        return 0.0  # No preferences cached yet
    
    return sum(scores) / len(scores)


def _select_opponent_psro(state, agent_idx):
    """
    Select opponent for agent_idx based on PSRO method (uniform, weaker, or stronger).
    Uses cached preferences to determine weaker/stronger agents.
    Falls back to uniform if no preferences are cached yet.
    """
    import random
    import numpy as np
    
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
        score = _compute_agent_scores(state, agent_idx, opponent_idx)
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
        # Fallback to uniform if no weaker agents
        return random.choice(available_indices)
    
    elif improvement_method == "stronger":
        # Select from agents that beat agent_idx (score < 0)
        stronger_indices = [idx for idx, score in agent_scores.items() if score < 0]
        if stronger_indices:
            return random.choice(stronger_indices)
        # Fallback to uniform if no stronger agents
        return random.choice(available_indices)
    
    # Default fallback
    return random.choice(available_indices)


def _improve_agent_strategy(state):
    """Improve agent strategy based on collected transcript."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from games.llms.llm_competition import improve_strategy, COMPETITION_GAME_PROMPT
    
    if not state.get("improving_agent", False):
        return
    
    agent_idx = state["current_agent_a_idx"]
    opponent_idx = state["current_agent_b_idx"]
    
    # Improve the agent using optimizer
    u_prompt = state["population"][agent_idx]
    v_prompt = state["population"][opponent_idx]
    
    # Use the refactored improve_strategy function
    transcript = state["current_transcript"]
    try:
        u_new = improve_strategy(u_prompt, v_prompt, transcript, COMPETITION_GAME_PROMPT)
    except Exception as e:
        st.warning(f"Error improving agent: {e}")
        u_new = u_prompt
    
    # Update agent
    state["population"][agent_idx] = u_new
    
    # Store history
    avg_payout = sum(t[3] for t in state["current_transcript"]) / len(state["current_transcript"])
    state["training_history"].append({
        "agent_idx": agent_idx,
        "opponent_idx": opponent_idx,
        "transcript": state["current_transcript"],
        "avg_payout": avg_payout,
        "old_strategy": u_prompt,
        "new_strategy": u_new
    })
    
    # Clear transcript
    state["current_transcript"] = []
    state["improving_agent"] = False
    state["waiting_for_feedback"] = False
    
    # After improving agent, generate answers for all fixed questions FIRST
    # This ensures all answers are ready before asking for preferences
    # Note: _generate_answers_for_agent already handles caching, so this should be fast if cached
    _generate_answers_for_agent(state, agent_idx)
    
    # If interactive mode, set up preference collection for new agent vs all existing agents
    if state.get("user_mode", "interactive") == "interactive":
        # Build queue of comparisons needed: (agent_idx, existing_agent_idx, question)
        preference_queue = []
        fixed_questions = state.get("fixed_questions", [])
        for existing_idx in range(agent_idx):  # Compare with all previous agents
            for question in fixed_questions:
                cache_key = (agent_idx, existing_idx, question)
                # Only add if not already cached
                if cache_key not in state.get("preference_cache", {}):
                    preference_queue.append((agent_idx, existing_idx, question))
        
        if preference_queue:
            state["preference_collection_queue"] = preference_queue
            state["collecting_preferences"] = True
            state["preference_comparisons_pending"] = {}  # Track which comparisons are pending
            st.rerun()
            return
    
    # If simulated mode, automatically generate preferences
    user_mode = state.get("user_mode", "interactive")
    if user_mode in ["simulated_feature", "simulated_llm"]:
        _generate_simulated_preferences_for_agent(state, agent_idx)
    
    # Continue with next agent or finish
    if len(state["population"]) < state["n_agents"]:
        new_agent = state["population"][-1]
        state["population"].append(new_agent)
        # Select opponent using PSRO method
        agent_idx = len(state["population"]) - 1
        opponent_idx = _select_opponent_psro(state, agent_idx)
        state["current_agent_a_idx"] = agent_idx
        state["current_agent_b_idx"] = opponent_idx
        # Auto-start next game
        state["generating_answers"] = True
    else:
        # Training complete - compute EGS automatically
        if state["egs_matrix"] is None:
            state["computing_egs"] = True
    
    st.rerun()


def _compute_egs_from_cache(state):
    """
    Compute EGS matrix from cached preferences.
    Returns None if some preferences are missing.
    """
    import numpy as np
    
    population = state["population"]
    n = len(population)
    fixed_questions = state.get("fixed_questions", [])
    preference_cache = state.get("preference_cache", {})
    
    if not fixed_questions:
        return None
    
    egs_matrix = np.zeros((n, n))
    missing_pairs = []
    
    # Check all pairs
    for i in range(n):
        for j in range(i + 1, n):
            payoffs = []
            all_cached = True
            
            for question in fixed_questions:
                cache_key = (i, j, question)
                if cache_key in preference_cache:
                    user_choice = preference_cache[cache_key]
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
    
    # If any preferences are missing, return None
    if missing_pairs:
        return None
    
    return egs_matrix


def _generate_simulated_preferences_for_agent(state, agent_idx):
    """Generate simulated user preferences for new agent against all existing agents."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from games.llms.llm_competition import simulate_user_choice, simulate_user_choice_llm
    
    fixed_questions = state.get("fixed_questions", [])
    answer_cache = state.get("answer_cache", {})
    preference_cache = state.get("preference_cache", {})
    game = state["game"]
    user_mode = state.get("user_mode", "interactive")
    
    for existing_idx in range(agent_idx):
        for question in fixed_questions:
            cache_key = (agent_idx, existing_idx, question)
            if cache_key not in preference_cache:
                # Get cached answers
                answer_i = answer_cache.get((agent_idx, question))
                answer_j = answer_cache.get((existing_idx, question))
                
                if answer_i and answer_j:
                    # Simulate user choice based on mode
                    if user_mode == "simulated_feature":
                        user_choice = simulate_user_choice(
                            answer_i, answer_j,
                            game.user_prefs,
                            question
                        )
                    else:  # simulated_llm
                        llm_persona = state.get("llm_user_persona", "You are a helpful user evaluating answers to questions.")
                        user_choice = simulate_user_choice_llm(
                            answer_i, answer_j,
                            question,
                            llm_persona,
                            f"simulate_prefs_{agent_idx}_{existing_idx}"
                        )
                    preference_cache[cache_key] = user_choice
    
    state["preference_cache"] = preference_cache




# NOTE: _compute_egs_matrix has been moved to llm_competition.py as compute_empirical_gamescape()
# This function is kept for backward compatibility but should not be used.
# Use compute_empirical_gamescape() from llm_competition.py instead.


def _handle_interactive_egs(state):
    """Handle interactive EGS computation where user provides feedback."""
    import numpy as np
    import random
    from games.llms.llm_competition import COMPETITION_GAME_PROMPT, call_model
    
    population = state["population"]
    game = state["game"]
    n = len(population)
    questions = game.questions
    
    # Initialize answer cache if needed
    answer_cache = state.get("egs_answer_cache", {})
    cache_key_prefix = f"egs_{len(population)}_"
    
    # Generate all answers if not cached
    if not answer_cache or len([k for k in answer_cache.keys() if k[0].startswith(cache_key_prefix)]) < n * len(questions):
        st.info("🔄 Generating answers for all agents...")
        progress_text = st.empty()
        
        for i in range(n):
            for question in questions:
                cache_key = (cache_key_prefix + str(i), question)
                if cache_key not in answer_cache:
                    progress_text.text(f"Generating answer for Agent {i+1}/{n}...")
                    full_prompt = f"{COMPETITION_GAME_PROMPT}\n\n{population[i]}\n\nQuestion: {question}\n\nProvide your answer:"
                    answer = call_model(full_prompt, f"egs_agent_{i}")
                    answer_cache[cache_key] = answer
        
        state["egs_answer_cache"] = answer_cache
        progress_text.empty()
        st.success("✅ All answers generated!")
    
    # Generate list of all comparisons needed
    if "egs_comparison_list" not in state:
        comparison_list = []
        n_questions_per_pair = state.get("n_questions_per_pair", 3)
        for i in range(n):
            for j in range(i + 1, n):
                # Use configured number of questions per pair
                eval_questions = random.sample(questions, min(n_questions_per_pair, len(questions)))
                for question in eval_questions:
                    comparison_list.append((i, j, question))
        random.shuffle(comparison_list)  # Randomize order
        state["egs_comparison_list"] = comparison_list
        state["egs_comparison_idx"] = 0
        state["egs_comparisons"] = []  # Results
    
    comparison_list = state["egs_comparison_list"]
    current_idx = state["egs_comparison_idx"]
    
    if current_idx >= len(comparison_list):
        # All comparisons done, build matrix
        egs_matrix = np.zeros((n, n))
        payoffs_dict = {}  # {(i, j): [payoffs]}
        
        for i, j, question, answer_i, answer_j, user_choice in state["egs_comparisons"]:
            payout = 1 if user_choice == "A" else (-1 if user_choice == "B" else 0)
            if (i, j) not in payoffs_dict:
                payoffs_dict[(i, j)] = []
            payoffs_dict[(i, j)].append(payout)
        
        # Average payoffs
        for (i, j), payoffs in payoffs_dict.items():
            avg_payoff = np.mean(payoffs)
            egs_matrix[i, j] = avg_payoff
            egs_matrix[j, i] = -avg_payoff
        
        state["egs_matrix"] = egs_matrix
        state["egs_interactive_mode"] = False
        st.success("✅ EGS matrix computed from your feedback!")
        st.rerun()
        return
    
    # Show current comparison
    i, j, question = comparison_list[current_idx]
    
    # Get cached answers
    cache_key_i = (cache_key_prefix + str(i), question)
    cache_key_j = (cache_key_prefix + str(j), question)
    answer_i = answer_cache.get(cache_key_i, "")
    answer_j = answer_cache.get(cache_key_j, "")
    
    if not answer_i or not answer_j:
        st.error("Error: Missing cached answers. Please refresh and try again.")
        return
    
    st.markdown("---")
    st.subheader(f"📊 Interactive EGS Computation")
    progress_pct = (current_idx / len(comparison_list)) * 100
    st.progress(progress_pct / 100)
    st.caption(f"Comparison {current_idx + 1} of {len(comparison_list)}: Agent A vs Agent B")
    
    st.markdown(f"**Question:** {question}")
    st.info("💡 **Blind Comparison:** The agents are labeled A and B to reduce bias. Please evaluate based on answer quality alone.")
    
    # Buttons at the top
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("✅ Prefer Agent A", type="primary", use_container_width=True, key=f"egs_prefer_{i}_{j}_{current_idx}_a"):
            state["egs_comparisons"].append((i, j, question, answer_i, answer_j, "A"))
            state["egs_comparison_idx"] = current_idx + 1
            st.rerun()
    
    with col_btn2:
        if st.button("🤝 Tie / No Preference", use_container_width=True, key=f"egs_prefer_{i}_{j}_{current_idx}_tie"):
            state["egs_comparisons"].append((i, j, question, answer_i, answer_j, "TIE"))
            state["egs_comparison_idx"] = current_idx + 1
            st.rerun()
    
    with col_btn3:
        if st.button("✅ Prefer Agent B", type="primary", use_container_width=True, key=f"egs_prefer_{i}_{j}_{current_idx}_b"):
            state["egs_comparisons"].append((i, j, question, answer_i, answer_j, "B"))
            state["egs_comparison_idx"] = current_idx + 1
            st.rerun()
    
    st.markdown("---")
    
    # Answers displayed below buttons without boxes, using native markdown
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("### Agent A Answer")
        # Use Streamlit's native markdown rendering
        st.markdown(answer_i)
    
    with col_b:
        st.markdown("### Agent B Answer")
        # Use Streamlit's native markdown rendering
        st.markdown(answer_j)


def render_comparison_tab():
    """Render the comparison tab."""
    st.header("📊 Comparison Mode")
    st.markdown("Compare multiple runs side by side.")
    
    # Get all runs
    all_runs = st.session_state.runs
    
    if not all_runs:
        st.info("No runs available for comparison. Run some simulations first!")
        return
    
    # Group runs by game
    games = {}
    for run_id, run_data in all_runs.items():
        game = run_data["game"]
        if game not in games:
            games[game] = []
        games[game].append((run_id, run_data))
    
    # Select runs to compare
    st.subheader("Select Runs to Compare")
    
    selected_runs = []
    for game, runs in games.items():
        st.markdown(f"**{game.replace('_', ' ').title()}**")
        for run_id, run_data in runs:
            label = f"{game} - {run_data.get('type', 'default')} - {time.strftime('%H:%M:%S', time.localtime(run_data['timestamp']))}"
            if st.checkbox(label, key=f"compare_{run_id}"):
                selected_runs.append((run_id, run_data))
    
    if selected_runs:
        st.subheader("Comparison")
        
        # Display visualizations side by side
        num_cols = min(len(selected_runs), 3)
        cols = st.columns(num_cols)
        
        for idx, (run_id, run_data) in enumerate(selected_runs):
            with cols[idx % num_cols]:
                st.markdown(f"**Run {idx + 1}**")
                game = run_data["game"]
                result = run_data.get("result", {})
                
                # Display appropriate visualization
                if game == "disc" and "gif_path" in result:
                    gif_path = result["gif_path"]
                    if Path(gif_path).exists():
                        st.image(gif_path)
                elif game == "blotto" and "plot_path" in result:
                    plot_path = result["plot_path"]
                    if Path(plot_path).exists():
                        st.image(plot_path)
                elif game == "differentiable_lotto" and "gif_path" in result:
                    gif_path = result["gif_path"]
                    if gif_path and Path(gif_path).exists():
                        st.image(gif_path)
                elif game == "penneys" and "plot_path" in result:
                    plot_path = result["plot_path"]
                    if Path(plot_path).exists():
                        st.image(plot_path)
                    # Also show GIFs if available
                    if "gif_path_population" in result and result["gif_path_population"]:
                        gif_path = result["gif_path_population"]
                        if Path(gif_path).exists():
                            with open(gif_path, "rb") as f:
                                gif_bytes = f.read()
                                gif_data = base64.b64encode(gif_bytes).decode()
                            st.markdown(
                                f'<img src="data:image/gif;base64,{gif_data}" style="max-width: 600px; height: auto;" />',
                                unsafe_allow_html=True
                            )
                
                # Show metadata
                with st.expander("Details"):
                    st.json({
                        "game": game,
                        "type": run_data.get("type", "N/A"),
                        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(run_data["timestamp"]))
                    })


def main():
    """Main application entry point."""
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🎮 Game Theory Simulations</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        st.markdown("""
        Select a game to run simulations and visualize results.
        """)
        
        st.markdown("---")
        st.header("About")
        st.markdown("""
        This application demonstrates Policy Space Response Oracles (PSRO) 
        for various game theory scenarios:
        
        - **Disc Game**: Population diversity vs. convergence
        - **Blotto Game**: Resource allocation strategies
        - **Differentiable Lotto**: Continuous optimization
        """)
        
        st.markdown("---")
        st.header("Session Info")
        st.metric("Total Runs", len(st.session_state.runs))
        
        if st.button("Clear All Runs", type="secondary"):
            st.session_state.runs = {}
            st.rerun()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Disc Game",
        "⚔️ Blotto Game",
        "🎲 Differentiable Lotto",
        "🪙 Penney's Game",
        "🤖 LLM Competition",
        "📊 Comparison"
    ])
    
    with tab1:
        render_disc_game_tab()
    
    with tab2:
        render_blotto_game_tab()
    
    with tab3:
        render_differentiable_lotto_tab()
    
    with tab4:
        render_penneys_game_tab()
    
    with tab5:
        render_llm_competition_tab()
    
    with tab6:
        render_comparison_tab()


if __name__ == "__main__":
    main()

