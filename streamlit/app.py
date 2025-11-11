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
    run_differentiable_lotto_demo
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
            min_value=3,
            max_value=3,
            value=3,
            disabled=True,
            help="Currently fixed at 3"
        )
        budget = st.number_input(
            "Budget",
            min_value=10,
            max_value=10,
            value=10,
            disabled=True,
            help="Currently fixed at 10"
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
                        budget=budget
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
                        budget=budget
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
    
    # List available demo files (plots and GIFs)
    demo_files_png = list_demo_files("blotto", ".png")
    demo_files_gif = list_demo_files("blotto", ".gif")
    blotto_plots = [f for f in demo_files_png if "blotto_PSRO" in f]
    blotto_gifs = [f for f in demo_files_gif if "blotto_PSRO" in f and ("population" in f or "matchups" in f)]
    
    # Show GIFs if available
    if blotto_gifs:
        st.markdown("#### Animated Visualizations")
        gif_type = st.radio(
            "Select GIF type",
            ["Population (Allocations & Entropy)", "Matchups (Win Rates)"],
            key="blotto_gif_type"
        )
        
        # Filter GIFs by type
        if "Population" in gif_type:
            gif_files = [f for f in blotto_gifs if "population" in f]
        else:
            gif_files = [f for f in blotto_gifs if "matchups" in f]
        
        if gif_files:
            selected_gif = st.selectbox(
                "Select GIF to display",
                gif_files,
                key="blotto_gif_selector"
            )
            
            if selected_gif:
                gif_path = Path("demos/blotto") / selected_gif
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
                        key="blotto_gif_download"
                    )
    
    # Show static plots
    if blotto_plots:
        st.markdown("#### Static Plots")
        selected_file = st.selectbox(
            "Select plot to display",
            blotto_plots,
            key="blotto_file_selector"
        )
        
        if selected_file:
            plot_path = Path("demos/blotto") / selected_file
            if plot_path.exists():
                st.image(str(plot_path))
                
                with open(plot_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Plot",
                        data=f.read(),
                        file_name=selected_file,
                        mime="image/png",
                        key="blotto_download"
                    )
    
    # Show final statistics if available
    if "blotto" in [r["game"] for r in st.session_state.runs.values()]:
        latest_run = max(
            [r for r in st.session_state.runs.values() if r["game"] == "blotto"],
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
    
    if not blotto_plots and not blotto_gifs:
        st.info("No visualizations available. Run a simulation to generate them.")


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
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Disc Game",
        "⚔️ Blotto Game",
        "🎲 Differentiable Lotto",
        "📊 Comparison"
    ])
    
    with tab1:
        render_disc_game_tab()
    
    with tab2:
        render_blotto_game_tab()
    
    with tab3:
        render_differentiable_lotto_tab()
    
    with tab4:
        render_comparison_tab()


if __name__ == "__main__":
    main()

