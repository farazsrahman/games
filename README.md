# Games - PSRO Implementations

Repository for Policy Space Response Oracles (PSRO) implementations as discussed in Balduzzi et al. (https://arxiv.org/pdf/1901.08106).

## Project Structure

```
src/games/
├── game.py              # Base Game abstract class and PSRO algorithms
├── disc/                # Disc Game implementation
│   ├── disc_game.py
│   └── disc_game_vis.py
└── blotto/              # Colonel Blotto game implementation
    ├── blotto.py
    └── differentiable_lotto_vis.py

streamlit/               # Streamlit web application
├── app.py              # Main Streamlit application
├── game_runners.py     # Game execution functions
└── utils.py            # Utility functions for progress tracking
```

## Setup

You can set up the environment using either `uv` or the classic `pip` and virtual environment approach:

### Option 1: Using `uv`
1. Install `uv` if you haven't:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2. Install dependencies (first time only):
    ```bash
    uv pip install -r requirements.txt
    ```

### Option 2: Using `pip` and virtual environment
1. Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. (Optional) Install the package in editable mode:
    ```bash
    pip install -e .
    ```

## Running the Demos

### Option 1: Streamlit Web Interface (Recommended for Interactive Use)

**Important:** Make sure your virtual environment is activated first!

```bash
# Activate virtual environment (if using venv)
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Then launch the app
streamlit run streamlit/app.py
```

Or use the launcher script (automatically activates venv):
```bash
./run_streamlit.sh
```

This opens a web interface in your browser where you can:
- Configure game parameters with sliders and inputs
- Run simulations with real-time progress tracking
- View visualizations (GIFs and plots) directly in the browser
- Download generated files
- Compare multiple runs side by side

The Streamlit app provides tabs for each game:
- **Disc Game**: Configure iterations, learning rate, and visualization settings
- **Blotto Game**: Adjust training iterations and evaluation rounds
- **Differentiable Lotto**: Full control over game parameters, server optimization, and constraints
- **Comparison**: Compare multiple simulation runs

### Option 2: Command Line (Quick Start)

Use the unified runner script:
```bash
# Run a specific game
python run.py disc
python run.py blotto
python run.py differentiable_lotto

# Or run all games
python run.py all
```

### Individual Game Demos

Each game file can also be run directly:

**Disc Game Demo:**
```bash
python -m games.disc.disc_game
```
The Disc Game demonstrates how different PSRO strategies affect population diversity:
- `PSRO_uniform_weaker`: Maintains diverse agent distribution
- `PSRO_uniform_stronger`: Causes convergence to same distribution

Generates two GIF files in `demos/disc/`:
- `demo_PSRO_u_weaker.gif` - Shows diverse distribution
- `demo_PSRO_u_stronger.gif` - Shows convergence

**Blotto Game (Discrete):**
```bash
python -m games.blotto.blotto
```
Generates a training plot in `demos/blotto/blotto_training.png`.
*Note: Runs 1000 iterations of training.*

**Differentiable Lotto:**
```bash
python -m games.differentiable_lotto.differentiable_lotto
```
Generates GIF visualizations showing agent evolution over time.
*Note: Runs 100 iterations and generates GIFs in `demos/blotto/`.*

## Demo Visualizations

The `demos/` folder contains:
- `demos/disc/demo_PSRO_u_weaker.gif` - Diverse agent distribution with uniform_weaker sampling
- `demos/disc/demo_PSRO_u_stronger.gif` - Convergence with uniform_stronger sampling
- `demos/blotto/blotto_training.png` - Training progress plot (from discrete blotto demo)
- `demos/blotto/demo_PSRO_u_weaker.gif` - Differentiable lotto visualization (if you run the differentiable demo)
- `demos/blotto/demo_PSRO_u_stronger.gif` - Differentiable lotto visualization (if you run the differentiable demo)

## Usage as a Package

After installation, you can import and use the games:

```python
from games import Game, run_PSRO_uniform_weaker
from games.disc import DiscGame
from games.blotto import BlottoGame, LogitAgent

# Use the games...
game = DiscGame()
# ...
```
