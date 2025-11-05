# mlgame

## Files

- `game.py` - Base Game abstract class and PSRO improvement algorithms (uniform, uniform_weaker, uniform_stronger)
- `disc_game.py` - DiscGame implementation as discussed in Balduzzi et al. (https://arxiv.org/pdf/1901.08106) and demo script
- `disc_game_vis.py` - Visualization functions for plotting points, creating GIFs from agent populations and game states

## Running the Disc Game Demo

To run the disc game demo with PSRO variants:

```bash
uv run python disc_game.py
```

This will generate two GIF files in the `static/` folder:
- `static/demo_PSRO_u_weaker.gif` - Shows diverse distribution when using uniform_weaker sampling
- `static/demo_PSRO_u_stronger.gif` - Shows convergence to same distribution when using uniform_stronger sampling

## Demo Visualizations

The `static/` folder contains pre-generated demo GIFs:
- `static/demo_PSRO_u_weaker.gif` - Demonstrates diverse agent distribution with uniform_weaker sampling
- `static/demo_PSRO_u_stronger.gif` - Demonstrates convergence with uniform_stronger sampling
