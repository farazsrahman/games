#!/usr/bin/env python3
"""
Unified runner for all game demos.
Usage:
    python run.py disc
    python run.py blotto
    python run.py differentiable_lotto
    python run.py all
"""

import sys
import os
import runpy

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def run_disc():
    """Run the Disc Game demo by executing its main block."""
    # Execute the module as if it were run directly
    runpy.run_module('games.disc.disc_game', run_name='__main__')


def run_blotto():
    """Run the Blotto Game demo by executing its main block."""
    # Execute the module as if it were run directly
    runpy.run_module('games.blotto.blotto', run_name='__main__')


def run_differentiable_lotto():
    """Run the Differentiable Lotto demo by executing its main block."""
    # Execute the module as if it were run directly
    runpy.run_module('games.blotto.differentiable_lotto', run_name='__main__')


def run_all():
    """Run all game demos."""
    print("=" * 70)
    print("Running All Game Demos")
    print("=" * 70 + "\n")
    
    run_disc()
    print("\n\n")
    run_blotto()
    print("\n\n")
    run_differentiable_lotto()
    
    print("\n" + "=" * 70)
    print("All demos completed!")
    print("=" * 70)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nAvailable games:")
        print("  disc              - Run Disc Game demo")
        print("  blotto            - Run Blotto Game demo")
        print("  differentiable_lotto - Run Differentiable Lotto demo")
        print("  all               - Run all demos")
        sys.exit(1)
    
    game = sys.argv[1].lower()
    
    if game == "disc":
        run_disc()
    elif game == "blotto":
        run_blotto()
    elif game == "differentiable_lotto" or game == "diff_lotto":
        run_differentiable_lotto()
    elif game == "all":
        run_all()
    else:
        print(f"Unknown game: {game}")
        print("Available games: disc, blotto, differentiable_lotto, all")
        sys.exit(1)


if __name__ == "__main__":
    main()

