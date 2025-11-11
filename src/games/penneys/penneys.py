import numpy as np
import os
import copy
from tqdm import trange
from games.game import Game, run_PSRO_uniform, run_PSRO_uniform_weaker, run_PSRO_uniform_stronger
from typing import List, Tuple
import random


class PennysAgent:
    """
    An agent in Penney's Game represented as a probability distribution
    over all possible sequences of a given length.
    
    For sequence length k, there are 2^k possible sequences.
    """
    
    def __init__(self, sequence_length: int = 3, seed: int = None):
        """
        Initialize agent with uniform distribution over all sequences.
        
        Args:
            sequence_length: Length of H/T sequences (default 3)
            seed: Random seed for reproducibility
        """
        self.sequence_length = sequence_length
        self.num_sequences = 2 ** sequence_length
        # Logits for each sequence (will be converted to probabilities)
        self.logits = np.zeros(self.num_sequences)
        self.rng = np.random.RandomState(seed)
    
    def get_sequence_from_index(self, idx: int) -> str:
        """Convert index to binary sequence string (e.g., 0 -> 'HHH', 3 -> 'HTT')."""
        binary = format(idx, f'0{self.sequence_length}b')
        return binary.replace('0', 'H').replace('1', 'T')
    
    def get_index_from_sequence(self, sequence: str) -> int:
        """Convert sequence string to index (e.g., 'HHH' -> 0, 'HTT' -> 3)."""
        binary = sequence.replace('H', '0').replace('T', '1')
        return int(binary, 2)
    
    def get_probabilities(self) -> np.ndarray:
        """Get probability distribution over sequences."""
        exp_logits = np.exp(self.logits - np.max(self.logits))
        return exp_logits / np.sum(exp_logits)
    
    def sample_sequence(self) -> str:
        """Sample a sequence according to current probability distribution."""
        probs = self.get_probabilities()
        idx = self.rng.choice(self.num_sequences, p=probs)
        return self.get_sequence_from_index(idx)
    
    def get_all_sequences(self) -> List[str]:
        """Get list of all possible sequences."""
        return [self.get_sequence_from_index(i) for i in range(self.num_sequences)]
    
    def copy(self):
        """Create a deep copy of the agent."""
        new_agent = PennysAgent(self.sequence_length)
        new_agent.logits = self.logits.copy()
        return new_agent


class PennysGame(Game):
    """
    Penney's Game: Two players choose sequences of H/T.
    A coin is flipped repeatedly, and the first player whose sequence
    appears wins. This game is non-transitive - for any sequence,
    there's a sequence that beats it with probability > 0.5.
    """
    
    def __init__(self, sequence_length: int = 3, max_flips: int = 1000):
        """
        Initialize Penney's Game.
        
        Args:
            sequence_length: Length of sequences (default 3)
            max_flips: Maximum coin flips before declaring a tie (default 1000)
        """
        self.sequence_length = sequence_length
        self.max_flips = max_flips
    
    def _simulate_coin_flips(self, sequence_u: str, sequence_v: str) -> int:
        """
        Simulate coin flips until one sequence appears.
        
        Returns:
            1 if sequence_u appears first (agent u wins)
            -1 if sequence_v appears first (agent v wins)
            0 if max_flips reached without either sequence (tie)
        """
        # If both sequences are the same, it's a tie
        if sequence_u == sequence_v:
            return 0
        
        history = []
        
        for _ in range(self.max_flips):
            # Flip coin (fair coin, 50/50)
            flip = 'H' if random.random() < 0.5 else 'T'
            history.append(flip)
            
            # Check if either sequence appears
            if len(history) >= self.sequence_length:
                recent = ''.join(history[-self.sequence_length:])
                if recent == sequence_u:
                    return 1
                if recent == sequence_v:
                    return -1
        
        return 0  # Tie
    
    def play(self, u: PennysAgent, v: PennysAgent, n_rounds: int = 1000) -> float:
        """
        Play n_rounds of Penney's Game between agents u and v.
        
        Returns:
            Average payoff for agent u (1.0 = always wins, -1.0 = always loses)
        """
        total_payoff = 0.0
        
        for _ in range(n_rounds):
            seq_u = u.sample_sequence()
            seq_v = v.sample_sequence()
            result = self._simulate_coin_flips(seq_u, seq_v)
            total_payoff += result
        
        return total_payoff / n_rounds
    
    def improve(self, u: PennysAgent, v: PennysAgent, *, learning_rate: float = 0.05, 
                n_rollouts: int = 200) -> PennysAgent:
        """
        Improve agent u's strategy against agent v.
        
        Strategy: Evaluate all sequences against opponent's distribution,
        then update to favor sequences with higher expected win rates.
        This is a better approximation of finding the best response.
        
        Args:
            u: Agent to improve
            v: Opponent agent
            learning_rate: How much to shift probabilities (lower = more stable)
            n_rollouts: Number of games per sequence for evaluation
        
        Returns:
            New improved agent
        """
        # Create new agent
        u_new = u.copy()
        
        # Get opponent's distribution
        v_probs = v.get_probabilities()
        
        # Evaluate each sequence against opponent's distribution
        # This is more efficient than sampling and gives better estimates
        expected_win_rates = np.zeros(u.num_sequences)
        
        for seq_idx in range(u.num_sequences):
            seq_u = u.get_sequence_from_index(seq_idx)
            
            # Evaluate this sequence against opponent's distribution
            total_wins = 0.0
            total_games = 0
            
            # Sample from opponent's distribution
            for _ in range(n_rollouts):
                seq_v = v.sample_sequence()
                result = self._simulate_coin_flips(seq_u, seq_v)
                total_games += 1
                if result > 0:  # u wins
                    total_wins += 1
            
            expected_win_rates[seq_idx] = total_wins / total_games if total_games > 0 else 0.5
        
        # Convert win rates to advantages (centered around 0.5)
        advantages = expected_win_rates - 0.5  # Range: [-0.5, 0.5]
        
        # Update logits proportionally to advantages
        # Use lower learning rate to prevent premature convergence
        u_new.logits = u.logits + learning_rate * advantages
        
        # Add small uniform prior to maintain exploration
        # This prevents complete convergence to a single sequence
        # Equivalent to adding a small constant to all logits
        uniform_prior_strength = 0.02
        u_new.logits = u_new.logits + uniform_prior_strength
        
        # Normalize to prevent logits from growing too large
        u_new.logits = u_new.logits - np.max(u_new.logits)
        
        return u_new


def demo_penneys_game(improvement_function, plot_file_name="penneys_PSRO"):
    """Run a PSRO demo of Penney's Game with a given improvement function."""
    os.makedirs("demos/penneys", exist_ok=True)
    
    game = PennysGame(sequence_length=3)
    num_iterations = 500
    n_rounds = 500
    
    # Create a population of 3 agents
    agent_1 = PennysAgent(sequence_length=3, seed=42)
    agent_2 = PennysAgent(sequence_length=3, seed=43)
    agent_3 = PennysAgent(sequence_length=3, seed=44)
    
    # Initialize with slightly different distributions
    agent_1.logits = np.random.RandomState(42).randn(8)
    agent_2.logits = np.random.RandomState(43).randn(8)
    agent_3.logits = np.random.RandomState(44).randn(8)
    
    print("=" * 70)
    print(f"Penney's Game Demo with {improvement_function.__name__}")
    print("=" * 70)
    print(f"Sequence length: {game.sequence_length}")
    print(f"Possible sequences: {agent_1.get_all_sequences()}")
    
    # Track win rates for each agent pair
    values_12 = []
    values_13 = []
    values_23 = []
    
    # Track agent history for visualization
    agents_history = [[agent_1.copy(), agent_2.copy(), agent_3.copy()]]
    
    import matplotlib.pyplot as plt
    from tqdm import trange
    
    for i in trange(num_iterations, desc="Training iterations"):
        population = [agent_1, agent_2, agent_3]
        
        # Improve each agent using the PSRO strategy
        agent_1 = improvement_function(0, population, game)
        agent_2 = improvement_function(1, population, game)
        agent_3 = improvement_function(2, population, game)
        
        # Store agent states
        agents_history.append([agent_1.copy(), agent_2.copy(), agent_3.copy()])
        
        # Evaluate win rates
        val_12 = game.play(agent_1, agent_2, n_rounds=n_rounds)
        val_13 = game.play(agent_1, agent_3, n_rounds=n_rounds)
        val_23 = game.play(agent_2, agent_3, n_rounds=n_rounds)
        
        values_12.append(val_12)
        values_13.append(val_13)
        values_23.append(val_23)
    
    # Create static plot
    plt.figure(figsize=(12, 6))
    plt.plot(values_12, label='Agent 1 vs Agent 2', alpha=0.7)
    plt.plot(values_13, label='Agent 1 vs Agent 3', alpha=0.7)
    plt.plot(values_23, label='Agent 2 vs Agent 3', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Win Rate (Agent i)')
    plt.title(f"Penney's Game: Win Rate Over Time ({improvement_function.__name__})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Tie')
    plt.tight_layout()
    
    plot_path = f"demos/penneys/{plot_file_name}.png"
    plt.savefig(plot_path)
    plt.close()
    
    print(f"\nSaved plot to {plot_path}")
    print(f"Final win rates:")
    print(f"  Agent 1 vs Agent 2: {values_12[-1]:.4f}")
    print(f"  Agent 1 vs Agent 3: {values_13[-1]:.4f}")
    print(f"  Agent 2 vs Agent 3: {values_23[-1]:.4f}")
    
    # Print final probability distributions
    print("\nFinal probability distributions:")
    sequences = agent_1.get_all_sequences()
    for i, agent in enumerate([agent_1, agent_2, agent_3], 1):
        probs = agent.get_probabilities()
        print(f"\nAgent {i}:")
        for seq, prob in zip(sequences, probs):
            print(f"  {seq}: {prob:.4f}")
    
    # Create GIF visualizations
    try:
        from games.penneys.penneys_vis import gif_from_population, gif_from_matchups
        
        # GIF showing probability distributions and entropy
        gif_path_pop = gif_from_population(
            agents_history,
            path=f"demos/penneys/{plot_file_name}_population.gif",
            fps=20,
            stride=max(1, num_iterations // 200),
            dpi=120,
            show_entropy=True
        )
        print(f"Saved population GIF: {gif_path_pop}")
        
        # GIF showing win rates over time
        gif_path_match = gif_from_matchups(
            game,
            agents_history,
            path=f"demos/penneys/{plot_file_name}_matchups.gif",
            fps=20,
            stride=max(1, num_iterations // 200),
            dpi=120,
            n_rounds=500
        )
        print(f"Saved matchups GIF: {gif_path_match}")
    except ImportError:
        print("\nNote: Visualization module not available.")
    except Exception as e:
        print(f"\nNote: Could not generate GIFs: {e}")


if __name__ == "__main__":
    print("Running Penney's Game Demo with PSRO_uniform...")
    demo_penneys_game(run_PSRO_uniform, "penneys_PSRO_uniform")
    
    print("\n" + "=" * 70)
    print("Running Penney's Game Demo with PSRO_uniform_weaker...")
    print("=" * 70 + "\n")
    demo_penneys_game(run_PSRO_uniform_weaker, "penneys_PSRO_uniform_weaker")
    
    print("\n" + "=" * 70)
    print("Running Penney's Game Demo with PSRO_uniform_stronger...")
    print("=" * 70 + "\n")
    demo_penneys_game(run_PSRO_uniform_stronger, "penneys_PSRO_uniform_stronger")
    
    print("\n" + "=" * 70)
    print("All PSRO demos completed! Check demos/penneys/ for the generated plots.")
    print("=" * 70)

