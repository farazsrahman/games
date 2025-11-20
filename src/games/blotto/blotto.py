import numpy as np

import os
import copy
from tqdm import trange
from games.game import Game, contract, run_PSRO_uniform, run_PSRO_uniform_weaker, run_PSRO_uniform_stronger, create_population
from games.blotto.differentiable_lotto_vis import gif_from_population


class BlottoAgent:

    """
    There are different ways to parametrize the Blotto player.
    This abstract class will help unify the interface for BlottoGame
    """

    def __init__(self, n_battlefields: int, budget: int):
        self.n_battlefields = n_battlefields
        self.budget = budget

    def sample(self) -> np.ndarray:
        pass

    def update(self, rollouts):
        pass

class LogitAgent(BlottoAgent):

    """
    There are different ways to parametrize the Blotto player.
    This abstract class will help unify the interface for BlottoGame
    """

    def __init__(self, n_battlefields: int, budget: int):
        assert n_battlefields == 3 and budget == 10
        self.n_battlefields = n_battlefields
        self.budget = budget
        self.logits = np.zeros(66)

    def sample(self) -> np.ndarray:
        # Sample an allocation index from self.action (treated as logits or probabilities)
        exp_logits = np.exp(self.logits - np.max(self.logits))
        probs = exp_logits / np.sum(exp_logits)
        index = np.random.choice(len(self.logits), p=probs)
        alloc = index_to_allocation(index, self.budget, self.n_battlefields)
        return np.array(alloc)

    def update(self, rollouts):
        """
        MPO-style update for a discrete (66-way) categorical policy.
        rollouts: list of (action_index, reward) pairs where reward ∈ {0, 0.5, 1}.
        """
        if not rollouts:
            return

        # --- E-step: build non-parametric target q(a) ∝ exp(R/eta) aggregated by action ---
        eta = 1.0                      # temperature for exponentiated-returns
        eps = 1e-8                     # numerical stability
        A = len(self.logits)           # 66
        weights_per_action = np.zeros(A, dtype=np.float64)

        # stabilize exponentials by subtracting max reward (≤ 1.0 anyway)
        r_max = max(r for _, r in rollouts) if rollouts else 0.0
        for a, r in rollouts:
            w = np.exp((r - r_max) / eta)
            weights_per_action[a] += w

        # target distribution q over all actions (smooth to avoid zeros)
        q = weights_per_action + eps
        q = q / q.sum()

        # --- M-step: trust-regioned projection back to parametrized categorical ---
        # Current policy probs
        logits_old = self.logits.copy()
        max_logit = np.max(logits_old)
        p = np.exp(logits_old - max_logit)
        p = p / p.sum()

        # Exponentiated-gradient / MPO-style interpolation in probability space:
        # pi_new ∝ p^(1-α) * q^α  (α is the "step size"/KL trust)
        alpha = 0.2                    # KL trust (0→no change, 1→match q)
        pi_new = (p ** (1.0 - alpha)) * (q ** alpha)
        pi_new = pi_new / pi_new.sum()

        # Back to logits (defined up to a constant shift)
        self.logits = np.log(pi_new + eps)


class BlottoGame(Game):

    def _rollout(self, u: BlottoAgent, v: BlottoAgent, n_rounds: int):
        """
        Collect a batch of off-policy samples for u vs fixed opponent v.
        Return [(a_idx, r), ...] which can be used to compute % of u wins 
        and update rule
        """

        assert u.n_battlefields == v.n_battlefields and u.budget == v.budget, (
            f"Error: Agents are not set up for the same game parameters. "
            f"u.n_battlefields = {u.n_battlefields}, v.n_battlefields = {v.n_battlefields}; "
            f"u.budget = {u.budget}, v.budget = {v.budget}"
        )

        rollouts = []
        rewards = []

        for _ in range(n_rounds):
            # sample both players' allocations
            u_alloc = u.sample()
            v_alloc = v.sample()

            # compute single-round Blotto reward for u_new (1 win, 0.5 tie, 0 loss)
            u_value = ((u_alloc > v_alloc) + 0.5 * (u_alloc == v_alloc)).sum().item()
            v_value = ((v_alloc > u_alloc) + 0.5 * (v_alloc == u_alloc)).sum().item()
            r = 1.0 if u_value > v_value else (0.5 if u_value == v_value else 0.0)
            rewards.append(r)

            # record the action index that u_new actually played
            a_idx = allocation_to_index(tuple(int(x) for x in u_alloc), u.budget, u.n_battlefields)
            rollouts.append((a_idx, r))

        return rollouts

    def play(self, u: BlottoAgent, v: BlottoAgent, n_rounds: int = 3) -> float:
        
        rollouts = self._rollout(u, v, n_rounds)
        num_u_wins = sum(1 for (_, r) in rollouts if (r == 1.0 or r == 0.5))
        
        return num_u_wins / n_rounds

    def improve(self, u: BlottoAgent, v: BlottoAgent) -> BlottoAgent:

        u_new = copy.deepcopy(u)
        rollouts = self._rollout(u_new, v, n_rounds=100)  # n_rounds can be adjusted as needed
        u_new.update(rollouts)
        
        return u_new
    
    def create_agent(self, n_battlefields: int = 3, budget: int = 10, seed: int = None) -> LogitAgent:
        """Create a random Blotto agent."""
        agent = LogitAgent(n_battlefields, budget)
        if seed is not None:
            np.random.seed(seed)
        # Initialize with small random logits for diversity
        agent.logits = np.random.randn(66) * 0.1
        return agent


from math import comb

def allocation_to_index(x, N, K):
    """
    Map an allocation tuple x (length K, nonnegatives summing to N)
    to a 0-based index in [0, comb(N+K-1, K-1)-1].
    """
    assert len(x) == K and sum(x) == N and all(t >= 0 for t in x)
    M = N + K - 1
    # bars at strictly increasing positions 1..M
    bars = []
    pos = 0
    for i in range(K-1):
        pos += x[i] + 1
        bars.append(pos)

    # rank the (K-1)-combination 'bars' in lexicographic order
    rank = 0
    prev = 0
    for i, b in enumerate(bars, start=1):
        for t in range(prev + 1, b):
            rank += comb(M - t, (K - 1) - i)
        prev = b
    return rank

def index_to_allocation(index, N, K):
    """
    Inverse of allocation_to_index: map index -> allocation tuple of length K.
    """
    M = N + K - 1
    total = comb(M, K - 1)
    assert 0 <= index < total

    bars = []
    prev = 0
    r = index
    for i in range(1, K):
        # choose the smallest feasible bar position at this step
        for t in range(prev + 1, M - ((K - 1) - i) + 1):
            cnt = comb(M - t, (K - 1) - i)
            if r < cnt:
                bars.append(t)
                prev = t
                break
            r -= cnt

    # convert bars -> allocation
    x = [0] * K
    x[0] = bars[0] - 1
    for i in range(1, K - 1):
        x[i] = bars[i] - bars[i - 1] - 1
    x[K - 1] = M - bars[-1]
    return tuple(x)

# --- quick sanity checks ---
def test_index_allocation_map():
    N, K = 10, 3
    # round-trip a few examples
    for alloc in [(10,0,0), (0,10,0), (0,0,10), (3,4,3), (1,2,7), (5,0,5)]:
        i = allocation_to_index(alloc, N, K)
        back = index_to_allocation(i, N, K)
        assert back == alloc, (alloc, i, back)
    # check coverage
    total = comb(N+K-1, K-1)
    seen = {index_to_allocation(i, N, K) for i in range(total)}
    assert len(seen) == total  # bijection


if __name__ == "__main__":
    def demo_blotto_PSRO(improvement_function, plot_file_name="blotto_PSRO", num_agents: int = 3):
        """Run a PSRO demo of the Blotto game with a given improvement function."""
        os.makedirs("demos/blotto", exist_ok=True)
        
        game = BlottoGame()
        num_iterations = 1000
        n_rounds = 1000
        
        # Create a population of N agents
        population = create_population(game, num_agents, seed=42, n_battlefields=3, budget=10)
        
        print("=" * 70)
        print(f"Blotto Game Demo with {improvement_function.__name__} ({num_agents} agents)")
        print("=" * 70)
        
        # Track win rates for all agent pairs
        win_rates = {f"{i}vs{j}": [] for i in range(num_agents) for j in range(i+1, num_agents)}
        
        # Track agent history for GIF generation
        agents_history = [[copy.deepcopy(agent) for agent in population]]
        
        import matplotlib.pyplot as plt
        from tqdm import trange
        
        for i in trange(num_iterations, desc="Training iterations"):
            # Improve each agent using the PSRO strategy
            new_population = []
            for agent_idx in range(num_agents):
                improved_agent = improvement_function(agent_idx, population, game)
                new_population.append(improved_agent)
            population = new_population
            
            # Store agent states (need to copy since agents are mutable)
            agents_history.append([copy.deepcopy(agent) for agent in population])
            
            # Evaluate win rates for all pairs
            for i in range(num_agents):
                for j in range(i+1, num_agents):
                    val = game.play(population[i], population[j], n_rounds=n_rounds)
                    win_rates[f"{i}vs{j}"].append(val)
        
        # Create static plot
        plt.figure(figsize=(12, 6))
        for pair_key, values in win_rates.items():
            i, j = map(int, pair_key.split('vs'))
            plt.plot(values, label=f'Agent {i} vs Agent {j}', alpha=0.7)
        plt.xlabel('Iteration')
        plt.ylabel('Win Rate')
        plt.title(f'Blotto Game: Win Rate Over Time ({improvement_function.__name__}, {num_agents} agents)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = f"demos/blotto/{plot_file_name}.png"
        plt.savefig(plot_path)
        plt.close()
        
        print(f"\nSaved plot to {plot_path}")
        print(f"Final win rates:")
        for pair_key, values in win_rates.items():
            i, j = map(int, pair_key.split('vs'))
            print(f"  Agent {i} vs Agent {j}: {values[-1]:.4f}")
        
        # Create expected allocation per battlefield plot
        try:
            from games.blotto.blotto_vis import get_expected_allocation
            
            plt.figure(figsize=(12, 6))
            battlefields = ['Battlefield 1', 'Battlefield 2', 'Battlefield 3']
            x = np.arange(len(battlefields))
            width = 0.8 / num_agents
            agent_colors = plt.cm.tab10(np.linspace(0, 1, num_agents))
            
            # Get final expected allocations for all agents
            for i in range(num_agents):
                offset = (i - (num_agents - 1) / 2) * width
                expected = get_expected_allocation(population[i])
                bars = plt.bar(x + offset, expected, width, 
                              label=f'Agent {i}', color=agent_colors[i], alpha=0.8)
                
                # Add value labels on bars
                for j, val in enumerate(expected):
                    plt.text(j + offset, val + 0.2, f'{val:.2f}', 
                            ha='center', va='bottom', fontsize=9)
            
            plt.xlabel('Battlefield', fontsize=12)
            plt.ylabel('Expected Troops', fontsize=12)
            plt.title(f'Expected Allocation per Battlefield (Final, {improvement_function.__name__}, {num_agents} agents)', fontsize=14)
            plt.xticks(x, battlefields)
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
            plt.ylim(0, 10)
            plt.tight_layout()
            
            allocation_plot_path = f"demos/blotto/{plot_file_name}_allocations.png"
            plt.savefig(allocation_plot_path)
            plt.close()
            print(f"Saved allocation plot to {allocation_plot_path}")
        except Exception as e:
            print(f"Could not generate allocation plot: {e}")
        
        # Create GIF visualizations
        try:
            from games.blotto.blotto_vis import gif_from_population, gif_from_matchups
            
            # GIF showing expected allocations and entropy
            gif_path_pop = gif_from_population(
                agents_history,
                path=f"demos/blotto/{plot_file_name}_population.gif",
                fps=20,
                stride=max(1, num_iterations // 200),  # Limit to ~200 frames
                dpi=120,
                show_entropy=True
            )
            print(f"Saved population GIF: {gif_path_pop}")
            
            # GIF showing win rates over time
            gif_path_match = gif_from_matchups(
                game,
                agents_history,
                path=f"demos/blotto/{plot_file_name}_matchups.gif",
                fps=20,
                stride=max(1, num_iterations // 200),
                dpi=120,
                n_rounds=500  # Use fewer rounds for faster computation
            )
            print(f"Saved matchups GIF: {gif_path_match}")
        except ImportError:
            print("\nNote: Visualization module not available.")
        except Exception as e:
            print(f"\nNote: Could not generate GIFs: {e}")
    
    # Run all PSRO variants
    print("Running Blotto Game Demo with PSRO_uniform...")
    demo_blotto_PSRO(run_PSRO_uniform, "blotto_PSRO_uniform")
    
    print("\n" + "=" * 70)
    print("Running Blotto Game Demo with PSRO_uniform_weaker...")
    print("=" * 70 + "\n")
    demo_blotto_PSRO(run_PSRO_uniform_weaker, "blotto_PSRO_uniform_weaker")
    
    print("\n" + "=" * 70)
    print("Running Blotto Game Demo with PSRO_uniform_stronger...")
    print("=" * 70 + "\n")
    demo_blotto_PSRO(run_PSRO_uniform_stronger, "blotto_PSRO_uniform_stronger")
    
    print("\n" + "=" * 70)
    print("All PSRO demos completed! Check demos/blotto/ for the generated plots.")
    print("=" * 70)
