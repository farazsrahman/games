import numpy as np
import os
from tqdm import trange
from games.game import Game, run_PSRO_uniform, run_PSRO_uniform_weaker, run_PSRO_uniform_stronger
from typing import Tuple

"""
Differentiable Lotto game implementation with PROJECTED GRADIENT ASCENT.

An agent (p, v) distributes one unit of mass over k servers, where each server is a point in R².
Customers are softly assigned to the nearest servers using softmax, determining the agents' payoffs.

Optimization Method: Projected Gradient Ascent
1. Compute unconstrained gradient ∇f(x)
2. Take gradient step: x_step = x_old + η * ∇f(x_old)
3. Project x_step onto feasible region: x_new = proj_C(x_step)
"""


class DifferentiableLotto(Game):
    """
    Differentiable Lotto game implementation.
    
    An agent is represented as (p, v) where:
    - p: k-dimensional vector of masses (probabilities, sums to 1)
    - v: k×2 matrix of server positions in R²
    """
    
    def __init__(self, num_customers: int = 9, num_servers: int = 3, 
                 customer_scale: float = 1.0, seed: int = 42,
                 optimize_server_positions: bool = False,
                 enforce_width_constraint: bool = True,
                 width_penalty_lambda: float = 10.0):
        """
        Initialize the Differentiable Lotto game.
        
        Args:
            num_customers: Number of customers (c). Default 9 as in paper experiments.
            num_servers: Number of servers per agent (k). Default 3.
            customer_scale: Scale for random customer positions. Default 1.0 for square [-1, 1]².
            seed: Random seed for customer generation
            optimize_server_positions: If True, optimize server positions v. 
                                      If False, only optimize mass distribution p (servers are fixed).
            enforce_width_constraint: If True, add penalty term λ(width - 1)² to push width to 1.
            width_penalty_lambda: Penalty coefficient for width constraint. Default 10.0.
        """
        self.c = num_customers
        self.k = num_servers
        self.optimize_server_positions = optimize_server_positions
        self.enforce_width_constraint = enforce_width_constraint
        self.width_penalty_lambda = width_penalty_lambda if enforce_width_constraint else 0.0
        np.random.seed(seed)
        # Generate fixed set of customers uniformly at random in square [-customer_scale, customer_scale]²
        self.customers = np.random.uniform(-customer_scale, customer_scale, size=(num_customers, 2))
    
    def play(self, agent_u: Tuple[np.ndarray, np.ndarray], 
            agent_v: Tuple[np.ndarray, np.ndarray]) -> float:
        """
        Compute payoff φ((p, v), (q, w)) = Σ(i=1 to c, j=1 to k) (pj vij - qj wij)
        """
        p, v = agent_u
        q, w = agent_v
        
        payoff = 0.0
        for i in range(self.c):
            customer = self.customers[i]
            vi, wi = self._compute_softmax_distances(customer, v, w)
            payoff += np.sum(p * vi) - np.sum(q * wi)
        
        return payoff
        
    def _compute_softmax_distances(self, customer: np.ndarray, 
                                  servers_p: np.ndarray, 
                                  servers_q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute soft assignments for a customer to all servers."""
        dists_p = np.sum((customer - servers_p) ** 2, axis=1)
        dists_q = np.sum((customer - servers_q) ** 2, axis=1)
        all_dists = np.concatenate([-dists_p, -dists_q])
        softmax_all = self._softmax(all_dists)
        return softmax_all[:self.k], softmax_all[self.k:]
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax with numerical stability."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def _compute_width(self, p: np.ndarray, v: np.ndarray) -> float:
        """Compute the width (expected distance from barycenter)."""
        barycenter = np.sum(p[:, np.newaxis] * v, axis=0)
        distances = np.linalg.norm(v - barycenter, axis=1)
        return np.sum(p * distances)
    
    def _compute_width_gradient(self, p: np.ndarray, v: np.ndarray, 
                                epsilon: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradient of width with respect to p and v.
        
        Width = Σ(i) p_i * ||v_i - barycenter|| where barycenter = Σ(i) p_i * v_i
        """
        barycenter = np.sum(p[:, np.newaxis] * v, axis=0)
        v_centered = v - barycenter
        distances = np.linalg.norm(v_centered, axis=1)
        
        # Gradient w.r.t. p: direct term + indirect term from barycenter change
        grad_p = distances.copy()
        for j in range(self.k):
            barycenter_deriv = v[j]  # ∂barycenter/∂p_j = v_j
            for i in range(self.k):
                if distances[i] > epsilon:
                    grad_p[j] -= p[i] * np.dot(v_centered[i], barycenter_deriv) / distances[i]
        
        # Gradient w.r.t. v
        unit_vectors = np.zeros_like(v)
        for i in range(self.k):
            if distances[i] > epsilon:
                unit_vectors[i] = v_centered[i] / distances[i]
        
        weighted_avg_unit = np.sum(p[:, np.newaxis] * unit_vectors, axis=0)
        grad_v = np.zeros_like(v)
        for j in range(self.k):
            if distances[j] > epsilon:
                grad_v[j] = p[j] * (unit_vectors[j] - weighted_avg_unit)
        
        return grad_p, grad_v
    
    def _project_simplex(self, v: np.ndarray) -> np.ndarray:
        """
        Euclidean projection of vector v onto the probability simplex.
        
        This finds: argmin_p ||p - v||² s.t. p >= 0, Σp_i = 1
        
        Algorithm from Duchi et al. (2008):
        "Efficient projections onto the l1-ball for learning in high dimensions"
        """
        n = len(v)
        u = np.sort(v)[::-1]  # Sort in descending order
        cssv = np.cumsum(u)  # Cumulative sum
        rho = np.where(u > (cssv - 1) / np.arange(1, n + 1))[0][-1]
        theta = (cssv[rho] - 1) / (rho + 1)
        return np.maximum(v - theta, 0)
    
    def _normalize_width(self, p: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Normalize server positions so that width equals 1. Only used for initialization."""
        current_width = self._compute_width(p, v)
        if current_width > 1e-10:
            barycenter = np.sum(p[:, np.newaxis] * v, axis=0)
            v_centered = v - barycenter
            return v_centered / current_width + barycenter
        return v
    
    def _compute_gradient_p(self, agent_u: Tuple[np.ndarray, np.ndarray],
                            agent_v: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Compute UNCONSTRAINED gradient of payoff with respect to mass distribution p.
        
        For projected gradient ascent, we compute the full unconstrained gradient.
        The projection step will handle the simplex constraint.
        
        Gradient formula:
        ∂φ/∂p_j = Σ(i=1 to c) v_ij - 2λ(width - 1) * ∂width/∂p_j
        """
        p, v = agent_u
        q, w = agent_v
        
        # Compute soft assignments for all customers
        vij_all = np.array([self._compute_softmax_distances(c, v, w)[0] for c in self.customers])
        grad_p = np.sum(vij_all, axis=0)  # Σ(i) v_ij
        
        # Add penalty term gradient if width constraint is enforced
        if self.enforce_width_constraint:
            width = self._compute_width(p, v)
            width_grad_p, _ = self._compute_width_gradient(p, v)
            grad_p -= 2 * self.width_penalty_lambda * (width - 1.0) * width_grad_p
        
        # Return UNCONSTRAINED gradient (no mean subtraction for projected gradient ascent)
        return grad_p
    
    def _compute_gradient_v(self, agent_u: Tuple[np.ndarray, np.ndarray],
                            agent_v: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Compute gradient of payoff with respect to server positions v.
        
        Gradient formula:
        ∂φ/∂v_jd = Σ(i=1 to c) [Σ(k) p_k * (∂v_ik/∂v_jd) - Σ(k) q_k * (∂w_ik/∂v_jd)]
                  - 2λ(width - 1) * ∂width/∂v_jd
        """
        p, v = agent_u
        q, w = agent_v
        grad_v = np.zeros_like(v)
        
        # Pre-compute softmax gradients for all customers
        for i, customer in enumerate(self.customers):
            dists_p = np.sum((customer - v) ** 2, axis=1)
            dists_q = np.sum((customer - w) ** 2, axis=1)
            all_dists = np.concatenate([-dists_p, -dists_q])
            softmax_all = self._softmax(all_dists)
            
            # Jacobian of softmax: ∂softmax_i/∂dist_j = softmax_i(δ_ij - softmax_j)
            softmax_grad = np.diag(softmax_all) - np.outer(softmax_all, softmax_all)
            
            # Compute gradient contribution from this customer
            for j in range(self.k):
                dist_j_deriv = 2 * (customer - v[j])  # ∂(-||c_i - v_j||²)/∂v_j = 2(c_i - v_j)
                
                # Contribution from agent p's servers
                for k in range(self.k):
                    grad_v[j] += p[k] * softmax_grad[k, j] * dist_j_deriv
                
                # Contribution from agent q's servers
                for k in range(self.k):
                    grad_v[j] -= q[k] * softmax_grad[self.k + k, j] * dist_j_deriv
        
        # Add penalty term gradient if width constraint is enforced
        if self.enforce_width_constraint:
            width = self._compute_width(p, v)
            _, width_grad_v = self._compute_width_gradient(p, v)
            grad_v -= 2 * self.width_penalty_lambda * (width - 1.0) * width_grad_v
        
        return grad_v
    
    def _compute_gradient(self, agent_u: Tuple[np.ndarray, np.ndarray],
                          agent_v: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute unconstrained gradient of payoff with respect to agent_u."""
        grad_p = self._compute_gradient_p(agent_u, agent_v)
        if self.optimize_server_positions:
            grad_v = self._compute_gradient_v(agent_u, agent_v)
        else:
            p, v = agent_u
            grad_v = np.zeros_like(v)
        return grad_p, grad_v
    
    def improve(self, agent_u: Tuple[np.ndarray, np.ndarray],
                agent_v: Tuple[np.ndarray, np.ndarray],
                *, learning_rate: float = 0.01, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Improve agent_u using PROJECTED GRADIENT ASCENT against agent_v.
        
        Algorithm:
        1. Compute unconstrained gradient: grad = ∇f(x)
        2. Gradient ascent step: x_step = x + η * grad
        3. Project onto feasible region: x_new = proj(x_step)
        
        For p: Feasible region is probability simplex {p : p >= 0, Σp_i = 1}
        For v: No projection needed (width constraint handled via penalty in gradient)
        """
        p, v = agent_u
        
        # Step 1: Compute unconstrained gradient
        grad_p, grad_v = self._compute_gradient(agent_u, agent_v)
        
        # Step 2: Gradient ascent step
        p_step = p + learning_rate * grad_p
        v_step = v + learning_rate * grad_v if self.optimize_server_positions else v.copy()
        
        # Step 3: Project onto feasible region
        # For p: Project onto probability simplex
        p_new = self._project_simplex(p_step)
        
        # For v: No projection (width constraint enforced via penalty term only)
        v_new = v_step
        
        return (p_new, v_new)
    
    def create_random_agent(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create a random agent with normalized mass and width=1."""
        p = np.random.uniform(0.1, 1.0, size=self.k)
        p = self._project_simplex(p)  # Ensure valid probability distribution
        
        v = np.random.uniform(-1.0, 1.0, size=(self.k, 2))
        if self.enforce_width_constraint:
            v = self._normalize_width(p, v)
        return (p, v)


if __name__ == "__main__":
    def demo(improvement_function, gif_file_name="demo_PSRO"):
        """Demo function for Differentiable Lotto game."""
        os.makedirs("demos/blotto", exist_ok=True)
        
        game = DifferentiableLotto(
            num_customers=9, num_servers=3, customer_scale=1.0,
            optimize_server_positions=True, enforce_width_constraint=True,
            width_penalty_lambda=1
        )
        num_iterations = 100
        
        agent1 = game.create_random_agent()
        agent2 = game.create_random_agent()
        agent3 = game.create_random_agent()
        
        print("=" * 70)
        print("PROJECTED GRADIENT ASCENT DEMO")
        print("=" * 70)
        print("Initial states:")
        print("=" * 70)
        for i, agent in enumerate([agent1, agent2, agent3], 1):
            print(f"  Agent {i}: p={agent[0]}, width={game._compute_width(agent[0], agent[1]):.4f}")
        
        payoffs = [
            game.play(agent1, agent2),
            game.play(agent1, agent3),
            game.play(agent2, agent3)
        ]
        print(f"\nInitial payoffs:")
        print(f"  Agent 1 vs Agent 2: {payoffs[0]:.4f}")
        print(f"  Agent 1 vs Agent 3: {payoffs[1]:.4f}")
        print(f"  Agent 2 vs Agent 3: {payoffs[2]:.4f}")
        
        agents_history = [[agent1, agent2, agent3]]
        for _ in trange(num_iterations, desc="Iterations"):
            population = [agent1, agent2, agent3]
            agent1 = improvement_function(0, population, game)
            agent2 = improvement_function(1, population, game)
            agent3 = improvement_function(2, population, game)
            agents_history.append([agent1, agent2, agent3])
        
        payoffs = [
            game.play(agent1, agent2),
            game.play(agent1, agent3),
            game.play(agent2, agent3)
        ]
        print(f"\nFinal payoffs:")
        print(f"  Agent 1 vs Agent 2: {payoffs[0]:.4f}")
        print(f"  Agent 1 vs Agent 3: {payoffs[1]:.4f}")
        print(f"  Agent 2 vs Agent 3: {payoffs[2]:.4f}")
        
        print("\n" + "=" * 70)
        print("Final states:")
        print("=" * 70)
        for i, agent in enumerate([agent1, agent2, agent3], 1):
            print(f"  Agent {i}: p={agent[0]}, width={game._compute_width(agent[0], agent[1]):.4f}")
        
        try:
            from games.blotto.differentiable_lotto_vis import gif_from_matchups
            gif_path = gif_from_matchups(
                game, agents_history, path=f"demos/blotto/{gif_file_name}.gif",
                fps=20, stride=1, dpi=120, show_customers=True,
                show_gradients=True, gradient_scale=0.3
            )
            print(f"\nSaved visualization GIF: {gif_path}")
        except ImportError:
            print("\nNote: Visualization module not available.")
        except Exception as e:
            print(f"\nNote: Could not generate visualization: {e}")
    
    demo(run_PSRO_uniform_weaker, "demo_PSRO_u_weaker")
    print("\n" + "=" * 70 + "\n")
    demo(run_PSRO_uniform_stronger, "demo_PSRO_u_stronger")