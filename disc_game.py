import numpy as np
from tqdm import trange
from game import Game, contract, run_PSRO_uniform, run_PSRO_uniform_weaker, run_PSRO_uniform_stronger
from disc_game_vis import gif_from_population

"""
This file introduces DiscGame as discussed in Balduzzi et. al. (https://arxiv.org/pdf/1901.08106)
Running the demo with PSRO_uniform_stronger will result in all players converging to the same distribution.
Running the demo with PSRO_uniform_weaker will result in a diverse distribution.
"""

class DiscGame(Game):
    def __init__(self):
        self.A = np.array([[0, -1], [1, 0]])

    def play(self, u, v) -> float:
        return float(u.T @ self.A @ v > 0) * 2 - 1

    def improve(self, u, v, *, learning_rate: float = 0.1) -> float:
        descent_dir = self.A @ v
        u_new = u + learning_rate * descent_dir
        u_new = contract(u_new)
        return u_new


if __name__ == "__main__":

    def get_RPS_triangle():
        return [
            np.array([0.5, 0]),
            np.array([-np.sqrt(1)/4,  np.sqrt(3)/4]),
            np.array([-np.sqrt(1)/4, -np.sqrt(3)/4])
        ]

    def demo_disc_game(improvement_function, gif_file_name="demo_PSRO_disc_game"):

        game = DiscGame()
        learning_rate = 0.01
        num_iterations = 500

        rock, paper, scissors = [agent.copy() for agent in get_RPS_triangle()]
        print("=" * 70)
        print("Initial states:")
        print("=" * 70)
        print(f"  Rock:     {rock}")
        print(f"  Paper:    {paper}")
        print(f"  Scissors: {scissors}")

        agents_history = [[rock.copy(), paper.copy(), scissors.copy()]]

        for iteration in trange(num_iterations, desc="Iterations"):
            population  = [rock, paper, scissors] 
            rock        = improvement_function(0, population, game)
            paper       = improvement_function(1, population, game)
            scissors    = improvement_function(2, population, game)
            agents_history.append([rock.copy(), paper.copy(), scissors.copy()])

        # Create GIF
        gif_path = gif_from_population(
            np.array(agents_history),
            path=gif_file_name + ".gif",
            fps=20,
            stride=1,
            dpi=120,
            unit_circle=True,
            normalize_difference_vector = .3
        )
        print(f"\nSaved GIF: {gif_path}")

        # Print final states
        print("\n" + "=" * 70)
        print("Final states:")
        print("=" * 70)
        print(f"  Rock:     {rock}")
        print(f"  Paper:    {paper}")
        print(f"  Scissors: {scissors}")

    demo_disc_game(run_PSRO_uniform_weaker,   "demo_PSRO_u_weaker")
    demo_disc_game(run_PSRO_uniform_stronger, "demo_PSRO_u_stronger")






## OLD CODE - Leaving this for now


# ####################################


# A = np.array([[0, -1], [1, 0]])

# def descent_dir(v: np.ndarray, w: np.ndarray) -> np.ndarray:
#     return A @ w, A @ v

# def step(v: np.ndarray, dv: np.ndarray, eta_v: np.ndarray):
#     return v + eta_v * dv

# def contract(v: np.ndarray):
#     norm_v = np.linalg.norm(v)
#     if norm_v >= 1:
#         v = v / norm_v
#     return v

# def eval(v: np.ndarray, w: np.ndarray) -> float:
#     return v.T @ A @ w



# get_eta_v = lambda: .1
# get_eta_w = lambda: .1
# # get_eta_v = lambda: np.random.uniform()
# # get_eta_w = lambda: np.random.uniform()

# v = np.array([ .4, .3])
# w = np.array([ .4, .2])

# # note this is not exactly uniform on the circle
# v = np.array([1, 0])
# w = np.array([-.5, 0])
# v = np.random.uniform(size=(2,)) * 2 - 1
# w = np.random.uniform(size=(2,)) * 2 - 1
# v = contract(v)
# w = contract(w)

# imgs = []
# vs, ws, dvs, dws = [], [], [], []
# vals = []
# cums = []


# for i in trange(1000):
#     eta_v, eta_w = get_eta_v(), get_eta_w()
#     dv, dw = descent_dir(v, w)

#     vs.append(v)
#     ws.append(w)
#     dvs.append(dv)
#     dws.append(dw)

#     v = step(v, dv, eta_v)
#     w = step(w, dw, eta_w)

#     v = contract(v)
#     w = contract(w)
#     vals.append(eval(v, w))
#     cums.append(sum(vals))

# # gif_path = images_to_gif(imgs, "descent.gif", duration=120)
# # gif_path = gif_from_states(vs, ws, dvs=dvs, dws=dws, A=A,
#                         #    path="descent.gif", fps=5, stride=1, dpi=30)
# # print("Saved:", gif_path)

# # breakpoint()
# """
# I want to implement the PSRO algorithm here,
# what does this mean

# I need a main loop iterating the population, inside I need
#     I need a nash calculator (generally a meta-strategy solver
#     I need a policy improvement oracle that looks like oracle: NashEq, Population |-> new agent
#     I need to play the new policy against every one else to add to the evaluation matrix


# FIRST TODO: Implement a self-play loop
#     - This should be kind of weird cause it should litterally just add agents in a circle!

# FIRST.5 TODO: Implement the policy improvement Oracle given a nash eq. (would be sum of gradients (or even average opponent since everything is linear?))
#     - 

# SECOND TODO: Create the basic data structures for PSRO in the self-play loop


# THIRD TODO: Compute the necessary statistics (i.e. NashEq given the data srtuctures and test that methods are robust, diversiy measure)


# """


# # def get_nash(A_B: np.ndarray) -> np.ndarray:
# #     # call a subroutine to solve the Matrix game 
# #     # to get a nash equilibrium (or whatever other variant 
# #     # of
# #     # PSRO we might be interested in
# #     pass


### Some initial populations
def get_RPS_triangle():
    return [
        np.array([0.5, 0]),
        np.array([-np.sqrt(1)/4,  np.sqrt(3)/4]),
        np.array([-np.sqrt(1)/4, -np.sqrt(3)/4])
    ]
### 

# # population = [sample_unit_hypersphere(1)[0] for _ in range(50)]
# population = get_RPS_triangle()
# dvs = [np.zeros_like(v) for v in population[:-1]]

# def get_agent_self_play(vt: np.ndarray, eta: float) -> np.ndarray:

#     vtp1 = vt.copy()
#     _, dvtp1 = descent_dir(vt, vtp1)
#     vtp1 = step(vtp1, dvtp1, eta)
#     vtp1 = contract(vtp1)

#     return vtp1

# def get_PSRO_agent(population: list[np.ndarray], eta: float) -> np.ndarray:
#     population_np = np.stack(population)
#     descent_dirs = (A @ population_np.T).T # altenatively population_np @ A.T
#     # descent_dirs = descent_dirs[-1:]
#     # idxs = np.random.choice(descent_dirs.shape[0], min(1, descent_dirs.shape[0]), replace=False)
#     # descent_dirs = descent_dirs[idxs]

#     # assume uniform PSRO for now
#     descent_dir_mean = np.mean(descent_dirs, axis=0)
#     vtp1 = step(population_np[-1], descent_dir_mean, eta)
#     vtp1 = contract(vtp1)

#     return vtp1, descent_dir_mean


# for i in trange(4):
#     # vtp1 = get_agent_self_play(population[-1], 0.01)
#     # population.append(vtp1)

#     vtp1, dirs_mean = get_PSRO_agent(population, 0.1)
#     population.append(vtp1)
#     dvs.append(dirs_mean)

# # dvs.append(np.array([0, 0]))
# dvinds = [A @ v for v in population ]
# image = plot_image(population, dvinds)
# image.save("debug.png")








# if __name__ == "__main__":
#     # Import required functions
#     from disc_game import get_RPS_triangle, gif_from_population

#     # Initialize game and parameters
#     game = DiscGame()
#     learning_rate = 0.01
#     num_iterations = 500

#     # Initialize 3 agents individually as rock, paper, scissors
#     rock, paper, scissors = [agent.copy() for agent in get_RPS_triangle()]
#     print("=" * 70)
#     print("Initial states:")
#     print("=" * 70)
#     print(f"  Rock:     {rock}")
#     print(f"  Paper:    {paper}")
#     print(f"  Scissors: {scissors}")

#     # Track evolution of all agents: order is [rock, paper, scissors]
#     agents_history = [[rock.copy(), paper.copy(), scissors.copy()]]

#     # Update agents regardless of play value
#     for iteration in trange(num_iterations, desc="Iterations"):
#         # Create copies for this iteration
#         new_rock     = rock.copy()
#         new_paper    = paper.copy()
#         new_scissors = scissors.copy()

#         condition = lambda u, v: game.play(u, v) < 0

#         # RPS cyclic tournament logic with variable names
#         # First round
#         if condition(new_rock, new_paper):
#             new_rock = game.improve(new_rock, new_paper, learning_rate=learning_rate)
#         if condition(new_paper, new_scissors):
#             new_paper = game.improve(new_paper, new_scissors, learning_rate=learning_rate)
#         if condition(new_scissors, new_rock):
#             new_scissors = game.improve(new_scissors, new_rock, learning_rate=learning_rate)
#         # Second round
#         if condition(new_rock, new_scissors):
#             new_rock = game.improve(new_rock, new_scissors, learning_rate=learning_rate)
#         if condition(new_paper, new_rock):
#             new_paper = game.improve(new_paper, new_rock, learning_rate=learning_rate)
#         if condition(new_scissors, new_paper):
#             new_scissors = game.improve(new_scissors, new_paper, learning_rate=learning_rate)

#         # Update agents
#         rock, paper, scissors = new_rock, new_paper, new_scissors
#         agents_history.append([rock.copy(), paper.copy(), scissors.copy()])

#     # Convert to numpy arrays for visualization
#     agents_array = np.array(agents_history)  # Shape: (num_iterations+1, 3, 2)

#     # Create GIF - show all agents evolving
#     # dus will be computed automatically from first-order differences
#     gif_path = gif_from_population(
#         agents_array,
#         path="case2_unconditional_updates.gif",
#         fps=20,
#         stride=1,
#         dpi=120,
#         unit_circle=True,
#         normalize_difference_vector = .3
#     )
#     print(f"\nSaved GIF: {gif_path}")

#     # Print final states
#     print("\n" + "=" * 70)
#     print("Final states:")
#     print("=" * 70)
#     print(f"  Rock:     {rock}")
#     print(f"  Paper:    {paper}")
#     print(f"  Scissors: {scissors}")
