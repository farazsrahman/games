# Experiment Analysis: Four Games in Balduzzi et al. (2019)

This document explains the purpose of each of the four game experiments and how they relate to the paper "Open-Ended Learning in Symmetric Zero-Sum Games" by Balduzzi et al. (2019).

## Overview

The paper introduces **Policy Space Response Oracles (PSRO)**, a framework for training populations of agents in symmetric zero-sum games. The key contribution is demonstrating how different **improvement strategies** affect population diversity and convergence:

- **PSRO_uniform**: Samples uniformly from all opponents
- **PSRO_uniform_weaker**: Samples from opponents the agent beats (maintains diversity)
- **PSRO_uniform_stronger**: Samples from opponents that beat the agent (causes convergence)

Each game serves a different purpose in validating and demonstrating these PSRO variants.

---

## 1. 🎯 Disc Game

### Purpose
The Disc Game is the **primary demonstration** of how different PSRO strategies affect population diversity. It provides a simple, continuous game where the effects of improvement strategies are visually clear.

### Game Mechanics
- **Action Space**: Continuous 2D unit disc (agents are points in R²)
- **Payoff**: `u^T A v` where `A = [[0, -1], [1, 0]]` (antisymmetric matrix)
- **Initialization**: Three agents start at Rock-Paper-Scissors triangle vertices
- **Improvement**: Gradient ascent in the direction `A @ v` (where v is the opponent)

### Why It Fits the Paper
This game directly illustrates the paper's **main theoretical result**:
- **PSRO_uniform_weaker**: Agents maintain diverse positions on the disc, forming a stable cycle
- **PSRO_uniform_stronger**: All agents converge to the same point (loss of diversity)

The disc game is ideal because:
1. **Visual clarity**: The 2D embedding makes convergence vs. diversity immediately apparent
2. **Continuous action space**: Demonstrates PSRO works beyond discrete games
3. **Simple dynamics**: The antisymmetric payoff structure creates clear cyclic behavior

### Outputs Generated
1. **Training plot** (`disc_PSRO_{variant}.png`): Shows agent positions over time
2. **Population GIF** (`disc_PSRO_{variant}_population.gif`): Animated visualization of agents moving on the unit disc
3. **EGS Visualizations** (PCA, Schur, SVD, t-SNE): Low-dimensional embeddings of the gamescape showing agent relationships

### Key Insight for Paper
Demonstrates that **sampling from weaker opponents preserves diversity**, while **sampling from stronger opponents causes premature convergence**—the core finding of the paper.

---

## 2. ⚔️ Colonel Blotto Game

### Purpose
The Blotto game demonstrates PSRO on a **discrete, strategic resource allocation game** with a large but finite action space. It shows how PSRO scales to more complex games with strategic depth.

### Game Mechanics
- **Action Space**: Discrete allocations of a fixed budget across multiple battlefields
  - For 3 battlefields and budget 10: `C(10+3-1, 3-1) = 66` possible allocations
- **Payoff**: Agent wins a battlefield if they allocate more resources; total wins determine payoff
- **Agent Representation**: LogitAgent with logits over all possible allocations
- **Improvement**: MPO-style update using rollouts against opponents

### Why It Fits the Paper
1. **Discrete action space**: Complements the continuous disc game, showing PSRO works across game types
2. **Strategic complexity**: Multiple battlefields create non-trivial strategic interactions
3. **Scalability**: Demonstrates PSRO can handle games with many possible strategies (66+ actions)
4. **Population dynamics**: Shows how different PSRO variants affect strategy diversity in a strategic game

### Outputs Generated
1. **Training plot** (`blotto_PSRO_{variant}.png`): Win rates between agent pairs over iterations
2. **Population GIF** (`blotto_PSRO_{variant}_population.gif`): Shows expected allocations and entropy over time
3. **Matchups GIF** (`blotto_PSRO_{variant}_matchups.gif`): Win rate heatmaps between all agent pairs over time
4. **EGS Visualizations**: Gamescape embeddings showing strategic relationships

### Key Insight for Paper
Validates that PSRO improvement strategies work on **real strategic games** (not just toy examples), and shows how population diversity affects the discovery of effective strategies in competitive settings.

---

## 3. 🎲 Differentiable Lotto

### Purpose
The Differentiable Lotto demonstrates PSRO on a **continuous optimization game** with geometric constraints. It shows how PSRO handles games where agents optimize both discrete choices (mass distribution) and continuous parameters (server positions).

### Game Mechanics
- **Action Space**: 
  - `p`: Probability distribution over k servers (discrete choice)
  - `v`: k×2 matrix of server positions in R² (continuous optimization)
- **Payoff**: Based on soft customer assignments to nearest servers
- **Constraints**: Optional width constraint (penalty for servers being too spread out)
- **Improvement**: Projected gradient ascent on both p and v

### Why It Fits the Paper
1. **Hybrid action space**: Combines discrete (mass allocation) and continuous (server positions) optimization
2. **Geometric interpretation**: Server positions create visualizable strategies in 2D space
3. **Constraint handling**: Demonstrates PSRO with constrained optimization (width penalty)
4. **Real-world relevance**: Models competitive facility location problems

### Outputs Generated
1. **Training plot** (`diff_lotto_PSRO_{variant}.png`): Payoffs and width constraints over time
2. **Population GIF** (`diff_lotto_PSRO_{variant}_population.gif`): Shows server positions and customer assignments evolving
3. **Matchups GIF** (`diff_lotto_PSRO_{variant}_matchups.gif`): Win rates between agents over time
4. **EGS Visualizations**: Gamescape showing strategic relationships in server placement space

### Key Insight for Paper
Shows PSRO can handle **complex continuous games with constraints**, and demonstrates how different improvement strategies affect the exploration of the strategy space in optimization problems.

---

## 4. 🪙 Penney's Game

### Purpose
Penney's Game demonstrates PSRO on a **non-transitive game** with probabilistic outcomes. It shows how PSRO handles games where the relationship between strategies is non-transitive (A beats B, B beats C, but C beats A).

### Game Mechanics
- **Action Space**: Probability distribution over 2^k possible H/T sequences (e.g., 8 sequences for length 3)
- **Gameplay**: Two players choose sequences; coin is flipped repeatedly; first sequence to appear wins
- **Non-transitivity**: For any sequence, there exists a sequence that beats it with probability > 0.5
- **Agent Representation**: LogitAgent with logits over all sequences
- **Improvement**: Gradient-based updates using win probabilities

### Why It Fits the Paper
1. **Non-transitive structure**: Demonstrates PSRO on games without clear hierarchy
2. **Probabilistic outcomes**: Shows PSRO works with stochastic games (not just deterministic)
3. **Cyclic relationships**: Similar to disc game, but in discrete space with probabilistic payoffs
4. **Population diversity**: Different PSRO variants will show different patterns of sequence discovery

### Outputs Generated
1. **Training plot** (`penneys_PSRO_{variant}.png`): Win rates and sequence probabilities over time
2. **Population GIF** (`penneys_PSRO_{variant}_population.gif`): Shows probability distributions over sequences evolving
3. **Matchups GIF** (`penneys_PSRO_{variant}_matchups.gif`): Win rates between agents over time
4. **EGS Visualizations**: Gamescape showing relationships between different sequence strategies

### Key Insight for Paper
Demonstrates PSRO's ability to handle **non-transitive games** where there's no single dominant strategy, and shows how different improvement strategies affect the discovery of cyclic strategy relationships.

---

## Common Outputs Across All Games

### EGS (Empirical Game-theoretic Strategy) Visualizations
All games generate EGS visualizations using different dimensionality reduction techniques:
- **Matrix**: Full payoff matrix between all agents
- **PCA**: Principal Component Analysis embedding
- **Schur**: Schur decomposition embedding
- **SVD**: Singular Value Decomposition embedding
- **t-SNE**: t-distributed Stochastic Neighbor Embedding

These visualizations show the **gamescape**—the geometric structure of the strategy space and how agents relate to each other.

### Why EGS Visualizations Matter
The paper emphasizes understanding the **structure of the strategy space**. These visualizations reveal:
- How agents cluster or spread out
- The dimensionality of the effective strategy space
- Relationships between different strategies
- How PSRO variants affect exploration of the gamescape

---

## Summary: How All Four Games Fit Together

1. **Disc Game**: Simple continuous demonstration of diversity vs. convergence
2. **Blotto Game**: Discrete strategic game showing scalability
3. **Differentiable Lotto**: Continuous optimization with constraints
4. **Penney's Game**: Non-transitive probabilistic game

Together, these four games validate PSRO across:
- ✅ Continuous and discrete action spaces
- ✅ Deterministic and probabilistic outcomes
- ✅ Transitive and non-transitive game structures
- ✅ Simple toy games and complex strategic games
- ✅ Unconstrained and constrained optimization

The experiments collectively demonstrate that **PSRO_uniform_weaker** maintains population diversity across diverse game types, while **PSRO_uniform_stronger** tends to cause premature convergence—the central finding of the paper.

---

## Generated Plots and Outputs

All plots and visualizations in the `demos/` folder were generated using the baseline configurations specified in `baseline_configs.py` and `STREAMLIT_BASELINE_CONFIGS.md`. For each game, **all three PSRO variants** (uniform, weaker, stronger) were run to enable direct comparison.

**Note**: The game demo scripts (e.g., `src/games/blotto/blotto.py`) have hardcoded parameter values that match these baseline configurations. When running the demos directly (via `python run.py <game>`), they use these hardcoded values. The Streamlit app and other runners can use the baseline config dictionaries from `baseline_configs.py`.

### Configuration Summary

The following baseline configurations were used to generate all plots:

#### Disc Game
- **Iterations**: 500
- **Learning Rate**: 0.01
- **Number of Agents**: 3
- **All variants**: uniform, weaker, stronger

#### Colonel Blotto Game
- **Iterations**: 1000
- **Evaluation Rounds**: 1000
- **Battlefields**: 3
- **Budget**: 10
- **Number of Agents**: 3
- **All variants**: uniform, weaker, stronger

#### Differentiable Lotto
- **Iterations**: 100
- **Customers**: 9
- **Servers**: 3
- **Number of Agents**: 3
- **Server Optimization**: Enabled
- **Width Constraint**: Enabled (λ = 1.0)
- **All variants**: uniform, weaker, stronger

#### Penney's Game
- **Iterations**: 500
- **Sequence Length**: 3 (8 possible sequences)
- **Evaluation Rounds**: 500
- **Number of Agents**: 3
- **All variants**: uniform, weaker, stronger

### What to Look For in the Plots

When examining the generated plots in `demos/`, compare across the three PSRO variants to observe:

#### 1. Training Plots (`{game}_PSRO_{variant}.png`)

**Disc Game** (`disc_PSRO_{variant}.png`):
- **What it shows**: Win rates (mapped from payoffs) between all agent pairs over 500 iterations
- **Interpretation**:
  - **Weaker variant**: Win rates should oscillate or remain balanced, showing agents maintain competitive diversity. The three lines (Agent 1 vs 2, Agent 1 vs 3, Agent 2 vs 3) may cross frequently, indicating no single agent dominates.
  - **Stronger variant**: Win rates should converge to stable values quickly, with one agent typically dominating others (one line near 1.0, others near 0.0), indicating convergence to similar strategies.
  - **Uniform variant**: Intermediate behavior—win rates stabilize but may maintain some diversity longer than stronger variant.

**Blotto Game** (`blotto_PSRO_{variant}.png`):
- **What it shows**: Win rates between all agent pairs over 1000 iterations (each agent plays 1000 rounds per evaluation)
- **Interpretation**:
  - **Weaker variant**: Win rates should show more variation and slower convergence, with agents discovering diverse allocation strategies. Lines may oscillate as agents adapt to each other's evolving strategies.
  - **Stronger variant**: Win rates should converge rapidly to stable values, indicating agents quickly find dominant strategies and stop exploring. One agent may consistently win against others.
  - **Uniform variant**: Moderate convergence speed, showing balanced exploration-exploitation trade-off.

**Differentiable Lotto** (`diff_lotto_PSRO_{variant}.png`):
- **What it shows**: Win rates (sigmoid-transformed payoffs) between all agent pairs over 100 iterations
- **Interpretation**:
  - **Weaker variant**: Win rates should show gradual evolution as agents explore different server placement strategies. The three lines may remain relatively balanced, indicating diverse server configurations.
  - **Stronger variant**: Win rates should converge quickly, with agents finding similar optimal server positions. One agent may dominate, showing convergence to a single strategy.
  - **Uniform variant**: Intermediate convergence, with agents finding good strategies but maintaining some diversity in server placements.

**Penney's Game** (`penneys_PSRO_{variant}.png`):
- **What it shows**: Win rates between all agent pairs over 500 iterations (each matchup uses 500 coin flip rounds)
- **Interpretation**:
  - **Weaker variant**: Win rates should show cyclical patterns or remain balanced, reflecting the non-transitive nature of the game. Agents may discover different sequence preferences, maintaining diversity.
  - **Stronger variant**: Win rates should converge to stable values, with agents specializing in similar sequences. The non-transitive structure may cause one agent to consistently win.
  - **Uniform variant**: Balanced exploration of the sequence space, with moderate convergence to stable win rates.

#### 2. Population GIFs (`{game}_PSRO_{variant}_population.gif`)

**Disc Game** (`disc_PSRO_{variant}_population.gif`):
- **What it shows**: Animated visualization of three agents (colored points) moving on the unit disc, with arrows showing movement direction and magnitude
- **Interpretation**:
  - **Weaker variant**: Agents should maintain distinct positions on the disc, moving in a stable cycle or maintaining separation. The three colored points stay apart, and movement vectors show continuous adaptation without convergence.
  - **Stronger variant**: All three agents should converge to the same point on the disc, with movement vectors shrinking to zero. The three colored points merge into one location.
  - **Uniform variant**: Agents may drift toward each other but maintain some separation, showing intermediate convergence behavior.

**Blotto Game** (`blotto_PSRO_{variant}_population.gif`):
- **What it shows**: Two-panel animation showing (1) expected resource allocations per battlefield as bar charts, and (2) policy entropy over time
- **Interpretation**:
  - **Weaker variant**: Allocation bars should remain distinct across agents, showing diverse strategies (e.g., Agent 1 favors battlefield 1, Agent 2 favors battlefield 2). Entropy should remain relatively high (above 2.0), indicating diverse probability distributions over the 66 possible allocations.
  - **Stronger variant**: Allocation bars should converge to similar patterns across all agents, showing they adopt similar strategies. Entropy should decrease significantly (toward 0), indicating convergence to deterministic or near-deterministic strategies.
  - **Uniform variant**: Moderate diversity in allocations and entropy, showing balanced exploration.

**Differentiable Lotto** (`diff_lotto_PSRO_{variant}_population.gif`):
- **What it shows**: Animated 2D plot showing customer positions (gray points), server positions for each agent (colored markers), and optional gradient vectors showing optimization direction
- **Interpretation**:
  - **Weaker variant**: Server positions should remain distinct across agents, with each agent finding different optimal placements. The three agents' servers should be spread out, showing diverse strategies for serving customers.
  - **Stronger variant**: Server positions should converge to similar locations across all agents, with all agents placing servers in nearly identical positions. The three agents' markers should cluster together.
  - **Uniform variant**: Intermediate server placement diversity, with some clustering but maintained separation.

**Penney's Game** (`penneys_PSRO_{variant}_population.gif`):
- **What it shows**: Two-panel animation showing (1) probability distributions over 8 sequences (HHH, HHT, HTH, HTT, THH, THT, TTH, TTT) as bar charts for each agent, and (2) policy entropy over time
- **Interpretation**:
  - **Weaker variant**: Probability distributions should remain distinct across agents, with different agents preferring different sequences. Entropy should remain relatively high (above 1.5), showing diverse sequence preferences.
  - **Stronger variant**: Probability distributions should converge to similar patterns across all agents, with all agents preferring the same sequences. Entropy should decrease, showing convergence to specific sequence choices.
  - **Uniform variant**: Moderate diversity in sequence preferences and entropy, showing balanced exploration of the sequence space.

#### 3. Matchups GIFs (`{game}_PSRO_{variant}_matchups.gif`)

**Blotto, Differentiable Lotto, and Penney's Games**:
- **What it shows**: Animated heatmap showing win rates between all agent pairs over time (3×3 matrix for 3 agents)
- **Interpretation**:
  - **Weaker variant**: The heatmap should show balanced colors (near 0.5 win rates) or oscillating patterns, indicating competitive matchups. The matrix should show that no single agent consistently dominates, reflecting diverse strategies.
  - **Stronger variant**: The heatmap should quickly stabilize to extreme values (near 0.0 or 1.0), with one agent consistently winning against others. The matrix should show clear dominance patterns, indicating convergence.
  - **Uniform variant**: Moderate stabilization with some balance remaining, showing intermediate competitive dynamics.

#### 4. EGS Visualizations (`{game}_PSRO_{variant}_egs_{method}.png`)

**All Games** (PCA, Schur, SVD, t-SNE methods):
- **What it shows**: Two-panel visualization:
  - **Left panel**: EGS matrix heatmap showing payoffs between all agent pairs (green = positive payoff, red = negative payoff)
  - **Right panel**: 2D embedding of agents colored by row average payoff, with convex hull showing the area covered

**Detailed Analysis by Component:**

**Left Panel - EGS Matrix Heatmap:**
- **Weaker variant**: 
  - Matrix should show **balanced colors** (mix of green and red), indicating competitive matchups where no single agent dominates
  - Values should be **moderate** (not extreme), typically in the range [-0.3, 0.3], showing agents are relatively evenly matched
  - The matrix should appear **symmetric** in pattern (since it's antisymmetric: M[i,j] = -M[j,i])
  - **Interpretation**: Diverse strategies lead to varied competitive outcomes, with each agent having strengths and weaknesses against different opponents

- **Stronger variant**:
  - Matrix should show **extreme values** (mostly bright green or bright red), indicating clear dominance patterns
  - Values should be **closer to ±0.5** (near maximum), showing one agent consistently wins or loses
  - The matrix may show **clear dominance structure** (e.g., Agent 1 beats Agent 2 and 3, Agent 2 beats Agent 3)
  - **Interpretation**: Converged strategies create clear hierarchy, with dominant agents consistently winning

- **Uniform variant**:
  - Matrix should show **moderate values** with some structure, intermediate between weaker and stronger
  - Values typically in range [-0.4, 0.4]
  - **Interpretation**: Balanced exploration leads to some competitive structure but maintains diversity

**Right Panel - 2D Embeddings:**

**Key Metrics to Observe:**
1. **Convex Hull Area** (shown in title): Area covered by the agent embeddings
2. **Point Spread**: Distance between agent points
3. **Row Average Colors**: How well each agent performs on average (viridis colormap)

- **Weaker variant**:
  - **Large convex hull area** (typically > 0.1 for 3 agents): Agents are spread out in the gamescape
  - **Wide point separation**: Three distinct points with significant distance between them
  - **Varied row average colors**: Different agents may have different average performance (different colors), indicating diverse competitive profiles
  - **Geometric interpretation**: The gamescape has high dimensionality/volume, reflecting diverse strategy space exploration

- **Stronger variant**:
  - **Small convex hull area** (typically < 0.01 for 3 agents): Agents cluster tightly
  - **Tight point clustering**: All three points nearly overlap or form a very tight cluster
  - **Similar row average colors**: All agents have similar average performance (similar colors), indicating convergence to similar strategies
  - **Geometric interpretation**: The gamescape collapses to low dimensionality, reflecting strategy convergence

- **Uniform variant**:
  - **Medium convex hull area** (typically 0.01-0.1): Intermediate spread
  - **Moderate point separation**: Points are closer than weaker but more separated than stronger
  - **Mixed row averages**: Some variation in agent performance
  - **Geometric interpretation**: Moderate gamescape volume, showing balanced exploration

**Embedding Method Differences:**
- **PCA**: Linear dimensionality reduction preserving maximum variance. Best for seeing overall structure.
- **SVD**: Similar to PCA but optimized for matrix decomposition. Often shows similar structure to PCA.
- **Schur**: Based on Schur decomposition of the antisymmetric matrix. May reveal cyclic structures better.
- **t-SNE**: Non-linear embedding preserving local neighborhoods. May show different clustering patterns, especially useful for non-linear gamescape structures.

**What to Compare Across Variants:**
1. **Convex hull area**: Larger = more diversity (weaker > uniform > stronger)
2. **Point spread**: Wider = more diversity
3. **Matrix values**: More extreme = more convergence (stronger), more balanced = more diversity (weaker)
4. **Row average variance**: More variation = more diverse competitive profiles (weaker), less variation = convergence (stronger)

**Game-Specific Patterns:**
- **Disc Game**: Embeddings should reflect the cyclic structure of the game. Weaker variant may show agents forming a triangle or cycle, stronger variant collapses to a point.
- **Blotto Game**: Embeddings reflect strategic diversity in allocation strategies. Weaker variant shows diverse allocation patterns, stronger variant shows convergence to similar allocations.
- **Differentiable Lotto**: Embeddings reflect server placement diversity. Weaker variant shows diverse server configurations, stronger variant shows convergence to similar placements.
- **Penney's Game**: Embeddings reflect sequence preference diversity. Weaker variant shows diverse sequence strategies, stronger variant shows convergence to similar sequences.

### Interpreting the Results

When comparing plots across variants:

1. **Diversity Preservation**: The `weaker` variant should show agents maintaining distinct strategies throughout training, visible in:
   - Population GIFs showing agents staying separated
   - EGS visualizations with wider agent spread
   - Training plots showing different trajectories for each agent

2. **Premature Convergence**: The `stronger` variant should show agents converging to similar strategies, visible in:
   - Population GIFs showing agents moving toward the same region
   - EGS visualizations with tight clustering
   - Training plots showing similar trajectories for all agents

3. **Baseline Behavior**: The `uniform` variant provides a baseline comparison, typically showing intermediate behavior between weaker and stronger variants.

### File Organization

All outputs are organized in `demos/{game_name}/` with the following naming convention:
- `{game}_PSRO_{variant}.png` - Training plot
- `{game}_PSRO_{variant}_population.gif` - Population evolution animation
- `{game}_PSRO_{variant}_matchups.gif` - Matchup dynamics animation (where applicable)
- `{game}_PSRO_{variant}_egs_{method}.png` - EGS visualization (PCA, Schur, SVD, t-SNE)

This organization allows for easy side-by-side comparison of the three PSRO variants across all four games, demonstrating the consistent effect of improvement strategies on population diversity.

