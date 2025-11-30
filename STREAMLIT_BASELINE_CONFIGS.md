# Baseline Configurations for Streamlit App

Use these exact values in the Streamlit app for baseline experiments.

---

## 🎯 Disc Game

### Baseline Configuration
- **Improvement Strategy**: `uniform` (or use "Run All Variations" button)
- **Number of Iterations**: `500`
- **Learning Rate**: `0.01`
- **Number of Agents**: `3`
- **FPS**: `20`
- **DPI**: `120`

**Quick Test** (faster):
- **Number of Iterations**: `200`
- Everything else same as above

**Full Run** (better results):
- **Number of Iterations**: `1000`
- Everything else same as above

---

## ⚔️ Colonel Blotto Game

### Baseline Configuration
- **Improvement Strategy**: `uniform` (or use "Run All Variations" button)
- **Number of Iterations**: `500`
- **Evaluation Rounds**: `1000`
- **Number of Battlefields**: `3`
- **Budget**: `10`
- **Number of Agents**: `3`

**Quick Test** (faster):
- **Number of Iterations**: `300`
- **Evaluation Rounds**: `500`
- **Budget**: `8`
- Everything else same as above

**Full Run** (better results):
- **Number of Iterations**: `1000`
- **Evaluation Rounds**: `2000`
- **Number of Battlefields**: `4`
- Everything else same as above

---

## 🎲 Differentiable Lotto

### Baseline Configuration
- **Improvement Strategy**: `weaker` (or use "Run All Variations" button)
- **Number of Iterations**: `100`
- **Number of Customers**: `9`
- **Number of Servers**: `3`
- **Number of Agents**: `3`
- **Evaluation Rounds**: `1000`
- **Optimize Server Positions**: ✅ (checked)
- **Enforce Width Constraint**: ✅ (checked)
- **Width Penalty λ**: `1.0`
- **FPS**: `20`
- **DPI**: `120`

**Quick Test** (faster):
- **Number of Iterations**: `50`
- **Number of Customers**: `6`
- **Number of Servers**: `2`
- **Evaluation Rounds**: `500`
- Everything else same as above

**Full Run** (better results):
- **Number of Iterations**: `200`
- **Number of Customers**: `12`
- **Number of Servers**: `4`
- **Evaluation Rounds**: `2000`
- Everything else same as above

---

## 🪙 Penney's Game

### Baseline Configuration
- **PSRO Strategy**: `uniform` (or use "Run All Variations" button)
- **Number of Iterations**: `500`
- **Sequence Length**: `3` (2^3 = 8 possible sequences: HHH, HHT, HTH, HTT, THH, THT, TTH, TTT)
- **Evaluation Rounds**: `500`
- **Number of Agents**: `3`

**Quick Test** (faster):
- **Number of Iterations**: `300`
- **Evaluation Rounds**: `300`
- Everything else same as above

**Full Run** (better results):
- **Number of Iterations**: `1000`
- **Evaluation Rounds**: `1000`
- Everything else same as above

**Note**: Sequence length 3 is standard (8 sequences). Length 2 gives 4 sequences (too simple), length 4 gives 16 sequences (larger action space, slower).

---

## 📋 Recommended Baseline Run Sequence

For a complete baseline comparison, run **all three PSRO variants** on each game:

### 1. Disc Game
- Click "🔄 Run All Variations" button (runs uniform, weaker, stronger automatically)
- Use baseline config above

### 2. Colonel Blotto
- Click "🔄 Run All Variations" button
- Use baseline config above

### 3. Differentiable Lotto
- Click "🔄 Run All Variations" button
- Use baseline config above

### 4. Penney's Game
- Click "🔄 Run All Variations" button
- Use baseline config above

This gives you **12 total runs** (4 games × 3 variants) for comprehensive comparison.

---

## ⚡ Quick Baseline (for testing)

If you want to test quickly before running full baselines:

1. **Disc Game**: 200 iterations
2. **Colonel Blotto**: 300 iterations, 500 rounds, budget 8
3. **Differentiable Lotto**: 50 iterations, 6 customers, 2 servers
4. **Penney's Game**: 300 iterations, 300 rounds

---

## 💡 Tips

- Use the **"🔄 Run All Variations"** button to automatically run all 3 PSRO variants (uniform, weaker, stronger) with the same settings
- Results are saved in `demos/` folder and displayed in the app
- Each run generates: training plots, GIFs, and EGS visualizations

