# Lecture 2: Process Parameter Optimization in Additive Manufacturing

This directory contains all materials for the second lecture of the ML for AM course.

## 📋 Contents

### Code Demonstrations
1. **`generate_data.py`** - Generate synthetic datasets for all demos
2. **`01_regression_comparison.py`** - Compare regression algorithms for surface roughness prediction
3. **`02_gaussian_process_demo.py`** - Gaussian Process with uncertainty quantification
4. **`03_bayesian_optimization.py`** - Bayesian Optimization for parameter tuning

### Generated Outputs
- **`data/`** - Synthetic datasets (created by `generate_data.py`)
- **`outputs/`** - Visualizations and results from running demos

## 🚀 Getting Started

### 1. Prerequisites

Make sure you've completed Lecture 1 and have the environment set up:

```bash
source .venv/bin/activate  # Activate virtual environment
```

### 2. Generate Datasets

```bash
cd lecture_02
python generate_data.py
```

This creates three datasets in `data/`:
- `slm_surface_roughness.csv` - SLM parameters → surface roughness (150 samples)
- `print_adhesion_strength.csv` - Speed + temp → adhesion strength (30 samples)
- `slm_porosity_optimization.csv` - 4D parameter space → porosity (200 samples)

### 3. Run Demonstrations

Run demos in order during the lecture:

```bash
# Demo 1: Regression Comparison (after Slide 8, ~20 min)
python 01_regression_comparison.py

# Demo 2: Gaussian Process (after Slide 12, ~35 min)
python 02_gaussian_process_demo.py

# Demo 3: Bayesian Optimization (after Slide 19, ~50 min)
python 03_bayesian_optimization.py
```

Each script produces:
- Detailed console logging showing the optimization workflow
- Multiple visualization files saved to `outputs/`

## 📊 Demo Overview

### Demo 1: Regression Model Comparison
**Learning Goals:**
- Compare 5 regression algorithms (Linear, Polynomial, RF, GB, SVR)
- Understand trade-offs: accuracy vs speed vs interpretability
- See which algorithms excel for AM problems
- Why we need uncertainty for optimization

**Key Outputs:**
- Model comparison charts (R², RMSE, training time)
- Predicted vs actual plots
- Feature importance analysis

**Duration:** ~4 minutes

**Dataset:** 150 SLM experiments with 4 process parameters

---

### Demo 2: Gaussian Process Regression
**Learning Goals:**
- Understand GP predictions with confidence intervals
- Compare GP with Random Forest (no uncertainty)
- Visualize uncertainty in sparse data regions
- See how uncertainty guides experimental design

**Key Outputs:**
- GP predictions with 68% and 95% confidence intervals
- Comparison with RF (no uncertainty)
- Uncertainty map showing where to sample next
- Acquisition strategy demonstration

**Duration:** ~4 minutes

**Dataset:** 30 adhesion strength measurements (small dataset!)

---

### Demo 3: Bayesian Optimization
**Learning Goals:**
- See complete BO algorithm in action
- Understand exploration vs exploitation trade-off
- Compare BO efficiency vs random search
- Visualize convergence to optimum

**Key Outputs:**
- Convergence plot (BO vs random search)
- Parameter space exploration visualization
- Improvement per iteration analysis
- Final optimal parameters

**Duration:** ~5 minutes

**Dataset:** 4D optimization problem (laser power, speed, hatch spacing, layer thickness)

## 📧 Contact

**Instructor:** Nazarii Mediukh, PhD  
**Email:** n.mediukh@ipms.kyiv.ua  
**Institution:** Institute for Problems of Materials Science, NASU

---

**Next:** Proceed to Lecture 3 (Advanced Applications) after completing this lecture.

**Prerequisites:** Completion of Lecture 1 recommended.
