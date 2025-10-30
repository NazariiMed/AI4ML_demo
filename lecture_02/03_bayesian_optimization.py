"""
Demo 3: Bayesian Optimization for Process Parameter Tuning

This script demonstrates Bayesian Optimization for efficiently finding
optimal AM process parameters with minimal experiments.

Learning Goals:
- Understand the BO algorithm workflow
- See acquisition functions in action
- Compare BO efficiency vs random search
- Visualize convergence to optimum
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from scipy.optimize import minimize
from scipy.stats import norm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)


def load_data(data_path):
    """Load porosity optimization dataset."""
    logger.info("="*70)
    logger.info("STEP 1: LOADING POROSITY DATASET")
    logger.info("="*70)
    
    df = pd.read_csv(data_path)
    logger.info(f"✓ Loaded dataset: {len(df)} samples")
    logger.info(f"✓ Features: laser_power_W, scan_speed_mm_s, hatch_spacing_mm, layer_thickness_um")
    logger.info(f"✓ Target: porosity_pct (minimize this!)")
    
    logger.info(f"\nPorosity statistics:")
    logger.info(f"  Mean: {df['porosity_pct'].mean():.3f}%")
    logger.info(f"  Min: {df['porosity_pct'].min():.3f}%")
    logger.info(f"  Max: {df['porosity_pct'].max():.3f}%")
    logger.info(f"  Target: < 0.5%")
    
    return df


def create_black_box_function(df):
    """
    Create a black-box function for BO demonstration.
    
    In real life, this would be running an actual experiment.
    Here, we use the dataset to simulate expensive experiments.
    """
    logger.info("\n" + "="*70)
    logger.info("STEP 2: CREATING BLACK-BOX OBJECTIVE FUNCTION")
    logger.info("="*70)
    
    # Normalize features
    X = df[['laser_power_W', 'scan_speed_mm_s', 'hatch_spacing_mm', 'layer_thickness_um']].values
    y = df['porosity_pct'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit a GP to the full dataset (this simulates the true unknown function)
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.01)
    true_gp = GaussianProcessRegressor(kernel=kernel, random_state=42, n_restarts_optimizer=5)
    true_gp.fit(X_scaled, y)
    
    logger.info(f"✓ Created surrogate 'true' function from {len(df)} samples")
    logger.info(f"  In reality, this function is unknown - we'd run real experiments")
    logger.info(f"  Each 'experiment' costs time and money!")
    
    # Parameter bounds (original scale)
    bounds = np.array([
        [150, 400],     # laser_power_W
        [600, 1400],    # scan_speed_mm_s
        [0.08, 0.15],   # hatch_spacing_mm
        [25, 50]        # layer_thickness_um
    ])
    
    def objective_function(X_input):
        """Simulate running an experiment at given parameters."""
        X_normalized = scaler.transform(X_input.reshape(1, -1))
        y_pred, y_std = true_gp.predict(X_normalized, return_std=True)
        # Add small noise to simulate experimental variability
        noise = np.random.randn() * 0.05
        return y_pred[0] + noise
    
    return objective_function, bounds, scaler


def expected_improvement(X, X_sample, y_sample, gp, xi=0.01):
    """
    Calculate Expected Improvement acquisition function.
    
    EI(x) = E[max(f(x) - f(x_best), 0)]
    
    Higher EI = more promising point for next experiment
    """
    mu, sigma = gp.predict(X, return_std=True)
    mu_sample = gp.predict(X_sample)
    
    mu = mu.reshape(-1, 1)
    sigma = sigma.reshape(-1, 1)
    
    # Current best observed value
    mu_sample_opt = np.min(mu_sample)  # minimize porosity
    
    with np.errstate(divide='warn'):
        imp = mu_sample_opt - mu - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    
    return ei


def propose_location(acquisition, X_sample, y_sample, gp, bounds):
    """
    Propose next sampling point by optimizing acquisition function.
    """
    dim = bounds.shape[0]
    min_val = 1e10
    min_x = None
    
    # Try multiple random starts to avoid local minima
    n_restarts = 25
    for _ in range(n_restarts):
        # Random starting point within bounds
        x0 = np.random.uniform(bounds[:, 0], bounds[:, 1], size=dim)
        
        # Minimize negative acquisition (maximize acquisition)
        res = minimize(
            lambda x: -acquisition(x.reshape(-1, dim), X_sample, y_sample, gp).flatten(),
            x0=x0,
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        if res.fun < min_val:
            min_val = res.fun
            min_x = res.x
    
    # Ensure result is within bounds (clip if necessary)
    min_x = np.clip(min_x, bounds[:, 0], bounds[:, 1])
    
    return min_x.reshape(1, -1)


def bayesian_optimization(objective_function, bounds, scaler, n_init=10, n_iter=30):
    """
    Run Bayesian Optimization algorithm.
    
    Args:
        objective_function: Function to minimize
        bounds: Parameter bounds in ORIGINAL scale
        scaler: Feature scaler
        n_init: Number of random initial samples
        n_iter: Number of BO iterations
    """
    logger.info("\n" + "="*70)
    logger.info("STEP 3: RUNNING BAYESIAN OPTIMIZATION")
    logger.info("="*70)
    
    dim = bounds.shape[0]
    
    # Phase 1: Random initialization
    logger.info(f"\nPhase 1: Random Initialization ({n_init} experiments)")
    X_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_init, dim))
    y_sample = np.array([objective_function(x) for x in X_sample])
    
    best_idx = np.argmin(y_sample)
    logger.info(f"✓ Initial best porosity: {y_sample[best_idx]:.4f}%")
    logger.info(f"  At parameters: Power={X_sample[best_idx, 0]:.1f}W, "
                f"Speed={X_sample[best_idx, 1]:.1f}mm/s, "
                f"Hatch={X_sample[best_idx, 2]:.3f}mm, "
                f"Layer={X_sample[best_idx, 3]:.1f}μm")
    
    # Track progress - record best-so-far for each random experiment
    best_y_history = [np.min(y_sample[:i+1]) for i in range(n_init)]
    X_history = [X_sample.copy()]
    y_history = [y_sample.copy()]
    
    # Create normalized bounds for acquisition function optimization
    # After scaling, all features should be roughly in [-3, 3] range
    bounds_normalized = np.array([[-3.0, 3.0]] * dim)
    
    # Phase 2: Bayesian Optimization iterations
    logger.info(f"\nPhase 2: Bayesian Optimization ({n_iter} iterations)")
    logger.info(f"{'─'*70}")
    
    for iteration in range(n_iter):
        # Normalize data for GP
        X_scaled = scaler.transform(X_sample)
        
        # Fit GP to current data
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + \
                 WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
        gp.fit(X_scaled, y_sample)
        
        # Find next point using Expected Improvement in NORMALIZED space
        X_next_scaled = propose_location(expected_improvement, X_scaled, y_sample, gp, bounds_normalized)
        
        # Denormalize back to original scale
        X_next_original = scaler.inverse_transform(X_next_scaled)
        
        # Clip to ensure within original bounds (safety check)
        X_next_original = np.clip(X_next_original, bounds[:, 0], bounds[:, 1])
        
        # Run "experiment" at proposed location
        y_next = objective_function(X_next_original[0])
        
        # Update dataset
        X_sample = np.vstack((X_sample, X_next_original))
        y_sample = np.append(y_sample, y_next)
        
        # Track best so far
        current_best = np.min(y_sample)
        best_y_history.append(current_best)
        X_history.append(X_sample.copy())
        y_history.append(y_sample.copy())
        
        # Log every 5 iterations
        if (iteration + 1) % 5 == 0:
            improvement = best_y_history[n_init-1] - current_best
            logger.info(f"Iteration {iteration + 1:2d}: Best = {current_best:.4f}% "
                       f"(improved by {improvement:.4f}%)")
    
    # Final results
    logger.info(f"{'─'*70}")
    best_idx = np.argmin(y_sample)
    best_params = X_sample[best_idx]
    best_value = y_sample[best_idx]
    
    logger.info(f"\n✓ OPTIMIZATION COMPLETE!")
    logger.info(f"  Final best porosity: {best_value:.4f}%")
    logger.info(f"  Improvement from initial: {best_y_history[n_init-1] - best_value:.4f}%")
    logger.info(f"  Total experiments: {n_init + n_iter}")
    
    logger.info(f"\n✓ Optimal Parameters:")
    logger.info(f"  Laser Power: {best_params[0]:.1f} W")
    logger.info(f"  Scan Speed: {best_params[1]:.1f} mm/s")
    logger.info(f"  Hatch Spacing: {best_params[2]:.3f} mm")
    logger.info(f"  Layer Thickness: {best_params[3]:.1f} μm")
    
    if best_value < 0.5:
        logger.info(f"\n🎉 SUCCESS! Achieved target porosity < 0.5%")
    else:
        logger.info(f"\n⚠ Close to target, may need more iterations or better initialization")
    
    return X_sample, y_sample, best_y_history


def random_search_baseline(objective_function, bounds, n_total=40):
    """Run random search as baseline for comparison."""
    logger.info("\n" + "="*70)
    logger.info("STEP 4: RUNNING RANDOM SEARCH BASELINE")
    logger.info("="*70)
    
    dim = bounds.shape[0]
    logger.info(f"Running {n_total} random experiments for comparison...")
    
    X_random = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_total, dim))
    y_random = np.array([objective_function(x) for x in X_random])
    
    # Track best so far at each iteration
    best_random = [np.min(y_random[:i+1]) for i in range(len(y_random))]
    
    logger.info(f"✓ Random search final best: {best_random[-1]:.4f}%")
    logger.info(f"  (This is what you'd get without intelligent optimization)")
    
    return best_random


def create_visualizations(X_bo, y_bo, best_history_bo, best_history_random, output_dir, n_init=10):
    """Create comprehensive visualization of BO results."""
    logger.info("\n" + "="*70)
    logger.info("STEP 5: CREATING VISUALIZATIONS")
    logger.info("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Bayesian Optimization for SLM Parameter Tuning', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Convergence comparison
    ax = axes[0, 0]
    iterations_bo = range(len(best_history_bo))
    iterations_random = range(len(best_history_random))
    
    # Split BO history into random init and BO phases
    best_history_init = best_history_bo[:n_init]
    best_history_bo_phase = best_history_bo[n_init:]
    
    # Plot random initialization phase (same for both)
    ax.plot(range(n_init), best_history_init, 'gray', linewidth=2, 
           marker='o', markersize=6, label='Random Initialization', linestyle='--')
    
    # Plot BO phase
    if len(best_history_bo_phase) > 0:
        ax.plot(range(n_init, len(best_history_bo)), best_history_bo_phase, 'b-o', linewidth=2, 
               markersize=6, label='Bayesian Optimization', markevery=5)
    
    # Plot random search
    ax.plot(iterations_random, best_history_random, 'r--s', linewidth=2, 
           markersize=6, label='Random Search', markevery=5, alpha=0.7)
    
    # Mark transition point
    ax.axvline(x=n_init-0.5, color='orange', linestyle=':', linewidth=2, alpha=0.7,
              label=f'BO starts (after {n_init} random)')
    
    ax.axhline(y=0.5, color='green', linestyle=':', linewidth=2, label='Target: 0.5%')
    
    ax.set_xlabel('Number of Experiments', fontsize=12)
    ax.set_ylabel('Best Porosity Found (%)', fontsize=12)
    ax.set_title('Convergence: BO vs Random Search', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add efficiency gain text
    bo_final = best_history_bo[-1]
    random_final = best_history_random[-1]
    improvement = ((random_final - bo_final) / random_final) * 100
    textstr = f'BO found {improvement:.1f}% better\nsolution than random search!'
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
    ax.text(0.5, 0.95, textstr, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=props, ha='center')
    
    # Plot 2: Experiment values over time
    ax = axes[0, 1]
    ax.scatter(range(len(y_bo)), y_bo, c=range(len(y_bo)), cmap='viridis', 
              s=100, edgecolors='black', linewidth=1)
    ax.plot(range(len(y_bo)), y_bo, 'k-', alpha=0.3, linewidth=1)
    
    # Mark initial random phase
    ax.axvline(x=n_init-0.5, color='red', linestyle='--', linewidth=2, 
              label=f'BO starts (after n={n_init} random)', alpha=0.7)
    
    ax.axhline(y=0.5, color='green', linestyle=':', linewidth=2, label='Target')
    
    ax.set_xlabel('Experiment Number', fontsize=12)
    ax.set_ylabel('Porosity (%)', fontsize=12)
    ax.set_title('All Experiment Results (colored by iteration)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Parameter exploration (2D projection)
    ax = axes[1, 0]
    
    # Plot laser power vs scan speed
    scatter = ax.scatter(X_bo[:, 0], X_bo[:, 1], c=y_bo, cmap='RdYlGn_r', 
                        s=100, edgecolors='black', linewidth=1)
    
    # Mark best point
    best_idx = np.argmin(y_bo)
    ax.scatter(X_bo[best_idx, 0], X_bo[best_idx, 1], 
              marker='*', s=500, color='gold', edgecolors='black', 
              linewidth=2, label='Best Found', zorder=10)
    
    # Mark initial points
    ax.scatter(X_bo[:10, 0], X_bo[:10, 1], 
              marker='x', s=100, color='red', linewidth=2, 
              label='Initial Random', zorder=5)
    
    ax.set_xlabel('Laser Power (W)', fontsize=12)
    ax.set_ylabel('Scan Speed (mm/s)', fontsize=12)
    ax.set_title('Parameter Space Exploration (Power vs Speed)', fontsize=13, fontweight='bold')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Porosity (%)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Improvement per iteration
    ax = axes[1, 1]
    
    improvements = np.abs(np.diff(best_history_bo))
    iterations = range(1, len(improvements) + 1)
    
    ax.bar(iterations, improvements, color='steelblue', edgecolor='black', linewidth=1)
    ax.axhline(y=0.01, color='red', linestyle='--', linewidth=2, 
              label='Significant Improvement (0.01%)', alpha=0.7)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Improvement in Best Value (%)', fontsize=12)
    ax.set_title('Improvement per BO Iteration', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add convergence text
    last_improvement = improvements[-5:].mean()
    textstr = f'Last 5 iterations avg\nimprovement: {last_improvement:.4f}%\n→ Converging!'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', ha='right', bbox=props)
    
    plt.tight_layout()
    viz_path = output_dir / '03_bayesian_optimization.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved visualization: {viz_path}")
    plt.close()


def main():
    """Main execution function."""
    logger.info("\n" + "="*70)
    logger.info("BAYESIAN OPTIMIZATION DEMO")
    logger.info("="*70)
    logger.info("Efficiently finding optimal SLM parameters to minimize porosity")
    logger.info("="*70 + "\n")
    
    # Setup paths
    script_dir = Path(__file__).parent
    data_path = script_dir / 'data' / 'slm_porosity_optimization.csv'
    output_dir = script_dir / 'outputs' / '03_bayesian_optimization'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Execute pipeline
    n_init = 10
    n_iter = 30
    df = load_data(data_path)
    objective_function, bounds, scaler = create_black_box_function(df)
    X_bo, y_bo, best_history_bo = bayesian_optimization(objective_function, bounds, scaler, n_init=n_init, n_iter=n_iter)
    best_history_random = random_search_baseline(objective_function, bounds, n_total=n_init+n_iter)
    create_visualizations(X_bo, y_bo, best_history_bo, best_history_random, output_dir, n_init=n_init)
    
    logger.info(f"\n✓ All outputs saved to: {output_dir.absolute()}")
    logger.info("\n" + "="*70)
    logger.info("KEY TAKEAWAYS")
    logger.info("="*70)
    logger.info("1. BO finds better solution faster than random search")
    logger.info("2. Intelligently balances exploration vs exploitation")
    logger.info("3. Uses GP uncertainty to guide next experiments")
    logger.info("4. Converges to optimum in ~40 experiments vs hundreds with DOE")
    logger.info("5. Critical for expensive AM experiments!")
    logger.info("\n" + "="*70)
    logger.info("DEMO COMPLETE")
    logger.info("="*70 + "\n")


if __name__ == '__main__':
    main()
