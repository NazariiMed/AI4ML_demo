"""
Demo 1: Multi-Objective Bayesian Optimization (MOBO) - Improved 3D Visualization

Enhanced version with clearer 3D Pareto front visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.spatial import ConvexHull
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (18, 12)

# Set seed for reproducibility
np.random.seed(42)

def create_objective_functions():
    """Create three conflicting objectives."""
    bounds = np.array([[150, 400], [600, 1400], [0.08, 0.15], [25, 50]])
    
    def multi_objective_function(X):
        X = np.atleast_2d(X)
        results = []
        for x in X:
            power, speed, hatch, layer = x
            
            # Objective 1: Porosity (MINIMIZE)
            porosity = 0.3 + 0.4 * (
                ((power - 280) / 100)**2 + 
                ((speed - 1000) / 200)**2 + 
                ((hatch - 0.10) / 0.02)**2 + 
                ((layer - 32) / 8)**2
            )
            porosity += np.random.randn() * 0.03
            porosity = np.clip(porosity, 0.1, 3.0)
            
            # Objective 2: Build Rate (MAXIMIZE → minimize negative)
            volume_rate = speed * hatch * (layer / 1000) * 3600
            power_eff = max(0.5, 1.0 - 0.3 * ((power - 300) / 150)**2)
            build_rate = volume_rate * power_eff
            build_rate += np.random.randn() * 200
            build_rate = np.clip(build_rate, 2000, 35000)
            
            # Objective 3: Energy (MINIMIZE)
            volume_rate_cm3 = speed * hatch * layer / 1000
            energy = power / max(volume_rate_cm3, 0.01) + 20
            energy += np.random.randn() * 2
            energy = np.clip(energy, 10, 150)
            
            results.append([porosity, -build_rate, energy])  # Negate build_rate to minimize
        
        return np.array(results)
    
    return multi_objective_function, bounds


def is_pareto_efficient(costs):
    """Find Pareto efficient points."""
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient


def compute_hypervolume_2d(pareto_front, reference_point):
    """Compute 2D hypervolume (first 2 objectives only)."""
    if len(pareto_front) == 0:
        return 0.0
    
    front_2d = pareto_front[:, :2]
    ref_2d = reference_point[:2]
    
    sorted_indices = np.argsort(front_2d[:, 0])
    sorted_front = front_2d[sorted_indices]
    
    hv = 0.0
    prev_x = ref_2d[0]
    
    for point in sorted_front:
        width = prev_x - point[0]
        height = ref_2d[1] - point[1]
        if width > 0 and height > 0:
            hv += width * height
        prev_x = point[0]
    
    return max(0, hv)


def expected_improvement(x, gp, y_best):
    """Standard EI acquisition for single GP."""
    x = np.atleast_2d(x)
    mu, sigma = gp.predict(x, return_std=True)
    
    if sigma > 1e-6:
        Z = (y_best - mu) / sigma
        ei = (y_best - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)
    else:
        ei = 0.0
    
    return ei[0] if isinstance(ei, np.ndarray) else ei


def propose_next_point_scalarized(X_sample, Y_sample, bounds):
    """Propose next point using random scalarization."""
    n_obj = Y_sample.shape[1]
    
    # Random weights (encourages diversity)
    weights = np.random.dirichlet(np.ones(n_obj))
    
    # Fit GPs for each objective
    gps = []
    for j in range(n_obj):
        kernel = ConstantKernel(1.0) * RBF(1.0)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, random_state=42)
        gp.fit(X_sample, Y_sample[:, j])
        gps.append(gp)
    
    # Weighted scalarization GP
    def weighted_ei(x):
        x = np.atleast_2d(x)
        eis = []
        weighted_Y = np.dot(Y_sample, weights)
        y_best_weighted = np.min(weighted_Y)
        
        # Predict for each objective
        mus, sigmas = [], []
        for gp in gps:
            mu, sigma = gp.predict(x, return_std=True)
            mus.append(mu[0])
            sigmas.append(sigma[0])
        
        # Weighted mean and std
        mu_w = np.dot(weights, mus)
        sigma_w = np.sqrt(np.dot(weights**2, np.array(sigmas)**2))
        
        # EI for weighted objective
        if sigma_w > 1e-6:
            Z = (y_best_weighted - mu_w) / sigma_w
            ei = (y_best_weighted - mu_w) * norm.cdf(Z) + sigma_w * norm.pdf(Z)
        else:
            ei = 0.0
        
        return -ei  # Negative for minimization
    
    # Optimize
    best_x = None
    best_val = np.inf
    
    for _ in range(20):
        x0 = np.random.uniform(bounds[:, 0], bounds[:, 1])
        result = minimize(weighted_ei, x0, method='L-BFGS-B', bounds=bounds)
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x
    
    return best_x.reshape(1, -1)


def multi_objective_bo(obj_function, bounds, n_init=15, n_iter=25):
    """Run multi-objective BO with random scalarization."""
    logger.info("\n" + "="*70)
    logger.info("MULTI-OBJECTIVE BAYESIAN OPTIMIZATION")
    logger.info("="*70)
    
    dim = bounds.shape[0]
    
    # Phase 1: Random initialization
    logger.info(f"\nPhase 1: Random initialization ({n_init} samples)")
    X_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_init, dim))
    Y_sample = obj_function(X_sample)
    
    pareto_mask = is_pareto_efficient(Y_sample)
    logger.info(f"✓ Initial Pareto front: {np.sum(pareto_mask)} solutions")
    logger.info(f"  Porosity: [{Y_sample[:, 0].min():.3f}, {Y_sample[:, 0].max():.3f}]%")
    logger.info(f"  Build rate: [{-Y_sample[:, 1].max():.0f}, {-Y_sample[:, 1].min():.0f}] mm³/hr")
    logger.info(f"  Energy: [{Y_sample[:, 2].min():.1f}, {Y_sample[:, 2].max():.1f}] kJ/cm³")
    
    # Reference point
    reference_point = np.max(Y_sample, axis=0) * 1.2
    
    # History
    X_history = [X_sample.copy()]
    Y_history = [Y_sample.copy()]
    pareto_history = [pareto_mask.copy()]
    hv_history = [compute_hypervolume_2d(Y_sample[pareto_mask], reference_point)]
    
    # Phase 2: BO iterations
    logger.info(f"\nPhase 2: BO iterations ({n_iter} iterations)")
    logger.info("─"*70)
    
    for iteration in range(n_iter):
        # Propose next point
        X_next = propose_next_point_scalarized(X_sample, Y_sample, bounds)
        Y_next = obj_function(X_next)
        
        # Update
        X_sample = np.vstack([X_sample, X_next])
        Y_sample = np.vstack([Y_sample, Y_next])
        
        # Update Pareto front
        pareto_mask = is_pareto_efficient(Y_sample)
        n_pareto = np.sum(pareto_mask)
        
        # Compute hypervolume
        hv = compute_hypervolume_2d(Y_sample[pareto_mask], reference_point)
        
        # Store
        X_history.append(X_sample.copy())
        Y_history.append(Y_sample.copy())
        pareto_history.append(pareto_mask.copy())
        hv_history.append(hv)
        
        if (iteration + 1) % 5 == 0:
            logger.info(f"Iter {iteration + 1:2d}: Pareto size = {n_pareto}, HV = {hv:.1f}")
    
    logger.info("─"*70)
    logger.info(f"✓ Final Pareto front: {np.sum(pareto_mask)} solutions")
    logger.info(f"  Hypervolume improvement: {hv_history[-1] - hv_history[0]:.1f}")
    
    return X_sample, Y_sample, pareto_mask, Y_history, pareto_history, hv_history


def create_visualizations(X_sample, Y_sample, pareto_mask, Y_history, pareto_history, hv_history, output_dir):
    """Create visualizations with improved 3D plot."""
    logger.info("\n" + "="*70)
    logger.info("CREATING VISUALIZATIONS")
    logger.info("="*70)
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    fig.suptitle('Multi-Objective Bayesian Optimization Results', fontsize=18, fontweight='bold', y=0.98)
    
    pareto_front = Y_sample[pareto_mask]
    
    # Plot 1: Porosity vs Build Rate
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(Y_sample[:, 0], -Y_sample[:, 1], c='lightgray', s=50, alpha=0.5, label='Non-Pareto', edgecolors='black')
    ax1.scatter(pareto_front[:, 0], -pareto_front[:, 1], c='red', s=150, marker='*', label='Pareto Front', edgecolors='darkred', linewidth=1.5, zorder=10)
    sorted_idx = np.argsort(pareto_front[:, 0])
    ax1.plot(pareto_front[sorted_idx, 0], -pareto_front[sorted_idx, 1], 'r--', linewidth=2, alpha=0.7, zorder=5)
    ax1.set_xlabel('Porosity (%)', fontsize=12)
    ax1.set_ylabel('Build Rate (mm³/hr)', fontsize=12)
    ax1.set_title('Pareto Frontier: Quality vs Speed', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Porosity vs Energy
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(Y_sample[:, 0], Y_sample[:, 2], c='lightgray', s=50, alpha=0.5, edgecolors='black')
    ax2.scatter(pareto_front[:, 0], pareto_front[:, 2], c='red', s=150, marker='*', edgecolors='darkred', linewidth=1.5, zorder=10)
    sorted_idx = np.argsort(pareto_front[:, 0])
    ax2.plot(pareto_front[sorted_idx, 0], pareto_front[sorted_idx, 2], 'r--', linewidth=2, alpha=0.7, zorder=5)
    ax2.set_xlabel('Porosity (%)', fontsize=12)
    ax2.set_ylabel('Energy (kJ/cm³)', fontsize=12)
    ax2.set_title('Pareto Frontier: Quality vs Cost', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Build Rate vs Energy
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(-Y_sample[:, 1], Y_sample[:, 2], c='lightgray', s=50, alpha=0.5, edgecolors='black')
    ax3.scatter(-pareto_front[:, 1], pareto_front[:, 2], c='red', s=150, marker='*', edgecolors='darkred', linewidth=1.5, zorder=10)
    sorted_idx = np.argsort(-pareto_front[:, 1])
    ax3.plot(-pareto_front[sorted_idx, 1], pareto_front[sorted_idx, 2], 'r--', linewidth=2, alpha=0.7, zorder=5)
    ax3.set_xlabel('Build Rate (mm³/hr)', fontsize=12)
    ax3.set_ylabel('Energy (kJ/cm³)', fontsize=12)
    ax3.set_title('Pareto Frontier: Speed vs Cost', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: IMPROVED 3D Pareto front with surface and annotations
    ax4 = fig.add_subplot(gs[1, :2], projection='3d')
    
    # Plot non-Pareto points (very transparent)
    ax4.scatter(Y_sample[:, 0], -Y_sample[:, 1], Y_sample[:, 2], 
                c='lightgray', s=20, alpha=0.15, edgecolors='none', label='All evaluations')
    
    # Plot Pareto front points (large, prominent)
    scatter = ax4.scatter(pareto_front[:, 0], -pareto_front[:, 1], pareto_front[:, 2], 
                c=pareto_front[:, 2], cmap='hot_r', s=250, marker='*', 
                edgecolors='darkred', linewidth=2, label='Pareto optimal', zorder=100)
    
    # Add colorbar for energy values
    cbar = plt.colorbar(scatter, ax=ax4, pad=0.1, shrink=0.6)
    cbar.set_label('Energy (kJ/cm³)', fontsize=10)
    
    # Try to create a surface through Pareto points
    if len(pareto_front) >= 4:
        try:
            # Create convex hull of Pareto points
            points_3d = np.column_stack([pareto_front[:, 0], 
                                        -pareto_front[:, 1], 
                                        pareto_front[:, 2]])
            hull = ConvexHull(points_3d)
            
            # Plot the triangulated surface with transparency
            for simplex in hull.simplices:
                triangle = points_3d[simplex]
                ax4.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2],
                               color='red', alpha=0.1, edgecolor='red', linewidth=0.5)
        except Exception as e:
            logger.info(f"  Note: Could not create 3D surface (needs 4+ non-coplanar points)")
    
    # Add vertical projection lines for key Pareto points
    z_min = ax4.get_zlim()[0]
    for i in range(min(5, len(pareto_front))):  # Show lines for up to 5 points
        x, y, z = pareto_front[i, 0], -pareto_front[i, 1], pareto_front[i, 2]
        ax4.plot([x, x], [y, y], [z_min, z], 'gray', linestyle=':', alpha=0.3, linewidth=1)
    
    # Annotate extreme points
    best_quality_idx = np.argmin(pareto_front[:, 0])
    best_speed_idx = np.argmax(-pareto_front[:, 1])
    best_energy_idx = np.argmin(pareto_front[:, 2])
    
    extreme_indices = [best_quality_idx, best_speed_idx, best_energy_idx]
    labels = ['Best Quality', 'Best Speed', 'Best Energy']
    
    for idx, label in zip(extreme_indices, labels):
        x, y, z = pareto_front[idx, 0], -pareto_front[idx, 1], pareto_front[idx, 2]
        ax4.text(x, y, z, f'  {label}', fontsize=8, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax4.set_xlabel('\nPorosity (%)', fontsize=11, labelpad=10)
    ax4.set_ylabel('\nBuild Rate (mm³/hr)', fontsize=11, labelpad=10)
    ax4.set_zlabel('\nEnergy (kJ/cm³)', fontsize=11, labelpad=10)
    ax4.set_title('3D Pareto Frontier\n(colored by energy, stars show optimal trade-offs)', 
                  fontsize=13, fontweight='bold', pad=20)
    ax4.legend(loc='upper left', fontsize=9)
    ax4.view_init(elev=25, azim=45)  # Good viewing angle
    ax4.grid(True, alpha=0.2)
    
    # Make the 3D plot background lighter
    ax4.xaxis.pane.fill = True
    ax4.yaxis.pane.fill = True
    ax4.zaxis.pane.fill = True
    ax4.xaxis.pane.set_alpha(0.1)
    ax4.yaxis.pane.set_alpha(0.1)
    ax4.zaxis.pane.set_alpha(0.1)
    
    # Plot 5: Pareto front size evolution
    ax5 = fig.add_subplot(gs[1, 2])
    pareto_sizes = [np.sum(mask) for mask in pareto_history]
    ax5.plot(range(len(pareto_sizes)), pareto_sizes, 'g-o', linewidth=2, markersize=6, markevery=3)
    ax5.axhline(y=pareto_sizes[0], color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Initial')
    ax5.set_xlabel('Iteration', fontsize=12)
    ax5.set_ylabel('Number of Pareto Solutions', fontsize=12)
    ax5.set_title('Pareto Front Growth', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    improvement = pareto_sizes[-1] - pareto_sizes[0]
    textstr = f'Growth: +{improvement} solutions'
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
    ax5.text(0.5, 0.95, textstr, transform=ax5.transAxes, fontsize=11, verticalalignment='top', ha='center', bbox=props)
    
    # Plot 6: Best values per objective
    ax6 = fig.add_subplot(gs[2, 0])
    best_obj1 = [np.min(Y[:, 0]) for Y in Y_history]
    best_obj2 = [np.max(-Y[:, 1]) for Y in Y_history]
    best_obj3 = [np.min(Y[:, 2]) for Y in Y_history]
    
    ax6_twin1 = ax6.twinx()
    ax6_twin2 = ax6.twinx()
    ax6_twin2.spines['right'].set_position(('outward', 60))
    
    p1, = ax6.plot(range(len(best_obj1)), best_obj1, 'r-o', linewidth=2, markersize=4, markevery=3, label='Porosity')
    p2, = ax6_twin1.plot(range(len(best_obj2)), best_obj2, 'b-s', linewidth=2, markersize=4, markevery=3, label='Build Rate')
    p3, = ax6_twin2.plot(range(len(best_obj3)), best_obj3, 'g-^', linewidth=2, markersize=4, markevery=3, label='Energy')
    
    ax6.set_xlabel('Iteration', fontsize=12)
    ax6.set_ylabel('Porosity (%)', fontsize=11, color='r')
    ax6_twin1.set_ylabel('Build Rate (mm³/hr)', fontsize=11, color='b')
    ax6_twin2.set_ylabel('Energy (kJ/cm³)', fontsize=11, color='g')
    
    ax6.tick_params(axis='y', labelcolor='r')
    ax6_twin1.tick_params(axis='y', labelcolor='b')
    ax6_twin2.tick_params(axis='y', labelcolor='g')
    
    ax6.set_title('Best Values per Objective', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Parallel coordinates
    ax7 = fig.add_subplot(gs[2, 1])
    pareto_normalized = np.zeros_like(pareto_front)
    for j in range(3):
        min_val, max_val = Y_sample[:, j].min(), Y_sample[:, j].max()
        pareto_normalized[:, j] = (pareto_front[:, j] - min_val) / (max_val - min_val + 1e-10)
    
    pareto_normalized[:, 1] = 1 - pareto_normalized[:, 1]  # Flip build rate
    
    for i in range(len(pareto_normalized)):
        ax7.plot([0, 1, 2], pareto_normalized[i], 'o-', alpha=0.6, linewidth=1.5)
    
    ax7.set_xticks([0, 1, 2])
    ax7.set_xticklabels(['Porosity\n(minimize)', 'Build Rate\n(maximize)', 'Energy\n(minimize)'], fontsize=10)
    ax7.set_ylabel('Normalized Value', fontsize=12)
    ax7.set_ylim(-0.1, 1.1)
    ax7.set_title('Parallel Coordinates', fontsize=13, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Plot 8: Trade-off analysis
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    # Calculate statistics
    n_total = len(Y_sample)
    n_pareto = len(pareto_front)
    pareto_pct = 100 * n_pareto / n_total
    
    best_quality_idx = np.argmin(pareto_front[:, 0])
    best_speed_idx = np.argmax(-pareto_front[:, 1])
    best_energy_idx = np.argmin(pareto_front[:, 2])
    
    summary_text = f"""OPTIMIZATION SUMMARY

Total evaluations: {n_total}
Pareto solutions: {n_pareto} ({pareto_pct:.1f}%)

Best Quality Solution:
  Porosity: {pareto_front[best_quality_idx, 0]:.3f}%
  Build rate: {-pareto_front[best_quality_idx, 1]:.0f} mm³/hr
  Energy: {pareto_front[best_quality_idx, 2]:.1f} kJ/cm³

Best Speed Solution:
  Porosity: {pareto_front[best_speed_idx, 0]:.3f}%
  Build rate: {-pareto_front[best_speed_idx, 1]:.0f} mm³/hr
  Energy: {pareto_front[best_speed_idx, 2]:.1f} kJ/cm³

Best Energy Solution:
  Porosity: {pareto_front[best_energy_idx, 0]:.3f}%
  Build rate: {-pareto_front[best_energy_idx, 1]:.0f} mm³/hr
  Energy: {pareto_front[best_energy_idx, 2]:.1f} kJ/cm³

Key Insight:
Random scalarization found {n_pareto} 
diverse trade-off solutions!
    """
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=9,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    viz_path = output_dir / '01_multi_objective_optimization.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved: {viz_path}")
    plt.close()


def main():
    logger.info("\n" + "="*70)
    logger.info("MULTI-OBJECTIVE OPTIMIZATION DEMO (IMPROVED 3D)")
    logger.info("="*70)
    
    script_dir = Path(__file__).parent
    output_dir = script_dir / 'outputs' / '01_multi_objective_optimization'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    obj_function, bounds = create_objective_functions()
    X_sample, Y_sample, pareto_mask, Y_history, pareto_history, hv_history = \
        multi_objective_bo(obj_function, bounds, n_init=5, n_iter=35)
    
    create_visualizations(X_sample, Y_sample, pareto_mask, Y_history, pareto_history, hv_history, output_dir)
    
    logger.info(f"\n✓ All outputs saved to: {output_dir.absolute()}")
    logger.info("\n" + "="*70)
    logger.info("DEMO COMPLETE")
    logger.info("="*70 + "\n")


if __name__ == '__main__':
    main()
