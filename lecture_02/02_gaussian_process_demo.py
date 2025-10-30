"""
Demo 2: Gaussian Process Regression with Uncertainty Quantification

This script demonstrates Gaussian Process Regression and why uncertainty
quantification is crucial for optimization in additive manufacturing.

Learning Goals:
- Understand GP predictions with confidence intervals
- Compare GP with standard regression (no uncertainty)
- Visualize uncertainty in sparse data regions
- See how GP guides experimental design
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
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
plt.rcParams['figure.figsize'] = (14, 10)


def load_data(data_path):
    """Load adhesion strength dataset."""
    logger.info("="*70)
    logger.info("STEP 1: LOADING DATA")
    logger.info("="*70)
    
    df = pd.read_csv(data_path)
    logger.info(f"✓ Loaded dataset: {len(df)} samples")
    logger.info(f"  This is a SMALL dataset - perfect for demonstrating GP!")
    logger.info(f"✓ Features: print_speed_mm_s, nozzle_temp_C")
    logger.info(f"✓ Target: adhesion_strength_MPa")
    
    logger.info(f"\nDataset characteristics:")
    logger.info(f"  Print speed range: {df['print_speed_mm_s'].min():.1f} - {df['print_speed_mm_s'].max():.1f} mm/s")
    logger.info(f"  Temperature range: {df['nozzle_temp_C'].min():.1f} - {df['nozzle_temp_C'].max():.1f} °C")
    logger.info(f"  Adhesion strength: {df['adhesion_strength_MPa'].min():.2f} - {df['adhesion_strength_MPa'].max():.2f} MPa")
    
    return df


def prepare_1d_demo(df):
    """Prepare data for 1D visualization (speed vs adhesion at fixed temp)."""
    logger.info("\n" + "="*70)
    logger.info("STEP 2: PREPARING 1D DEMONSTRATION")
    logger.info("="*70)
    
    # For visualization, we'll focus on print_speed only
    # Select samples near median temperature
    median_temp = df['nozzle_temp_C'].median()
    temp_tolerance = 10  # °C
    
    df_subset = df[np.abs(df['nozzle_temp_C'] - median_temp) < temp_tolerance].copy()
    
    logger.info(f"✓ Selected {len(df_subset)} samples near {median_temp:.0f}°C (±{temp_tolerance}°C)")
    logger.info(f"  This gives us clean 1D relationship for visualization")
    
    X = df_subset[['print_speed_mm_s']].values
    y = df_subset['adhesion_strength_MPa'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    logger.info(f"✓ Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test, X


def train_gp_models(X_train, y_train):
    """Train Gaussian Process with different kernels."""
    logger.info("\n" + "="*70)
    logger.info("STEP 3: TRAINING GAUSSIAN PROCESS MODELS")
    logger.info("="*70)
    
    models = {}
    
    # Model 1: RBF Kernel (most common)
    logger.info(f"\nModel 1: GP with RBF (Squared Exponential) Kernel")
    kernel_rbf = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=10.0, length_scale_bounds=(1, 100)) + \
                 WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
    
    gp_rbf = GaussianProcessRegressor(
        kernel=kernel_rbf,
        n_restarts_optimizer=10,
        random_state=42
    )
    gp_rbf.fit(X_train, y_train)
    
    logger.info(f"✓ Optimized kernel: {gp_rbf.kernel_}")
    logger.info(f"  Length scale: {gp_rbf.kernel_.k1.k2.length_scale:.2f} (controls smoothness)")
    logger.info(f"  Noise level: {gp_rbf.kernel_.k2.noise_level:.4f} (measurement uncertainty)")
    
    models['GP-RBF'] = gp_rbf
    
    # Model 2: Matérn Kernel (more flexible)
    logger.info(f"\nModel 2: GP with Matérn Kernel (nu=2.5)")
    kernel_matern = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=10.0, length_scale_bounds=(1, 100), nu=2.5) + \
                    WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
    
    gp_matern = GaussianProcessRegressor(
        kernel=kernel_matern,
        n_restarts_optimizer=10,
        random_state=42
    )
    gp_matern.fit(X_train, y_train)
    
    logger.info(f"✓ Optimized kernel: {gp_matern.kernel_}")
    logger.info(f"  Matérn allows less smooth functions than RBF")
    
    models['GP-Matérn'] = gp_matern
    
    # For comparison: Random Forest (no uncertainty)
    logger.info(f"\nComparison Model: Random Forest (no uncertainty)")
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    logger.info(f"✓ Trained Random Forest baseline")
    models['Random Forest'] = rf
    
    return models


def evaluate_models(models, X_train, X_test, y_train, y_test):
    """Evaluate models on test set."""
    logger.info("\n" + "="*70)
    logger.info("STEP 4: EVALUATING MODELS")
    logger.info("="*70)
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        logger.info(f"\n{name}:")
        logger.info(f"  Test R²: {r2:.4f}")
        logger.info(f"  Test MAE: {mae:.4f} MPa")
        
        if 'GP' in name:
            # GP can predict with uncertainty
            y_pred, y_std = model.predict(X_test, return_std=True)
            logger.info(f"  ✓ Provides uncertainty: mean std = {y_std.mean():.4f} MPa")
            logger.info(f"  ✓ Uncertainty range: [{y_std.min():.4f}, {y_std.max():.4f}] MPa")
        else:
            logger.info(f"  ✗ No uncertainty quantification available")


def create_visualizations(models, X_train, y_train, X_test, y_test, X_all, output_dir):
    """Create comprehensive visualizations."""
    logger.info("\n" + "="*70)
    logger.info("STEP 5: CREATING VISUALIZATIONS")
    logger.info("="*70)
    
    # Create dense grid for smooth predictions
    X_grid = np.linspace(X_all.min(), X_all.max(), 200).reshape(-1, 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Gaussian Process Regression: Uncertainty Quantification', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: GP-RBF with confidence intervals
    ax = axes[0, 0]
    gp_rbf = models['GP-RBF']
    y_pred, y_std = gp_rbf.predict(X_grid, return_std=True)
    
    # Plot confidence intervals
    ax.fill_between(X_grid.ravel(), y_pred - 1.96*y_std, y_pred + 1.96*y_std,
                    alpha=0.3, color='blue', label='95% Confidence Interval')
    ax.fill_between(X_grid.ravel(), y_pred - y_std, y_pred + y_std,
                    alpha=0.5, color='blue', label='68% Confidence Interval')
    
    # Plot predictions
    ax.plot(X_grid, y_pred, 'b-', linewidth=2, label='GP Mean Prediction')
    
    # Plot training data
    ax.scatter(X_train, y_train, c='red', s=100, marker='o', 
              edgecolors='black', linewidth=1.5, label='Training Data', zorder=10)
    
    # Plot test data
    ax.scatter(X_test, y_test, c='green', s=100, marker='s',
              edgecolors='black', linewidth=1.5, label='Test Data', zorder=10)
    
    ax.set_xlabel('Print Speed (mm/s)', fontsize=12)
    ax.set_ylabel('Adhesion Strength (MPa)', fontsize=12)
    ax.set_title('GP with RBF Kernel - Uncertainty Visualization', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: GP-Matérn
    ax = axes[0, 1]
    gp_matern = models['GP-Matérn']
    y_pred, y_std = gp_matern.predict(X_grid, return_std=True)
    
    ax.fill_between(X_grid.ravel(), y_pred - 1.96*y_std, y_pred + 1.96*y_std,
                    alpha=0.3, color='purple')
    ax.plot(X_grid, y_pred, 'purple', linewidth=2, label='GP Mean Prediction')
    ax.scatter(X_train, y_train, c='red', s=100, marker='o', 
              edgecolors='black', linewidth=1.5, label='Training Data', zorder=10)
    ax.scatter(X_test, y_test, c='green', s=100, marker='s',
              edgecolors='black', linewidth=1.5, label='Test Data', zorder=10)
    
    ax.set_xlabel('Print Speed (mm/s)', fontsize=12)
    ax.set_ylabel('Adhesion Strength (MPa)', fontsize=12)
    ax.set_title('GP with Matérn Kernel', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Random Forest (no uncertainty)
    ax = axes[1, 0]
    rf = models['Random Forest']
    y_pred = rf.predict(X_grid)
    
    ax.plot(X_grid, y_pred, 'orange', linewidth=2, label='RF Prediction')
    ax.scatter(X_train, y_train, c='red', s=100, marker='o',
              edgecolors='black', linewidth=1.5, label='Training Data', zorder=10)
    ax.scatter(X_test, y_test, c='green', s=100, marker='s',
              edgecolors='black', linewidth=1.5, label='Test Data', zorder=10)
    
    ax.set_xlabel('Print Speed (mm/s)', fontsize=12)
    ax.set_ylabel('Adhesion Strength (MPa)', fontsize=12)
    ax.set_title('Random Forest - NO Uncertainty', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add text box explaining the problem
    textstr = 'Problem: RF gives predictions\nbut no confidence intervals!\nHow do we know where to\nexperiment next?'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props)
    
    # Plot 4: Uncertainty map
    ax = axes[1, 1]
    gp_rbf = models['GP-RBF']
    y_pred, y_std = gp_rbf.predict(X_grid, return_std=True)
    
    # Plot uncertainty magnitude
    ax.plot(X_grid, y_std, 'r-', linewidth=2, label='Prediction Uncertainty (σ)')
    ax.fill_between(X_grid.ravel(), 0, y_std, alpha=0.3, color='red')
    
    # Mark training data locations
    for x in X_train:
        ax.axvline(x=x, color='blue', linestyle='--', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('Print Speed (mm/s)', fontsize=12)
    ax.set_ylabel('Standard Deviation (MPa)', fontsize=12)
    ax.set_title('Where is GP Most Uncertain?', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add text explaining strategy
    textstr = 'High uncertainty = sparse data\n→ Good candidates for next\n   experiments in optimization!'
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    viz_path = output_dir / '02_gp_demonstration.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved visualization: {viz_path}")
    plt.close()


def demonstrate_acquisition_strategy(models, X_train, y_train, output_dir):
    """Show how GP uncertainty guides next experiment selection."""
    logger.info("\n" + "="*70)
    logger.info("STEP 6: ACQUISITION STRATEGY DEMONSTRATION")
    logger.info("="*70)
    
    logger.info("\nQuestion: Where should we run our next experiment?")
    
    X_grid = np.linspace(X_train.min() - 5, X_train.max() + 5, 200).reshape(-1, 1)
    gp = models['GP-RBF']
    y_pred, y_std = gp.predict(X_grid, return_std=True)
    
    # Simple acquisition functions
    # 1. Highest uncertainty (exploration)
    max_uncertainty_idx = np.argmax(y_std)
    max_uncertainty_x = X_grid[max_uncertainty_idx]
    
    # 2. Best predicted value (exploitation)
    best_pred_idx = np.argmax(y_pred)
    best_pred_x = X_grid[best_pred_idx]
    
    # 3. Upper Confidence Bound (UCB) - balance both
    kappa = 2.0  # exploration parameter
    ucb = y_pred + kappa * y_std
    ucb_idx = np.argmax(ucb)
    ucb_x = X_grid[ucb_idx]
    
    logger.info(f"\nStrategy 1 - Maximum Uncertainty (Pure Exploration):")
    logger.info(f"  Suggested: {max_uncertainty_x[0]:.1f} mm/s")
    logger.info(f"  Uncertainty: {y_std[max_uncertainty_idx]:.3f} MPa")
    logger.info(f"  Rationale: Most uncertain → learn most from this experiment")
    
    logger.info(f"\nStrategy 2 - Best Predicted Value (Pure Exploitation):")
    logger.info(f"  Suggested: {best_pred_x[0]:.1f} mm/s")
    logger.info(f"  Predicted: {y_pred[best_pred_idx]:.3f} MPa")
    logger.info(f"  Rationale: Highest predicted performance → refine optimum")
    
    logger.info(f"\nStrategy 3 - Upper Confidence Bound (Balanced):")
    logger.info(f"  Suggested: {ucb_x[0]:.1f} mm/s")
    logger.info(f"  UCB Value: {ucb[ucb_idx]:.3f} MPa")
    logger.info(f"  Rationale: Balance exploration and exploitation")
    
    logger.info(f"\n✓ In Bayesian Optimization, we typically use UCB or Expected Improvement")
    logger.info(f"  These will be demonstrated in Demo 3!")


def main():
    """Main execution function."""
    logger.info("\n" + "="*70)
    logger.info("GAUSSIAN PROCESS REGRESSION DEMO")
    logger.info("="*70)
    logger.info("Demonstrating uncertainty quantification with GP")
    logger.info("="*70 + "\n")
    
    # Setup paths
    script_dir = Path(__file__).parent
    data_path = script_dir / 'data' / 'print_adhesion_strength.csv'
    output_dir = script_dir / 'outputs' / '02_gaussian_process'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Execute pipeline
    df = load_data(data_path)
    X_train, X_test, y_train, y_test, X_all = prepare_1d_demo(df)
    models = train_gp_models(X_train, y_train)
    evaluate_models(models, X_train, X_test, y_train, y_test)
    create_visualizations(models, X_train, y_train, X_test, y_test, X_all, output_dir)
    demonstrate_acquisition_strategy(models, X_train, y_train, output_dir)
    
    logger.info(f"\n✓ All outputs saved to: {output_dir.absolute()}")
    logger.info("\n" + "="*70)
    logger.info("KEY TAKEAWAYS")
    logger.info("="*70)
    logger.info("1. GP provides both predictions AND uncertainty estimates")
    logger.info("2. Uncertainty is high where data is sparse → guides exploration")
    logger.info("3. Different kernels (RBF, Matérn) encode different assumptions")
    logger.info("4. This uncertainty is CRITICAL for Bayesian Optimization!")
    logger.info("5. Random Forest and other models lack this capability")
    logger.info("\n" + "="*70)
    logger.info("DEMO COMPLETE")
    logger.info("="*70 + "\n")


if __name__ == '__main__':
    main()
