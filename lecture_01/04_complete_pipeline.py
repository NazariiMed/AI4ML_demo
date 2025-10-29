"""
Demo 4: Complete ML Pipeline for AM

This script demonstrates the complete machine learning workflow from start to finish.
We combine all previous concepts into a realistic end-to-end pipeline.

Learning Goals:
- Understand the complete ML workflow
- See how all pieces fit together
- Practice cross-validation
- Learn model comparison and selection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
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
plt.rcParams['figure.figsize'] = (14, 8)


def load_and_prepare_data(data_path):
    """Step 1: Load data and initial exploration."""
    logger.info("="*70)
    logger.info("STEP 1: PROBLEM DEFINITION & DATA LOADING")
    logger.info("="*70)
    
    logger.info("\nProblem Statement:")
    logger.info("  Predict surface roughness from FDM process parameters")
    logger.info("  Goal: R² > 0.85, MAE < 0.5 μm")
    
    df = pd.read_csv(data_path)
    logger.info(f"\n✓ Loaded dataset: {len(df)} samples, {len(df.columns)} features")
    
    return df


def exploratory_data_analysis(df):
    """Step 2: Explore and understand the data."""
    logger.info("\n" + "="*70)
    logger.info("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
    logger.info("="*70)
    
    logger.info("\nDataset Shape: {} samples × {} features".format(*df.shape))
    
    # Check for missing values
    missing = df.isnull().sum()
    logger.info(f"\n✓ Missing values: {missing.sum()} total")
    
    # Basic statistics
    logger.info("\nTarget Variable (Surface Roughness):")
    logger.info(f"  Mean: {df['surface_roughness_um'].mean():.2f} μm")
    logger.info(f"  Std: {df['surface_roughness_um'].std():.2f} μm")
    logger.info(f"  Range: [{df['surface_roughness_um'].min():.2f}, {df['surface_roughness_um'].max():.2f}] μm")
    
    # Correlation analysis
    logger.info("\nTop Correlations with Target:")
    correlations = df.corr()['surface_roughness_um'].abs().sort_values(ascending=False)[1:]
    for feat, corr in correlations.head(3).items():
        logger.info(f"  {feat:30s}: {corr:.3f}")
    
    return df


def feature_engineering(df):
    """Step 3: Create new features based on domain knowledge."""
    logger.info("\n" + "="*70)
    logger.info("STEP 3: FEATURE ENGINEERING")
    logger.info("="*70)
    
    logger.info("Creating physics-informed features...")
    
    # Create derived features
    df['temp_deviation'] = np.abs(df['nozzle_temp_C'] - 205)  # Deviation from optimal
    df['temp_ratio'] = df['nozzle_temp_C'] / df['bed_temp_C']
    df['print_intensity'] = df['print_speed_mm_s'] * df['layer_height_mm']
    
    logger.info("✓ Created 3 new features:")
    logger.info("  • temp_deviation: Distance from optimal nozzle temp")
    logger.info("  • temp_ratio: Nozzle to bed temperature ratio")
    logger.info("  • print_intensity: Speed × layer height")
    
    logger.info("\nFeature Engineering Rationale:")
    logger.info("  Physics tells us optimal temperature exists (not linear)")
    logger.info("  Temperature ratio affects layer adhesion")
    logger.info("  Print intensity combines two related parameters")
    
    return df


def prepare_train_test_split(df):
    """Step 4: Split data properly."""
    logger.info("\n" + "="*70)
    logger.info("STEP 4: TRAIN-TEST SPLIT")
    logger.info("="*70)
    
    # Separate features and target
    X = df.drop('surface_roughness_um', axis=1)
    y = df['surface_roughness_um']
    
    logger.info(f"Features (X): {list(X.columns)}")
    logger.info(f"\nTotal features: {X.shape[1]}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"\n✓ Data split completed:")
    logger.info(f"  Training set: {len(X_train)} samples ({len(X_train)/len(df)*100:.0f}%)")
    logger.info(f"  Test set: {len(X_test)} samples ({len(X_test)/len(df)*100:.0f}%)")
    
    logger.info("\n✓ Test set held completely separate - will NEVER be used for training!")
    
    return X_train, X_test, y_train, y_test, X.columns


def preprocess_features(X_train, X_test):
    """Step 5: Scale features."""
    logger.info("\n" + "="*70)
    logger.info("STEP 5: FEATURE PREPROCESSING")
    logger.info("="*70)
    
    logger.info("Applying StandardScaler...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info("✓ Features scaled to zero mean and unit variance")
    logger.info("  CRITICAL: Scaler fitted ONLY on training data")
    logger.info("  Test data transformed using training statistics")
    logger.info("  This prevents data leakage!")
    
    return X_train_scaled, X_test_scaled, scaler


def train_baseline_models(X_train, X_test, y_train, y_test):
    """Step 6: Train multiple baseline models."""
    logger.info("\n" + "="*70)
    logger.info("STEP 6: TRAINING BASELINE MODELS")
    logger.info("="*70)
    
    logger.info("Training 7 different algorithms...")
    logger.info("Strategy: Start simple, add complexity\n")
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
        'Support Vector Machine': SVR(kernel='rbf', C=1.0)
    }
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"{'─'*70}")
        logger.info(f"Training: {name}")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Evaluate
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        logger.info(f"  Train R²: {train_r2:.4f}")
        logger.info(f"  Test R²:  {test_r2:.4f}")
        logger.info(f"  Test MAE: {test_mae:.4f} μm")
        
        # Check overfitting
        gap = train_r2 - test_r2
        if gap > 0.15:
            logger.warning(f"  ⚠ Overfitting detected! Gap: {gap:.4f}")
        elif gap > 0.05:
            logger.info(f"  ⚠ Minor overfitting. Gap: {gap:.4f}")
        else:
            logger.info(f"  ✓ Good generalization. Gap: {gap:.4f}")
        
        results[name] = {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'predictions': y_test_pred
        }
    
    return results


def perform_cross_validation(X_train, y_train, models_dict):
    """Step 7: Perform k-fold cross-validation."""
    logger.info("\n" + "="*70)
    logger.info("STEP 7: CROSS-VALIDATION")
    logger.info("="*70)
    
    logger.info("Performing 5-fold cross-validation...")
    logger.info("  This gives more robust performance estimates\n")
    
    cv_results = {}
    
    for name, model in models_dict.items():
        scores = cross_val_score(model, X_train, y_train, 
                                cv=5, scoring='r2', n_jobs=-1)
        
        cv_results[name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
        
        logger.info(f"{name:25s}: R² = {scores.mean():.4f} ± {scores.std():.4f}")
    
    logger.info("\n✓ Cross-validation complete")
    logger.info("  The ± value shows variability across folds")
    logger.info("  Lower variability = more stable model")
    
    return cv_results


def hyperparameter_tuning(X_train, y_train):
    """Step 8: Tune the best model."""
    logger.info("\n" + "="*70)
    logger.info("STEP 8: HYPERPARAMETER TUNING")
    logger.info("="*70)
    
    logger.info("Tuning Random Forest hyperparameters using Grid Search...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    logger.info(f"\nParameter grid:")
    for param, values in param_grid.items():
        logger.info(f"  {param}: {values}")
    
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    logger.info(f"\nTotal combinations to test: {total_combinations}")
    logger.info(f"With 5-fold CV: {total_combinations * 5} model fits")
    logger.info("\nThis may take a moment...")
    
    # Perform grid search
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, 
        cv=5, 
        scoring='r2',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    logger.info("\n✓ Grid search complete!")
    logger.info(f"\nBest parameters found:")
    for param, value in grid_search.best_params_.items():
        logger.info(f"  {param}: {value}")
    
    logger.info(f"\nBest cross-validation R²: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search


def final_model_evaluation(model, X_train, X_test, y_train, y_test, feature_names):
    """Step 9: Final evaluation on test set."""
    logger.info("\n" + "="*70)
    logger.info("STEP 9: FINAL MODEL EVALUATION")
    logger.info("="*70)
    
    logger.info("Evaluating tuned model on held-out test set...\n")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
    
    logger.info("="*70)
    logger.info("FINAL MODEL PERFORMANCE")
    logger.info("="*70)
    logger.info(f"\nTraining Set:")
    logger.info(f"  R² Score: {train_r2:.4f}")
    logger.info(f"\nTest Set (Unseen Data):")
    logger.info(f"  R² Score: {test_r2:.4f}")
    logger.info(f"  MAE: {test_mae:.4f} μm")
    logger.info(f"  RMSE: {test_rmse:.4f} μm")
    logger.info(f"  MAPE: {test_mape:.2f}%")
    
    # Goal assessment
    logger.info("\n" + "="*70)
    logger.info("GOAL ASSESSMENT")
    logger.info("="*70)
    goal_r2 = 0.85
    goal_mae = 0.5
    
    r2_met = "✓" if test_r2 >= goal_r2 else "✗"
    mae_met = "✓" if test_mae <= goal_mae else "✗"
    
    logger.info(f"  {r2_met} R² > {goal_r2}: {test_r2:.4f}")
    logger.info(f"  {mae_met} MAE < {goal_mae} μm: {test_mae:.4f}")
    
    if test_r2 >= goal_r2 and test_mae <= goal_mae:
        logger.info("\n🎉 SUCCESS! Model meets all performance goals!")
    else:
        logger.info("\n⚠ Model needs improvement to meet goals")
    
    # Feature importance
    logger.info("\n" + "="*70)
    logger.info("FEATURE IMPORTANCE")
    logger.info("="*70)
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    logger.info("\nTop features for predicting surface roughness:")
    for i, idx in enumerate(indices[:5], 1):
        logger.info(f"  {i}. {feature_names[idx]:30s}: {importances[idx]:.4f}")
    
    return y_test_pred, {
        'r2': test_r2,
        'mae': test_mae,
        'rmse': test_rmse,
        'mape': test_mape
    }


def create_comprehensive_visualizations(results_baseline, y_test, y_test_pred_final, 
                                        cv_results, metrics_final, output_dir):
    """Create comprehensive visualization suite."""
    logger.info("\n" + "="*70)
    logger.info("STEP 10: CREATING VISUALIZATIONS")
    logger.info("="*70)
    
    # Visualization 1: Model comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ML Pipeline: Comprehensive Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Baseline model comparison (R²)
    model_names = list(results_baseline.keys())
    test_r2_scores = [results_baseline[name]['test_r2'] for name in model_names]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    bars = axes[0, 0].barh(model_names, test_r2_scores, color=colors)
    axes[0, 0].set_xlabel('R² Score', fontsize=11)
    axes[0, 0].set_title('Baseline Model Comparison (Test R²)', fontsize=12, fontweight='bold')
    axes[0, 0].axvline(x=0.85, color='r', linestyle='--', linewidth=2, label='Goal: 0.85')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # Add values on bars
    for i, (bar, score) in enumerate(zip(bars, test_r2_scores)):
        axes[0, 0].text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{score:.3f}', va='center', fontsize=9)
    
    # Plot 2: Cross-validation scores (only for models that have CV results)
    cv_model_names = [name for name in model_names if name in cv_results]
    cv_means = [cv_results[name]['mean'] for name in cv_model_names]
    cv_stds = [cv_results[name]['std'] for name in cv_model_names]
    cv_colors = plt.cm.viridis(np.linspace(0, 1, len(cv_model_names)))
    
    axes[0, 1].barh(cv_model_names, cv_means, xerr=cv_stds, 
                    color=cv_colors, capsize=5, alpha=0.8)
    axes[0, 1].set_xlabel('R² Score', fontsize=11)
    axes[0, 1].set_title('Cross-Validation Results (Mean ± Std)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # Plot 3: Final model predictions
    axes[1, 0].scatter(y_test, y_test_pred_final, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_test_pred_final.min())
    max_val = max(y_test.max(), y_test_pred_final.max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    axes[1, 0].set_xlabel('Actual Surface Roughness (μm)', fontsize=11)
    axes[1, 0].set_ylabel('Predicted Surface Roughness (μm)', fontsize=11)
    axes[1, 0].set_title(f'Final Model: Predicted vs Actual\nR²={metrics_final["r2"]:.4f}, MAE={metrics_final["mae"]:.4f} μm', 
                        fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Residual plot
    residuals = y_test - y_test_pred_final
    axes[1, 1].scatter(y_test_pred_final, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Predicted Surface Roughness (μm)', fontsize=11)
    axes[1, 1].set_ylabel('Residuals (μm)', fontsize=11)
    axes[1, 1].set_title('Residual Plot (Check for Patterns)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add residual statistics
    residual_text = f'Mean: {residuals.mean():.3f}\nStd: {residuals.std():.3f}'
    axes[1, 1].text(0.05, 0.95, residual_text, transform=axes[1, 1].transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    viz_path = output_dir / '01_complete_pipeline_results.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved visualization: {viz_path}")
    plt.close()
    
    # Visualization 2: Learning curve / overfitting analysis
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    model_names_sorted = sorted(results_baseline.keys(), 
                                key=lambda x: results_baseline[x]['test_r2'], 
                                reverse=True)
    
    train_scores = [results_baseline[name]['train_r2'] for name in model_names_sorted]
    test_scores = [results_baseline[name]['test_r2'] for name in model_names_sorted]
    
    x = np.arange(len(model_names_sorted))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train_scores, width, label='Training R²', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_scores, width, label='Test R²', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('Train vs Test Performance: Detecting Overfitting', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names_sorted, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])
    
    # Add gap indicators
    for i, (train, test) in enumerate(zip(train_scores, test_scores)):
        gap = train - test
        if gap > 0.1:
            ax.plot([i-width/2, i+width/2], [train, test], 'r-', linewidth=2, alpha=0.5)
    
    plt.tight_layout()
    overfit_path = output_dir / '02_overfitting_analysis.png'
    plt.savefig(overfit_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved visualization: {overfit_path}")
    plt.close()


def save_final_model(model, scaler, feature_names, output_dir):
    """Save the final model for deployment."""
    logger.info("\n" + "="*70)
    logger.info("STEP 11: MODEL PERSISTENCE")
    logger.info("="*70)
    
    import joblib
    
    # Save model
    model_path = output_dir / 'final_model.joblib'
    joblib.dump(model, model_path)
    logger.info(f"✓ Saved model: {model_path}")
    
    # Save scaler
    scaler_path = output_dir / 'scaler.joblib'
    joblib.dump(scaler, scaler_path)
    logger.info(f"✓ Saved scaler: {scaler_path}")
    
    # Save feature names
    features_path = output_dir / 'feature_names.txt'
    with open(features_path, 'w') as f:
        for feat in feature_names:
            f.write(f"{feat}\n")
    logger.info(f"✓ Saved feature names: {features_path}")
    
    logger.info("\nThese files can be loaded later for making predictions on new data")


def main():
    """Main execution function."""
    logger.info("\n" + "="*70)
    logger.info("COMPLETE ML PIPELINE DEMO")
    logger.info("="*70)
    logger.info("This demonstrates the full ML workflow from data to deployment")
    logger.info("="*70 + "\n")
    
    # Setup paths
    script_dir = Path(__file__).parent
    data_path = script_dir / 'data' / 'fdm_surface_roughness.csv'
    output_dir = script_dir / 'outputs' / '04_complete_pipeline'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Execute complete pipeline
    df = load_and_prepare_data(data_path)
    df = exploratory_data_analysis(df)
    df = feature_engineering(df)
    X_train, X_test, y_train, y_test, feature_names = prepare_train_test_split(df)
    X_train_scaled, X_test_scaled, scaler = preprocess_features(X_train, X_test)
    
    # Model training and selection
    results_baseline = train_baseline_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Get models dict for CV (need unscaled for this)
    models_for_cv = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    }
    cv_results = perform_cross_validation(X_train_scaled, y_train, models_for_cv)
    
    # Hyperparameter tuning
    best_model, grid_search = hyperparameter_tuning(X_train_scaled, y_train)
    
    # Final evaluation
    y_test_pred_final, metrics_final = final_model_evaluation(
        best_model, X_train_scaled, X_test_scaled, y_train, y_test, feature_names
    )
    
    # Visualizations
    create_comprehensive_visualizations(
        results_baseline, y_test, y_test_pred_final, 
        cv_results, metrics_final, output_dir
    )
    
    # Save model
    save_final_model(best_model, scaler, feature_names, output_dir)
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info("COMPLETE PIPELINE SUMMARY")
    logger.info("="*70)
    
    logger.info("\n✓ Pipeline Steps Completed:")
    logger.info("  1. Problem definition & data loading")
    logger.info("  2. Exploratory data analysis")
    logger.info("  3. Feature engineering (3 new features)")
    logger.info("  4. Train-test split (80/20)")
    logger.info("  5. Feature scaling (StandardScaler)")
    logger.info("  6. Baseline model training (7 algorithms)")
    logger.info("  7. Cross-validation (5-fold)")
    logger.info("  8. Hyperparameter tuning (Grid Search)")
    logger.info("  9. Final evaluation on test set")
    logger.info("  10. Comprehensive visualizations")
    logger.info("  11. Model persistence for deployment")
    
    logger.info("\n✓ Key Learnings:")
    logger.info("  • Always split data before any preprocessing")
    logger.info("  • Feature engineering improves performance")
    logger.info("  • Cross-validation provides robust estimates")
    logger.info("  • Hyperparameter tuning can boost performance")
    logger.info("  • Monitor train-test gap for overfitting")
    logger.info("  • Save everything needed for deployment")
    
    logger.info(f"\n✓ All outputs saved to: {output_dir.absolute()}")
    logger.info("\n" + "="*70)
    logger.info("COMPLETE PIPELINE DEMO FINISHED")
    logger.info("="*70 + "\n")


if __name__ == '__main__':
    main()
