"""
Demo 3: Data Preprocessing for AM

This script demonstrates essential data preprocessing techniques for AM sensor data.
We work with time-series thermal data and show common preprocessing steps.

Learning Goals:
- Handle time-series sensor data
- Extract meaningful features from raw data
- Deal with noise and outliers
- Understand feature engineering for AM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import signal, stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
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


def load_time_series_data(data_path):
    """Load raw sensor data."""
    logger.info("="*70)
    logger.info("STEP 1: LOADING RAW SENSOR DATA")
    logger.info("="*70)
    
    df = pd.read_csv(data_path)
    logger.info(f"✓ Loaded time-series data: {len(df)} timesteps")
    logger.info(f"✓ Columns: {list(df.columns)}")
    logger.info(f"✓ Duration: {df['time_s'].max():.2f} seconds")
    logger.info(f"✓ Sampling rate: ~{1/(df['time_s'].diff().mean()):.0f} Hz")
    
    # Basic statistics
    logger.info(f"\nTemperature Statistics:")
    logger.info(f"  Mean: {df['temperature_C'].mean():.2f} °C")
    logger.info(f"  Std Dev: {df['temperature_C'].std():.2f} °C")
    logger.info(f"  Min: {df['temperature_C'].min():.2f} °C")
    logger.info(f"  Max: {df['temperature_C'].max():.2f} °C")
    
    return df


def visualize_raw_data(df, output_dir):
    """Visualize raw time-series data."""
    logger.info("\n" + "="*70)
    logger.info("STEP 2: VISUALIZING RAW DATA")
    logger.info("="*70)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Full time series
    axes[0].plot(df['time_s'], df['temperature_C'], linewidth=0.8, alpha=0.7)
    axes[0].set_xlabel('Time (s)', fontsize=12)
    axes[0].set_ylabel('Temperature (°C)', fontsize=12)
    axes[0].set_title('Raw Thermal Sensor Data', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Add mean line
    mean_temp = df['temperature_C'].mean()
    axes[0].axhline(y=mean_temp, color='r', linestyle='--', 
                    label=f'Mean: {mean_temp:.1f}°C', linewidth=2)
    axes[0].legend()
    
    # Plot 2: Histogram
    axes[1].hist(df['temperature_C'], bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Temperature (°C)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Temperature Distribution', fontsize=14, fontweight='bold')
    axes[1].axvline(x=mean_temp, color='r', linestyle='--', 
                    label=f'Mean: {mean_temp:.1f}°C', linewidth=2)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    raw_path = output_dir / '01_raw_sensor_data.png'
    plt.savefig(raw_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved visualization: {raw_path}")
    plt.close()
    
    logger.info("✓ We can see periodic variations and some noise/spikes")


def detect_and_handle_outliers(df, column='temperature_C', threshold=2.5, output_dir=None):
    """Detect and handle outliers using z-score method."""
    logger.info("\n" + "="*70)
    logger.info("STEP 3: DETECTING AND HANDLING OUTLIERS")
    logger.info("="*70)
    
    logger.info(f"Using z-score method with threshold: {threshold}")
    logger.info(f"  (More aggressive than default 3.0 - removes more outliers)")
    
    # Calculate z-scores
    z_scores = np.abs(stats.zscore(df[column]))
    outliers = z_scores > threshold
    n_outliers = outliers.sum()
    
    logger.info(f"\u2713 Detected {n_outliers} outliers ({n_outliers/len(df)*100:.2f}% of data)")
    
    if n_outliers > 0:
        logger.info(f"  Outlier temperature range: {df[column][outliers].min():.1f} to {df[column][outliers].max():.1f} °C")
        logger.info(f"  Normal range: {df[column][~outliers].min():.1f} to {df[column][~outliers].max():.1f} °C")
    
    # Create cleaned data (replace outliers with median)
    df_cleaned = df.copy()
    df_cleaned[column] = df[column].copy()
    
    # Replace outliers with median (more robust than mean)
    median_temp = df[column].median()
    df_cleaned.loc[outliers, column] = median_temp
    logger.info(f"\n✓ Replaced outliers with median value: {median_temp:.1f} °C")
    
    # Visualize outlier detection
    if output_dir:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Before and after
        axes[0].plot(df['time_s'], df[column], linewidth=0.8, alpha=0.5, label='Original', color='blue')
        axes[0].scatter(df['time_s'][outliers], df[column][outliers], 
                       c='red', s=50, alpha=0.8, label=f'Outliers ({n_outliers})', zorder=5)
        axes[0].set_xlabel('Time (s)', fontsize=12)
        axes[0].set_ylabel('Temperature (°C)', fontsize=12)
        axes[0].set_title('Outlier Detection (Red points = Outliers)', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Cleaned data
        axes[1].plot(df['time_s'], df_cleaned[column], linewidth=0.8, alpha=0.7, color='green')
        axes[1].set_xlabel('Time (s)', fontsize=12)
        axes[1].set_ylabel('Temperature (°C)', fontsize=12)
        axes[1].set_title('After Outlier Removal', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        outlier_path = output_dir / '02_outlier_removal.png'
        plt.savefig(outlier_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Saved outlier visualization: {outlier_path}")
        plt.close()
    
    return df_cleaned, outliers


def apply_smoothing_filters(df, column='temperature_C'):
    """Apply different smoothing techniques."""
    logger.info("\n" + "="*70)
    logger.info("STEP 4: APPLYING SMOOTHING FILTERS")
    logger.info("="*70)
    
    logger.info("Testing three smoothing methods...")
    
    # Method 1: Moving average
    window_size = 10
    df['temp_moving_avg'] = df[column].rolling(window=window_size, center=True).mean()
    logger.info(f"✓ Moving Average (window={window_size})")
    
    # Method 2: Savitzky-Golay filter (preserves peaks better)
    df['temp_savgol'] = signal.savgol_filter(df[column], window_length=11, polyorder=2)
    logger.info(f"✓ Savitzky-Golay Filter (window=11, polyorder=2)")
    
    # Method 3: Exponential weighted moving average
    span = 10
    df['temp_ewma'] = df[column].ewm(span=span).mean()
    logger.info(f"✓ Exponential Weighted Moving Average (span={span})")
    
    logger.info("\nSmoothing Trade-offs:")
    logger.info("  • Moving Average: Simple, but lags behind rapid changes")
    logger.info("  • Savitzky-Golay: Better preserves peaks and features")
    logger.info("  • EWMA: Responsive to recent changes, smooth for trends")
    
    return df


def visualize_smoothing_comparison(df, output_dir):
    """Compare different smoothing methods."""
    logger.info("\n" + "="*70)
    logger.info("STEP 5: VISUALIZING SMOOTHING EFFECTS")
    logger.info("="*70)
    
    # Select a zoomed-in portion for clarity
    start_idx = 200
    end_idx = 400
    df_zoom = df.iloc[start_idx:end_idx]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Comparison of Smoothing Techniques', fontsize=16, fontweight='bold')
    
    # Plot 1: Original vs Moving Average
    axes[0, 0].plot(df_zoom['time_s'], df_zoom['temperature_C'], 
                    alpha=0.5, label='Original', linewidth=1)
    axes[0, 0].plot(df_zoom['time_s'], df_zoom['temp_moving_avg'], 
                    label='Moving Average', linewidth=2, color='red')
    axes[0, 0].set_ylabel('Temperature (°C)', fontsize=11)
    axes[0, 0].set_title('Moving Average Filter', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Original vs Savitzky-Golay
    axes[0, 1].plot(df_zoom['time_s'], df_zoom['temperature_C'], 
                    alpha=0.5, label='Original', linewidth=1)
    axes[0, 1].plot(df_zoom['time_s'], df_zoom['temp_savgol'], 
                    label='Savitzky-Golay', linewidth=2, color='green')
    axes[0, 1].set_ylabel('Temperature (°C)', fontsize=11)
    axes[0, 1].set_title('Savitzky-Golay Filter', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Original vs EWMA
    axes[1, 0].plot(df_zoom['time_s'], df_zoom['temperature_C'], 
                    alpha=0.5, label='Original', linewidth=1)
    axes[1, 0].plot(df_zoom['time_s'], df_zoom['temp_ewma'], 
                    label='EWMA', linewidth=2, color='purple')
    axes[1, 0].set_xlabel('Time (s)', fontsize=11)
    axes[1, 0].set_ylabel('Temperature (°C)', fontsize=11)
    axes[1, 0].set_title('Exponential Weighted Moving Average', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: All together
    axes[1, 1].plot(df_zoom['time_s'], df_zoom['temperature_C'], 
                    alpha=0.3, label='Original', linewidth=1)
    axes[1, 1].plot(df_zoom['time_s'], df_zoom['temp_moving_avg'], 
                    label='Moving Avg', linewidth=1.5)
    axes[1, 1].plot(df_zoom['time_s'], df_zoom['temp_savgol'], 
                    label='Savitzky-Golay', linewidth=1.5)
    axes[1, 1].plot(df_zoom['time_s'], df_zoom['temp_ewma'], 
                    label='EWMA', linewidth=1.5)
    axes[1, 1].set_xlabel('Time (s)', fontsize=11)
    axes[1, 1].set_ylabel('Temperature (°C)', fontsize=11)
    axes[1, 1].set_title('All Methods Compared', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    smooth_path = output_dir / '02_smoothing_comparison.png'
    plt.savefig(smooth_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved visualization: {smooth_path}")
    plt.close()


def extract_statistical_features(df, column='temperature_C', window_size=50):
    """Extract rolling statistical features."""
    logger.info("\n" + "="*70)
    logger.info("STEP 6: EXTRACTING STATISTICAL FEATURES")
    logger.info("="*70)
    
    logger.info(f"Computing rolling features with window size: {window_size}")
    
    # Rolling statistics
    df['temp_rolling_mean'] = df[column].rolling(window=window_size).mean()
    df['temp_rolling_std'] = df[column].rolling(window=window_size).std()
    df['temp_rolling_min'] = df[column].rolling(window=window_size).min()
    df['temp_rolling_max'] = df[column].rolling(window=window_size).max()
    df['temp_rolling_range'] = df['temp_rolling_max'] - df['temp_rolling_min']
    
    logger.info("✓ Extracted features:")
    logger.info("  • Rolling mean (local average)")
    logger.info("  • Rolling std (local variability)")
    logger.info("  • Rolling min/max (local extremes)")
    logger.info("  • Rolling range (local stability)")
    
    # Compute derivatives (rate of change)
    df['temp_derivative'] = df[column].diff() / df['time_s'].diff()
    df['temp_derivative_smooth'] = df['temp_derivative'].rolling(window=5).mean()
    
    logger.info("  • Temperature derivative (cooling/heating rate)")
    
    # Peak detection
    peaks, properties = signal.find_peaks(df[column], prominence=10, distance=20)
    logger.info(f"\n✓ Detected {len(peaks)} temperature peaks")
    
    logger.info("\nWhy These Features Matter in AM:")
    logger.info("  • Mean temperature → energy input level")
    logger.info("  • Std deviation → process stability")
    logger.info("  • Cooling rate → microstructure formation")
    logger.info("  • Peaks → layer transitions or anomalies")
    
    return df, peaks


def demonstrate_scaling_methods(df, features_to_scale):
    """Compare different scaling methods."""
    logger.info("\n" + "="*70)
    logger.info("STEP 7: COMPARING SCALING METHODS")
    logger.info("="*70)
    
    # Extract features (drop NaN from rolling calculations)
    feature_data = df[features_to_scale].dropna()
    
    logger.info(f"Comparing scalers on {len(features_to_scale)} features...")
    
    # Method 1: StandardScaler (z-score normalization)
    standard_scaler = StandardScaler()
    data_standard = standard_scaler.fit_transform(feature_data)
    logger.info("\n✓ StandardScaler (mean=0, std=1)")
    logger.info(f"  Transformed mean: {data_standard.mean(axis=0).mean():.6f}")
    logger.info(f"  Transformed std: {data_standard.std(axis=0).mean():.6f}")
    
    # Method 2: MinMaxScaler (scale to [0, 1])
    minmax_scaler = MinMaxScaler()
    data_minmax = minmax_scaler.fit_transform(feature_data)
    logger.info("\n✓ MinMaxScaler (range=[0, 1])")
    logger.info(f"  Transformed min: {data_minmax.min():.6f}")
    logger.info(f"  Transformed max: {data_minmax.max():.6f}")
    
    # Method 3: RobustScaler (resistant to outliers)
    robust_scaler = RobustScaler()
    data_robust = robust_scaler.fit_transform(feature_data)
    logger.info("\n✓ RobustScaler (uses median and IQR)")
    logger.info(f"  Transformed median: {np.median(data_robust, axis=0).mean():.6f}")
    
    logger.info("\nWhen to Use Each Scaler:")
    logger.info("  • StandardScaler: Most common, assumes Gaussian distribution")
    logger.info("  • MinMaxScaler: When you need bounded range [0,1]")
    logger.info("  • RobustScaler: When data has outliers")
    
    return data_standard, data_minmax, data_robust, feature_data


def visualize_scaling_effects(feature_data, data_standard, data_minmax, data_robust, output_dir):
    """Visualize the effect of different scalers."""
    logger.info("\n" + "="*70)
    logger.info("STEP 8: VISUALIZING SCALING EFFECTS")
    logger.info("="*70)
    
    # Select one feature to visualize
    feature_idx = 0
    feature_name = feature_data.columns[feature_idx]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Scaling Methods Comparison: {feature_name}', 
                 fontsize=16, fontweight='bold')
    
    # Original data
    axes[0, 0].hist(feature_data.iloc[:, feature_idx], bins=50, 
                    edgecolor='black', alpha=0.7, color='gray')
    axes[0, 0].set_title('Original Data', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Value', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # StandardScaler
    axes[0, 1].hist(data_standard[:, feature_idx], bins=50, 
                    edgecolor='black', alpha=0.7, color='blue')
    axes[0, 1].set_title('StandardScaler (mean=0, std=1)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Value', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Mean=0')
    axes[0, 1].legend()
    
    # MinMaxScaler
    axes[1, 0].hist(data_minmax[:, feature_idx], bins=50, 
                    edgecolor='black', alpha=0.7, color='green')
    axes[1, 0].set_title('MinMaxScaler (range=[0,1])', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Value', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=1, label='Min=0')
    axes[1, 0].axvline(x=1, color='r', linestyle='--', linewidth=1, label='Max=1')
    axes[1, 0].legend()
    
    # RobustScaler
    axes[1, 1].hist(data_robust[:, feature_idx], bins=50, 
                    edgecolor='black', alpha=0.7, color='purple')
    axes[1, 1].set_title('RobustScaler (median-centered)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Value', fontsize=11)
    axes[1, 1].set_ylabel('Frequency', fontsize=11)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Median≈0')
    axes[1, 1].legend()
    
    plt.tight_layout()
    scaling_path = output_dir / '03_scaling_comparison.png'
    plt.savefig(scaling_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved visualization: {scaling_path}")
    plt.close()


def create_feature_summary(df, output_dir):
    """Create comprehensive feature summary visualization."""
    logger.info("\n" + "="*70)
    logger.info("STEP 9: CREATING FEATURE SUMMARY")
    logger.info("="*70)
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('Extracted Features Overview', fontsize=16, fontweight='bold')
    
    # Select a subset for visualization
    start, end = 100, 600
    df_viz = df.iloc[start:end]
    
    # Plot 1: Original vs smoothed
    axes[0, 0].plot(df_viz['time_s'], df_viz['temperature_C'], 
                    alpha=0.4, label='Original', linewidth=1)
    axes[0, 0].plot(df_viz['time_s'], df_viz['temp_savgol'], 
                    label='Smoothed', linewidth=2, color='red')
    axes[0, 0].set_ylabel('Temperature (°C)', fontsize=11)
    axes[0, 0].set_title('Raw vs Smoothed Data', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Rolling mean and std
    ax2 = axes[0, 1]
    ax2.plot(df_viz['time_s'], df_viz['temp_rolling_mean'], 
             label='Rolling Mean', linewidth=2, color='blue')
    ax2.fill_between(df_viz['time_s'], 
                      df_viz['temp_rolling_mean'] - df_viz['temp_rolling_std'],
                      df_viz['temp_rolling_mean'] + df_viz['temp_rolling_std'],
                      alpha=0.3, label='±1 Std Dev')
    ax2.set_ylabel('Temperature (°C)', fontsize=11)
    ax2.set_title('Rolling Mean ± Std Dev', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Rolling range
    axes[1, 0].plot(df_viz['time_s'], df_viz['temp_rolling_range'], 
                    linewidth=2, color='green')
    axes[1, 0].set_ylabel('Temperature Range (°C)', fontsize=11)
    axes[1, 0].set_title('Rolling Range (Stability Indicator)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=df_viz['temp_rolling_range'].mean(), 
                       color='r', linestyle='--', label='Mean Range')
    axes[1, 0].legend()
    
    # Plot 4: Derivative (cooling/heating rate)
    axes[1, 1].plot(df_viz['time_s'], df_viz['temp_derivative_smooth'], 
                    linewidth=2, color='purple')
    axes[1, 1].set_ylabel('dT/dt (°C/s)', fontsize=11)
    axes[1, 1].set_title('Temperature Rate of Change', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    
    # Plot 5: Min/Max envelope
    axes[2, 0].plot(df_viz['time_s'], df_viz['temperature_C'], 
                    alpha=0.3, linewidth=0.5, color='gray', label='Raw Data')
    axes[2, 0].plot(df_viz['time_s'], df_viz['temp_rolling_min'], 
                    label='Rolling Min', linewidth=2, color='blue')
    axes[2, 0].plot(df_viz['time_s'], df_viz['temp_rolling_max'], 
                    label='Rolling Max', linewidth=2, color='red')
    axes[2, 0].set_xlabel('Time (s)', fontsize=11)
    axes[2, 0].set_ylabel('Temperature (°C)', fontsize=11)
    axes[2, 0].set_title('Rolling Min/Max Envelope', fontsize=12, fontweight='bold')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Plot 6: Feature correlation heatmap
    feature_cols = ['temperature_C', 'temp_rolling_mean', 'temp_rolling_std', 
                    'temp_rolling_range']
    corr_matrix = df[feature_cols].corr()
    im = axes[2, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    axes[2, 1].set_xticks(range(len(feature_cols)))
    axes[2, 1].set_yticks(range(len(feature_cols)))
    axes[2, 1].set_xticklabels([col.replace('temp_', '').replace('_', ' ').title() 
                                 for col in feature_cols], rotation=45, ha='right', fontsize=9)
    axes[2, 1].set_yticklabels([col.replace('temp_', '').replace('_', ' ').title() 
                                 for col in feature_cols], fontsize=9)
    axes[2, 1].set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')
    
    # Add correlation values
    for i in range(len(feature_cols)):
        for j in range(len(feature_cols)):
            text = axes[2, 1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, ax=axes[2, 1])
    
    plt.tight_layout()
    summary_path = output_dir / '04_feature_summary.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved visualization: {summary_path}")
    plt.close()


def main():
    """Main execution function."""
    logger.info("\n" + "="*70)
    logger.info("DATA PREPROCESSING DEMO: TIME-SERIES SENSOR DATA")
    logger.info("="*70)
    logger.info("This demo shows essential preprocessing steps for AM sensor data")
    logger.info("="*70 + "\n")
    
    # Setup paths
    script_dir = Path(__file__).parent
    data_path = script_dir / 'data' / 'time_series_sensor.csv'
    output_dir = script_dir / 'outputs' / '03_data_preprocessing'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Execute pipeline
    df = load_time_series_data(data_path)
    visualize_raw_data(df, output_dir)
    df_cleaned, outliers = detect_and_handle_outliers(df)
    df_smoothed = apply_smoothing_filters(df_cleaned)
    visualize_smoothing_comparison(df_smoothed, output_dir)
    df_features, peaks = extract_statistical_features(df_smoothed)
    
    # Scaling demonstration
    features_to_scale = ['temp_rolling_mean', 'temp_rolling_std', 'temp_rolling_range']
    data_standard, data_minmax, data_robust, feature_data = demonstrate_scaling_methods(
        df_features, features_to_scale
    )
    visualize_scaling_effects(feature_data, data_standard, data_minmax, data_robust, output_dir)
    
    # Create comprehensive summary
    create_feature_summary(df_features, output_dir)
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY AND KEY TAKEAWAYS")
    logger.info("="*70)
    
    logger.info("\n✓ Preprocessing Pipeline Complete")
    logger.info(f"  • Original data points: {len(df)}")
    logger.info(f"  • Outliers detected: {outliers.sum()}")
    logger.info(f"  • Temperature peaks detected: {len(peaks)}")
    logger.info(f"  • Features extracted: {len([col for col in df_features.columns if 'temp_' in col])}")
    
    logger.info("\nKey Preprocessing Steps Demonstrated:")
    logger.info("  1. Outlier detection and handling (z-score method)")
    logger.info("  2. Smoothing filters (moving average, Savitzky-Golay, EWMA)")
    logger.info("  3. Feature extraction (rolling statistics, derivatives)")
    logger.info("  4. Scaling methods (Standard, MinMax, Robust)")
    
    logger.info("\nBest Practices for AM Data:")
    logger.info("  • Always visualize raw data first")
    logger.info("  • Handle outliers before feature extraction")
    logger.info("  • Choose smoothing based on your application")
    logger.info("  • Extract domain-relevant features (cooling rates, stability)")
    logger.info("  • Scale features before ML model training")
    logger.info("  • Document all preprocessing steps for reproducibility")
    
    logger.info(f"\n✓ All outputs saved to: {output_dir.absolute()}")
    logger.info("\n" + "="*70)
    logger.info("DEMO COMPLETE")
    logger.info("="*70 + "\n")


if __name__ == '__main__':
    main()
