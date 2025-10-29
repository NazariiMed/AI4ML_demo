"""
Data Generator for ML in AM Course
Generates synthetic datasets mimicking real AM process data

This script creates realistic synthetic data for educational purposes.
The relationships and noise levels are based on typical AM process behavior.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

def generate_fdm_surface_roughness_data(n_samples=200, noise_level=0.08):
    """
    Generate synthetic FDM process data with surface roughness output.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    noise_level : float
        Amount of noise to add (std dev as fraction of signal)
    
    Returns:
    --------
    pd.DataFrame with columns: layer_height, print_speed, nozzle_temp, 
                               bed_temp, infill_density, surface_roughness
    """
    print(f"Generating {n_samples} FDM surface roughness samples...")
    
    # Generate process parameters (realistic ranges for PLA)
    layer_height = np.random.uniform(0.1, 0.3, n_samples)  # mm
    print_speed = np.random.uniform(30, 80, n_samples)      # mm/s
    nozzle_temp = np.random.uniform(190, 220, n_samples)    # °C
    bed_temp = np.random.uniform(50, 70, n_samples)         # °C
    infill_density = np.random.uniform(10, 100, n_samples)  # %
    
    # Generate surface roughness based on physics-inspired relationships
    # Ra increases with layer height (dominant factor)
    # Ra increases with print speed (less time for settling)
    # Ra decreases with optimal temperature (better flow)
    
    # Ideal temperature (middle of range gives best results)
    temp_deviation = np.abs(nozzle_temp - 205) / 15.0
    
    # Base roughness calculation
    base_roughness = (
        2.0 * layer_height * 10 +                    # Layer height dominant
        0.02 * print_speed +                          # Speed effect
        0.3 * temp_deviation +                        # Temperature deviation
        0.01 * (100 - infill_density) +              # Lower infill = worse
        0.5 * np.sin(layer_height * 20)              # Non-linear effects
    )
    
    # Add realistic noise
    noise = np.random.normal(0, noise_level * base_roughness.mean(), n_samples)
    surface_roughness = base_roughness + noise
    
    # Ensure positive values
    surface_roughness = np.maximum(surface_roughness, 0.1)
    
    # Create DataFrame
    df = pd.DataFrame({
        'layer_height_mm': layer_height,
        'print_speed_mm_s': print_speed,
        'nozzle_temp_C': nozzle_temp,
        'bed_temp_C': bed_temp,
        'infill_density_pct': infill_density,
        'surface_roughness_um': surface_roughness
    })
    
    print(f"  Surface roughness range: {surface_roughness.min():.2f} - {surface_roughness.max():.2f} μm")
    print(f"  Mean: {surface_roughness.mean():.2f} μm")
    
    return df


def generate_defect_clustering_data(n_samples=150, n_features=5):
    """
    Generate synthetic defect data with natural clusters.
    
    Creates three types of defects:
    1. Thermal defects (high temperature, fast cooling)
    2. Mechanical defects (high stress, vibration)
    3. Material defects (contamination, poor flow)
    """
    print(f"Generating {n_samples} defect samples with {n_features} features...")
    
    samples_per_cluster = n_samples // 3
    
    # Cluster 1: Thermal defects
    thermal_defects = {
        'max_temperature': np.random.normal(350, 20, samples_per_cluster),
        'cooling_rate': np.random.normal(80, 10, samples_per_cluster),
        'energy_density': np.random.normal(120, 15, samples_per_cluster),
        'defect_size': np.random.normal(0.5, 0.1, samples_per_cluster),
        'layer_number': np.random.randint(10, 100, samples_per_cluster)
    }
    
    # Cluster 2: Mechanical defects
    mechanical_defects = {
        'max_temperature': np.random.normal(250, 15, samples_per_cluster),
        'cooling_rate': np.random.normal(40, 8, samples_per_cluster),
        'energy_density': np.random.normal(80, 12, samples_per_cluster),
        'defect_size': np.random.normal(1.2, 0.2, samples_per_cluster),
        'layer_number': np.random.randint(50, 150, samples_per_cluster)
    }
    
    # Cluster 3: Material defects
    material_defects = {
        'max_temperature': np.random.normal(280, 18, samples_per_cluster),
        'cooling_rate': np.random.normal(55, 12, samples_per_cluster),
        'energy_density': np.random.normal(95, 20, samples_per_cluster),
        'defect_size': np.random.normal(0.8, 0.15, samples_per_cluster),
        'layer_number': np.random.randint(20, 120, samples_per_cluster)
    }
    
    # Combine all clusters
    all_features = {}
    true_labels = []
    
    for feature in thermal_defects.keys():
        all_features[feature] = np.concatenate([
            thermal_defects[feature],
            mechanical_defects[feature],
            material_defects[feature]
        ])
    
    # Create true labels (for validation only, not used in unsupervised learning)
    true_labels = (
        ['thermal'] * samples_per_cluster +
        ['mechanical'] * samples_per_cluster +
        ['material'] * samples_per_cluster
    )
    
    df = pd.DataFrame(all_features)
    df['true_label'] = true_labels  # Hidden ground truth
    
    print(f"  Created 3 defect types: {samples_per_cluster} samples each")
    print(f"  Features: {', '.join(all_features.keys())}")
    
    return df


def generate_time_series_sensor_data(n_timesteps=1000, sampling_rate=100):
    """
    Generate synthetic time-series sensor data from AM process.
    
    Simulates thermal camera data during printing with:
    - Baseline temperature drift
    - Periodic variation (layer-by-layer)
    - Random noise
    - Anomalous events
    """
    print(f"Generating time-series data: {n_timesteps} timesteps @ {sampling_rate} Hz")
    
    time = np.linspace(0, n_timesteps/sampling_rate, n_timesteps)
    
    # Base temperature with slow drift
    base_temp = 250 + 10 * np.sin(2 * np.pi * time / 100)
    
    # Layer-by-layer periodic variation (assume 2 second layers)
    layer_variation = 15 * np.sin(2 * np.pi * time / 2)
    
    # High-frequency noise
    noise = np.random.normal(0, 2, n_timesteps)
    
    # Combine
    temperature = base_temp + layer_variation + noise
    
    # Add some anomalies (thermal spikes)
    anomaly_indices = np.random.choice(n_timesteps, size=5, replace=False)
    for idx in anomaly_indices:
        temperature[idx:idx+10] += 50  # Spike
    
    df = pd.DataFrame({
        'time_s': time,
        'temperature_C': temperature
    })
    
    print(f"  Duration: {time[-1]:.1f} seconds")
    print(f"  Temperature range: {temperature.min():.1f} - {temperature.max():.1f} °C")
    
    return df


def main():
    """Generate all datasets and save to data directory."""
    
    print("="*60)
    print("ML for AM Course - Data Generation")
    print("="*60)
    print()
    
    # Create output directory
    output_dir = Path(__file__).parent / 'data'
    output_dir.mkdir(exist_ok=True)
    
    # Generate datasets
    print("\n1. FDM Surface Roughness Dataset")
    print("-" * 60)
    fdm_data = generate_fdm_surface_roughness_data(n_samples=200)
    fdm_path = output_dir / 'fdm_surface_roughness.csv'
    fdm_data.to_csv(fdm_path, index=False)
    print(f"✓ Saved to: {fdm_path}")
    
    print("\n2. Defect Clustering Dataset")
    print("-" * 60)
    defect_data = generate_defect_clustering_data(n_samples=150)
    defect_path = output_dir / 'defect_clustering.csv'
    defect_data.to_csv(defect_path, index=False)
    print(f"✓ Saved to: {defect_path}")
    
    print("\n3. Time-Series Sensor Dataset")
    print("-" * 60)
    sensor_data = generate_time_series_sensor_data(n_timesteps=1000)
    sensor_path = output_dir / 'time_series_sensor.csv'
    sensor_data.to_csv(sensor_path, index=False)
    print(f"✓ Saved to: {sensor_path}")
    
    print("\n" + "="*60)
    print("Data generation complete!")
    print("="*60)
    print(f"\nAll datasets saved to: {output_dir.absolute()}")
    print("\nDataset summary:")
    print(f"  1. fdm_surface_roughness.csv - {len(fdm_data)} samples, {len(fdm_data.columns)} features")
    print(f"  2. defect_clustering.csv - {len(defect_data)} samples, {len(defect_data.columns)} features")
    print(f"  3. time_series_sensor.csv - {len(sensor_data)} timesteps")
    print()


if __name__ == '__main__':
    main()
