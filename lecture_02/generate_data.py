"""
Data Generator for Lecture 2: Process Parameter Optimization

This script generates synthetic datasets for all demos in Lecture 2.
The data is physics-informed to represent realistic AM scenarios.

Datasets created:
1. slm_surface_roughness.csv - Surface roughness prediction (regression comparison)
2. print_adhesion_strength.csv - Layer adhesion vs speed (GP demo)
3. slm_porosity_optimization.csv - Porosity minimization (Bayesian Optimization)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def generate_slm_surface_roughness(n_samples=150, noise_level=0.3):
    """
    Generate synthetic data for SLM surface roughness prediction.
    
    Physics-based model:
    - Higher laser power -> smoother (up to a point)
    - Higher scan speed -> rougher (less time for melting)
    - Thicker layers -> rougher
    - Optimal power exists (too high causes evaporation issues)
    """
    logger.info("Generating SLM surface roughness dataset...")
    
    np.random.seed(42)
    
    # Generate parameters with realistic ranges
    laser_power = np.random.uniform(150, 400, n_samples)  # W
    scan_speed = np.random.uniform(600, 1400, n_samples)  # mm/s
    layer_thickness = np.random.uniform(25, 50, n_samples)  # um
    hatch_spacing = np.random.uniform(0.08, 0.15, n_samples)  # mm
    
    # Physics-informed roughness model
    # Energy density: E = P / (v * h * t)
    energy_density = laser_power / (scan_speed * hatch_spacing * layer_thickness)
    
    # Optimal power around 280W
    power_effect = 0.5 * ((laser_power - 280) / 100) ** 2
    
    # Speed effect (higher speed = rougher)
    speed_effect = (scan_speed - 600) / 200
    
    # Layer thickness effect
    layer_effect = (layer_thickness - 25) / 10
    
    # Hatch spacing effect
    hatch_effect = (hatch_spacing - 0.08) * 20
    
    # Combined model with non-linear interactions
    surface_roughness = (
        3.0 +  # baseline
        power_effect + 
        0.8 * speed_effect + 
        0.5 * layer_effect +
        0.3 * hatch_effect +
        0.2 * power_effect * speed_effect / 100 +  # interaction
        noise_level * np.random.randn(n_samples)
    )
    
    # Clip to realistic range
    surface_roughness = np.clip(surface_roughness, 1.5, 8.0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'laser_power_W': laser_power,
        'scan_speed_mm_s': scan_speed,
        'layer_thickness_um': layer_thickness,
        'hatch_spacing_mm': hatch_spacing,
        'energy_density': energy_density,
        'surface_roughness_um': surface_roughness
    })
    
    logger.info(f"  Generated {n_samples} samples")
    logger.info(f"  Surface roughness range: {surface_roughness.min():.2f} - {surface_roughness.max():.2f} um")
    logger.info(f"  Mean: {surface_roughness.mean():.2f} um")
    
    return df


def generate_adhesion_strength(n_samples=30, noise_level=0.5):
    """
    Generate synthetic data for layer adhesion strength vs print speed.
    
    This is a smaller dataset for Gaussian Process demonstration.
    
    Physics model:
    - Optimal speed exists (around 40 mm/s for this material)
    - Too slow: overheating, degradation
    - Too fast: poor interlayer bonding
    - Non-linear relationship with clear optimum
    """
    logger.info("Generating adhesion strength dataset...")
    
    np.random.seed(43)
    
    # Print speed in mm/s (for extrusion-based AM)
    print_speed = np.random.uniform(10, 80, n_samples)
    
    # Nozzle temperature in C
    nozzle_temp = np.random.uniform(200, 240, n_samples)
    
    # Physics-informed adhesion model
    # Optimal speed around 40 mm/s, optimal temp around 220C
    speed_effect = -0.015 * (print_speed - 40) ** 2
    temp_effect = -0.01 * (nozzle_temp - 220) ** 2
    
    # Base adhesion strength
    adhesion_strength = (
        15.0 +  # baseline (MPa)
        speed_effect +
        temp_effect +
        0.002 * (print_speed - 40) * (nozzle_temp - 220) +  # interaction
        noise_level * np.random.randn(n_samples)
    )
    
    # Clip to realistic range
    adhesion_strength = np.clip(adhesion_strength, 5.0, 18.0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'print_speed_mm_s': print_speed,
        'nozzle_temp_C': nozzle_temp,
        'adhesion_strength_MPa': adhesion_strength
    })
    
    # Sort by print_speed for nicer visualization
    df = df.sort_values('print_speed_mm_s').reset_index(drop=True)
    
    logger.info(f"  Generated {n_samples} samples")
    logger.info(f"  Adhesion strength range: {adhesion_strength.min():.2f} - {adhesion_strength.max():.2f} MPa")
    logger.info(f"  Optimal around: 40 mm/s, 220C")
    
    return df


def generate_porosity_data_for_bo(n_samples=200, noise_level=0.08):
    """
    Generate synthetic data for Bayesian Optimization demonstration.
    
    This creates a challenging optimization landscape:
    - One global minimum (optimal parameters)
    - Several local minima (suboptimal regions)
    - Realistic noise
    - Non-trivial parameter interactions
    
    Physics model for porosity in SLM:
    - Depends on energy density
    - Optimal energy density exists
    - Too low: lack of fusion (high porosity)
    - Too high: keyholing (moderate porosity)
    - Parameter interactions create local minima
    """
    logger.info("Generating porosity optimization dataset...")
    
    np.random.seed(44)
    
    # Parameters with realistic ranges
    laser_power = np.random.uniform(150, 400, n_samples)  # W
    scan_speed = np.random.uniform(600, 1400, n_samples)  # mm/s
    hatch_spacing = np.random.uniform(0.08, 0.15, n_samples)  # mm
    layer_thickness = np.random.uniform(25, 50, n_samples)  # um
    
    # Energy density (J/mm3)
    energy_density = laser_power / (scan_speed * hatch_spacing * layer_thickness / 1000)
    
    # Optimal parameters
    optimal_power = 275.0  # W
    optimal_speed = 950.0  # mm/s
    optimal_hatch = 0.105  # mm
    optimal_layer = 33.0  # um
    
    # Calculate distance from optimal in normalized space
    power_dev = ((laser_power - optimal_power) / 125.0) ** 2
    speed_dev = ((scan_speed - optimal_speed) / 400.0) ** 2
    hatch_dev = ((hatch_spacing - optimal_hatch) / 0.035) ** 2
    layer_dev = ((layer_thickness - optimal_layer) / 12.5) ** 2
    
    # Base porosity model
    base_porosity = (
        0.15 +
        1.2 * power_dev +
        0.8 * speed_dev +
        0.6 * hatch_dev +
        0.5 * layer_dev
    )
    
    # Add interactions and local minima (no trailing + sign!)
    interactions = (
        0.4 * power_dev * speed_dev +
        0.3 * hatch_dev * layer_dev +
        0.2 * np.sin(12 * power_dev) * np.cos(12 * speed_dev) +
        0.15 * np.sin(15 * hatch_dev) * np.cos(10 * layer_dev) +
        0.1 * np.sin(8 * (power_dev + speed_dev))
    )
    
    # Local minimum 1: high power, low speed
    local_min_1_dist = ((laser_power - 350) / 125.0) ** 2 + ((scan_speed - 700) / 400.0) ** 2
    local_min_1 = 0.3 * np.exp(-5 * local_min_1_dist)
    
    # Local minimum 2: low power, high speed
    local_min_2_dist = ((laser_power - 200) / 125.0) ** 2 + ((scan_speed - 1300) / 400.0) ** 2
    local_min_2 = 0.35 * np.exp(-5 * local_min_2_dist)
    
    # Combine all effects
    porosity = base_porosity + interactions + local_min_1 + local_min_2 + noise_level * np.random.randn(n_samples)
    
    # Clip to realistic range
    porosity = np.clip(porosity, 0.1, 5.0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'laser_power_W': laser_power,
        'scan_speed_mm_s': scan_speed,
        'hatch_spacing_mm': hatch_spacing,
        'layer_thickness_um': layer_thickness,
        'energy_density_J_mm3': energy_density,
        'porosity_pct': porosity
    })
    
    logger.info(f"  Generated {n_samples} samples")
    logger.info(f"  Porosity range: {porosity.min():.3f} - {porosity.max():.3f} %")
    logger.info(f"  Mean: {porosity.mean():.3f} %, Std: {porosity.std():.3f} %")
    logger.info(f"  Target: minimize to < 0.5%")
    logger.info(f"  Global optimum: Power=275W, Speed=950mm/s, Hatch=0.105mm, Layer=33um")
    logger.info(f"  Minimum observed: {porosity.min():.3f}%")
    
    return df


def main():
    """Generate all datasets for Lecture 2."""
    logger.info("="*70)
    logger.info("GENERATING DATASETS FOR LECTURE 2")
    logger.info("="*70)
    
    # Create data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # Generate datasets
    logger.info("\nDataset 1: SLM Surface Roughness (Regression Comparison)")
    df1 = generate_slm_surface_roughness(n_samples=150)
    path1 = data_dir / 'slm_surface_roughness.csv'
    df1.to_csv(path1, index=False)
    logger.info(f"Saved to: {path1}\n")
    
    logger.info("Dataset 2: Print Adhesion Strength (Gaussian Process Demo)")
    df2 = generate_adhesion_strength(n_samples=30)
    path2 = data_dir / 'print_adhesion_strength.csv'
    df2.to_csv(path2, index=False)
    logger.info(f"Saved to: {path2}\n")
    
    logger.info("Dataset 3: SLM Porosity (Bayesian Optimization)")
    df3 = generate_porosity_data_for_bo(n_samples=200)
    path3 = data_dir / 'slm_porosity_optimization.csv'
    df3.to_csv(path3, index=False)
    logger.info(f"Saved to: {path3}\n")
    
    # Summary
    logger.info("="*70)
    logger.info("DATASET GENERATION COMPLETE")
    logger.info("="*70)
    logger.info(f"\nGenerated {len([df1, df2, df3])} datasets:")
    logger.info(f"  1. {path1.name} ({len(df1)} samples)")
    logger.info(f"  2. {path2.name} ({len(df2)} samples)")
    logger.info(f"  3. {path3.name} ({len(df3)} samples)")
    logger.info(f"\nAll files saved to: {data_dir.absolute()}")
    logger.info("\nReady to run demo scripts!")
    logger.info("="*70)


if __name__ == '__main__':
    main()
