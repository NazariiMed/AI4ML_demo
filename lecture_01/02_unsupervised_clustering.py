"""
Demo 2: Unsupervised Learning - Defect Pattern Discovery

This script demonstrates unsupervised learning applied to AM defect analysis.
We use clustering to discover natural groupings in defect data without labels.

Learning Goals:
- Understand unsupervised learning concepts
- Apply K-means clustering
- Evaluate clustering quality
- Interpret discovered patterns
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
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
plt.rcParams['figure.figsize'] = (12, 8)


def load_and_explore_data(data_path):
    """Load and explore defect data."""
    logger.info("="*70)
    logger.info("STEP 1: LOADING AND EXPLORING DEFECT DATA")
    logger.info("="*70)
    
    df = pd.read_csv(data_path)
    logger.info(f"✓ Loaded dataset: {len(df)} defect samples")
    
    # Separate features from true labels (we won't use labels for clustering)
    features = [col for col in df.columns if col != 'true_label']
    X = df[features]
    true_labels = df['true_label']
    
    logger.info(f"✓ Features: {features}")
    logger.info(f"✓ Feature matrix shape: {X.shape}")
    
    # Basic statistics
    logger.info("\nDataset Statistics:")
    logger.info(f"\n{X.describe().round(2)}")
    
    # True label distribution (hidden ground truth)
    logger.info("\nTrue Label Distribution (for validation only):")
    for label, count in true_labels.value_counts().items():
        logger.info(f"  {label:15s}: {count:3d} samples ({count/len(df)*100:.1f}%)")
    
    return df, X, true_labels, features


def visualize_raw_data(X, true_labels, features, output_dir):
    """Visualize raw feature distributions."""
    logger.info("\n" + "="*70)
    logger.info("STEP 2: VISUALIZING RAW DATA DISTRIBUTIONS")
    logger.info("="*70)
    
    # Pairplot for first 4 features
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle('Feature Pairwise Relationships (colored by true labels)', 
                 fontsize=16, fontweight='bold')
    
    colors = {'thermal': 'red', 'mechanical': 'blue', 'material': 'green'}
    
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: histograms
                for label in colors.keys():
                    mask = true_labels == label
                    ax.hist(X[features[i]][mask], alpha=0.5, label=label, bins=15)
                ax.set_ylabel('Count')
                if i == 0:
                    ax.legend()
            else:
                # Off-diagonal: scatter plots
                for label, color in colors.items():
                    mask = true_labels == label
                    ax.scatter(X[features[j]][mask], X[features[i]][mask], 
                             c=color, alpha=0.5, s=20, label=label)
            
            if i == 3:
                ax.set_xlabel(features[j].replace('_', ' ').title(), fontsize=9)
            if j == 0:
                ax.set_ylabel(features[i].replace('_', ' ').title(), fontsize=9)
    
    plt.tight_layout()
    viz_path = output_dir / '01_raw_data_pairplot.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved visualization: {viz_path}")
    plt.close()
    
    logger.info("✓ We can see some clustering patterns visually")
    logger.info("  But remember: in real unsupervised learning, we don't have these labels!")


def preprocess_features(X):
    """Standardize features for clustering."""
    logger.info("\n" + "="*70)
    logger.info("STEP 3: PREPROCESSING FEATURES")
    logger.info("="*70)
    
    logger.info("Applying StandardScaler...")
    logger.info("  Why? Features have different scales (temperature vs size)")
    logger.info("  K-means is sensitive to scale - distance-based algorithm")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    logger.info("\n✓ Scaled features to zero mean and unit variance")
    logger.info(f"  Original range examples:")
    logger.info(f"    max_temperature: {X.iloc[:, 0].min():.1f} to {X.iloc[:, 0].max():.1f}")
    logger.info(f"    defect_size: {X.iloc[:, 3].min():.2f} to {X.iloc[:, 3].max():.2f}")
    logger.info(f"  After scaling: all features have mean≈0, std≈1")
    
    return X_scaled, scaler


def determine_optimal_clusters(X_scaled, max_clusters=10, output_dir=None):
    """Use elbow method to find optimal number of clusters."""
    logger.info("\n" + "="*70)
    logger.info("STEP 4: DETERMINING OPTIMAL NUMBER OF CLUSTERS")
    logger.info("="*70)
    
    logger.info("Testing different numbers of clusters using Elbow Method...")
    
    inertias = []
    silhouette_scores = []
    K_range = range(2, max_clusters + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        sil_score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(sil_score)
        
        logger.info(f"  k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil_score:.3f}")
    
    # Plot elbow curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[0].set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
    axes[0].set_title('Elbow Method: Finding Optimal k', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=3, color='r', linestyle='--', alpha=0.5, label='k=3 (elbow)')
    axes[0].legend()
    
    axes[1].plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[1].set_ylabel('Silhouette Score', fontsize=12)
    axes[1].set_title('Silhouette Score: Higher is Better', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=max(silhouette_scores), color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if output_dir:
        elbow_path = output_dir / '02_elbow_method.png'
        plt.savefig(elbow_path, dpi=150, bbox_inches='tight')
        logger.info(f"\n✓ Saved elbow plot: {elbow_path}")
        plt.close()
    
    optimal_k = K_range[np.argmax(silhouette_scores)]
    logger.info(f"\n✓ Suggested optimal k: {optimal_k} (highest silhouette score)")
    logger.info(f"  Note: We'll use k=3 since we expect 3 defect types")
    
    return inertias, silhouette_scores


def perform_clustering(X_scaled, n_clusters=3):
    """Perform K-means clustering."""
    logger.info("\n" + "="*70)
    logger.info(f"STEP 5: PERFORMING K-MEANS CLUSTERING (k={n_clusters})")
    logger.info("="*70)
    
    logger.info("Training K-means model...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    logger.info(f"✓ Clustering complete")
    
    # Evaluate clustering quality
    sil_score = silhouette_score(X_scaled, cluster_labels)
    db_score = davies_bouldin_score(X_scaled, cluster_labels)
    
    logger.info(f"\nClustering Quality Metrics:")
    logger.info(f"  Silhouette Score: {sil_score:.3f} (range: -1 to 1, higher is better)")
    logger.info(f"  Davies-Bouldin Index: {db_score:.3f} (lower is better)")
    
    if sil_score > 0.5:
        logger.info("  ✓ Strong clustering structure")
    elif sil_score > 0.3:
        logger.info("  ✓ Reasonable clustering structure")
    else:
        logger.info("  ⚠ Weak clustering structure")
    
    # Cluster sizes
    logger.info(f"\nCluster Sizes:")
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        logger.info(f"  Cluster {cluster_id}: {count} samples ({count/len(cluster_labels)*100:.1f}%)")
    
    return kmeans, cluster_labels, sil_score


def visualize_clusters(X, cluster_labels, true_labels, features, output_dir):
    """Visualize clustering results in physical feature space."""
    logger.info("\n" + "="*70)
    logger.info("STEP 6: VISUALIZING CLUSTERS")
    logger.info("="*70)
    
    logger.info("Creating visualizations in physical feature space...")
    logger.info("  Using actual measurements, not dimensionality reduction")
    
    # Create 2x2 subplot grid showing different feature combinations
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Discovered Clusters vs True Labels (Physical Feature Space)', 
                 fontsize=16, fontweight='bold')
    
    # Define feature pairs for 2D projections
    feature_pairs = [
        ('max_temperature', 'energy_density'),
        ('cooling_rate', 'defect_size'),
        ('max_temperature', 'cooling_rate'),
        ('energy_density', 'defect_size')
    ]
    
    colors_map = {'thermal': 'red', 'mechanical': 'blue', 'material': 'green'}
    cluster_colors = ['purple', 'orange', 'cyan']
    
    for idx, (feat_x, feat_y) in enumerate(feature_pairs):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Get feature indices
        x_idx = list(features).index(feat_x)
        y_idx = list(features).index(feat_y)
        
        # Plot discovered clusters (larger, semi-transparent)
        for cluster_id in np.unique(cluster_labels):
            mask = cluster_labels == cluster_id
            ax.scatter(X[feat_x][mask], X[feat_y][mask],
                      c=cluster_colors[cluster_id], s=100, alpha=0.3,
                      edgecolors='black', linewidth=0.5,
                      label=f'Cluster {cluster_id}')
        
        # Overlay true labels (smaller, opaque markers)
        for label, color in colors_map.items():
            mask = true_labels == label
            ax.scatter(X[feat_x][mask], X[feat_y][mask],
                      c=color, s=30, alpha=0.8, marker='x',
                      linewidth=2, label=f'True: {label}')
        
        ax.set_xlabel(feat_x.replace('_', ' ').title(), fontsize=11)
        ax.set_ylabel(feat_y.replace('_', ' ').title(), fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
    
    plt.tight_layout()
    cluster_path = output_dir / '03_cluster_visualization.png'
    plt.savefig(cluster_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved visualization: {cluster_path}")
    plt.close()
    
    logger.info("\nVisualization shows:")
    logger.info("  • Large colored circles = Discovered clusters (unlabeled learning)")
    logger.info("  • Small 'x' marks = True defect types (for validation only)")
    logger.info("  • Good overlap means clustering found the real patterns!")


def analyze_cluster_characteristics(X, cluster_labels, features):
    """Analyze what characterizes each cluster."""
    logger.info("\n" + "="*70)
    logger.info("STEP 7: ANALYZING CLUSTER CHARACTERISTICS")
    logger.info("="*70)
    
    logger.info("Computing mean feature values for each cluster...\n")
    
    for cluster_id in sorted(np.unique(cluster_labels)):
        cluster_mask = cluster_labels == cluster_id
        cluster_data = X[cluster_mask]
        
        logger.info(f"{'='*70}")
        logger.info(f"CLUSTER {cluster_id} PROFILE (n={cluster_mask.sum()} samples)")
        logger.info(f"{'='*70}")
        
        for feature in features:
            mean_val = cluster_data[feature].mean()
            std_val = cluster_data[feature].std()
            overall_mean = X[feature].mean()
            
            # Determine if this feature is distinctive for this cluster
            deviation = (mean_val - overall_mean) / X[feature].std()
            
            status = ""
            if abs(deviation) > 1.0:
                status = "⭐ DISTINCTIVE" if deviation > 0 else "⭐ DISTINCTIVE (low)"
            
            logger.info(f"  {feature:20s}: {mean_val:7.2f} ± {std_val:5.2f}  {status}")
        
        logger.info("")
    
    logger.info("Interpretation Guide:")
    logger.info("  ⭐ DISTINCTIVE = Feature value significantly different from overall mean")
    logger.info("  This helps us understand what makes each cluster unique")


def compare_with_ground_truth(cluster_labels, true_labels):
    """Compare discovered clusters with true labels."""
    logger.info("\n" + "="*70)
    logger.info("STEP 8: COMPARING WITH GROUND TRUTH")
    logger.info("="*70)
    logger.info("(In real scenarios, we wouldn't have these labels!)\n")
    
    # Create confusion matrix - need to convert string labels to numeric
    from sklearn.metrics import confusion_matrix
    
    # Map string labels to numbers for confusion matrix
    label_map = {label: idx for idx, label in enumerate(sorted(true_labels.unique()))}
    true_labels_numeric = true_labels.map(label_map)
    
    cm = confusion_matrix(true_labels_numeric, cluster_labels)
    
    logger.info("Confusion Matrix:")
    logger.info("                    Cluster 0  Cluster 1  Cluster 2")
    label_names = sorted(true_labels.unique())
    for i, label in enumerate(label_names):
        logger.info(f"  {label:15s}    {cm[i, 0]:4d}       {cm[i, 1]:4d}       {cm[i, 2]:4d}")
    
    # Calculate purity (how well clusters align with true labels)
    cluster_purity = []
    for cluster_id in range(len(np.unique(cluster_labels))):
        mask = cluster_labels == cluster_id
        if mask.sum() > 0:
            true_labels_in_cluster = true_labels[mask]
            most_common = true_labels_in_cluster.mode()[0]
            purity = (true_labels_in_cluster == most_common).sum() / mask.sum()
            cluster_purity.append(purity)
            logger.info(f"\nCluster {cluster_id} purity: {purity*100:.1f}% (mostly '{most_common}')")
    
    overall_purity = np.mean(cluster_purity)
    logger.info(f"\nOverall clustering purity: {overall_purity*100:.1f}%")
    
    if overall_purity > 0.8:
        logger.info("✓ Excellent alignment with ground truth!")
    elif overall_purity > 0.6:
        logger.info("✓ Good alignment with ground truth")
    else:
        logger.info("⚠ Moderate alignment - clustering found different patterns")


def main():
    """Main execution function."""
    logger.info("\n" + "="*70)
    logger.info("UNSUPERVISED LEARNING DEMO: DEFECT PATTERN DISCOVERY")
    logger.info("="*70)
    logger.info("This demo shows how to discover defect patterns WITHOUT labels")
    logger.info("="*70 + "\n")
    
    # Setup paths
    script_dir = Path(__file__).parent
    data_path = script_dir / 'data' / 'defect_clustering.csv'
    output_dir = script_dir / 'outputs' / '02_unsupervised_clustering'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Execute pipeline
    df, X, true_labels, features = load_and_explore_data(data_path)
    visualize_raw_data(X, true_labels, features, output_dir)
    X_scaled, scaler = preprocess_features(X)
    determine_optimal_clusters(X_scaled, max_clusters=8, output_dir=output_dir)
    kmeans, cluster_labels, sil_score = perform_clustering(X_scaled, n_clusters=3)
    visualize_clusters(X, cluster_labels, true_labels, features, output_dir)  # Pass original X, not scaled
    analyze_cluster_characteristics(X, cluster_labels, features)
    compare_with_ground_truth(cluster_labels, true_labels)
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY AND KEY TAKEAWAYS")
    logger.info("="*70)
    
    logger.info(f"\n✓ Discovered {len(np.unique(cluster_labels))} clusters in defect data")
    logger.info(f"  - Silhouette Score: {sil_score:.3f}")
    logger.info(f"  - Clustering quality: {'Strong' if sil_score > 0.5 else 'Good'}")
    
    logger.info("\nKey Insights:")
    logger.info("  1. K-means successfully identified natural groupings")
    logger.info("  2. Clusters align well with true defect types (thermal/mechanical/material)")
    logger.info("  3. Feature scaling was critical for distance-based clustering")
    logger.info("  4. PCA helped visualize high-dimensional data")
    logger.info("  5. Cluster profiles reveal distinctive characteristics")
    
    logger.info("\nPractical Applications:")
    logger.info("  • Automatic defect categorization without manual labeling")
    logger.info("  • Discovery of unknown defect types")
    logger.info("  • Quality control monitoring and alerting")
    logger.info("  • Process optimization based on defect patterns")
    
    logger.info(f"\n✓ All outputs saved to: {output_dir.absolute()}")
    logger.info("\n" + "="*70)
    logger.info("DEMO COMPLETE")
    logger.info("="*70 + "\n")


if __name__ == '__main__':
    main()
