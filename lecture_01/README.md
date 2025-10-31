# Lecture 1: Fundamentals of Machine Learning for Additive Manufacturing

This directory contains all materials for the first lecture of the ML for AM course.

## 📋 Contents

### Code Demonstrations
1. **`generate_data.py`** - Generate synthetic datasets for all demos
2. **`01_supervised_learning_regression.py`** - Supervised learning demo (surface roughness prediction)
3. **`02_unsupervised_clustering.py`** - Unsupervised learning demo (defect pattern discovery)
4. **`03_data_preprocessing.py`** - Data preprocessing techniques (time-series sensor data)
5. **`04_complete_pipeline.py`** - End-to-end ML pipeline (comprehensive workflow)

### Generated Outputs
- **`data/`** - Synthetic datasets (created by `generate_data.py`)
- **`outputs/`** - Visualizations and results from running demos

## 🚀 Getting Started

### 1. Install Dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

### 2. Generate Datasets

```bash
cd lecture_01
python generate_data.py
```

This creates three datasets in `data/`:
- `fdm_surface_roughness.csv` - FDM process parameters → surface roughness (200 samples)
- `defect_clustering.csv` - AM defect characteristics (150 samples, 3 clusters)
- `time_series_sensor.csv` - Thermal sensor data during printing (1000 timesteps)

### 3. Run Demonstrations

Run demos in order during the lecture:

```bash
# Demo 1: Supervised Learning (after Slide 8, ~15 min)
python 01_supervised_learning_regression.py

# Demo 2: Unsupervised Learning (after Slide 10, ~25 min)
python 02_unsupervised_clustering.py

# Demo 3: Data Preprocessing (after Slide 15, ~40 min)
python 03_data_preprocessing.py

# Demo 4: Complete Pipeline (after Slide 19, ~52 min)
python 04_complete_pipeline.py
```

Each script produces:
- Detailed console logging showing the ML workflow
- Multiple visualization files saved to `outputs/`

## 📊 Demo Overview

### Demo 1: Supervised Learning Regression
**Learning Goals:**
- Understand supervised learning workflow
- Compare regression algorithms (Linear, Tree, Forest)
- Interpret performance metrics (R², MAE, RMSE)
- Visualize predictions vs actual values

**Key Outputs:**
- Data relationship plots
- Model comparison charts
- Prediction accuracy visualizations

---

### Demo 2: Unsupervised Clustering
**Learning Goals:**
- Discover patterns without labels
- Apply K-means clustering
- Evaluate clustering quality (Silhouette score)
- Interpret cluster characteristics

**Key Outputs:**
- Elbow method plot
- PCA visualization of clusters
- Cluster profile analysis

---

### Demo 3: Data Preprocessing
**Learning Goals:**
- Handle time-series sensor data
- Detect and remove outliers
- Apply smoothing techniques
- Extract statistical features
- Compare scaling methods

**Key Outputs:**
- Raw data visualizations
- Smoothing comparison plots
- Scaling effects
- Feature summary dashboard

---

### Demo 4: Complete ML Pipeline
**Learning Goals:**
- See the full workflow from data to deployment
- Practice cross-validation
- Perform hyperparameter tuning
- Evaluate and save final model

**Key Outputs:**
- Comprehensive results dashboard
- Overfitting analysis
- Trained model files (`.joblib`)

## 🎯 Learning Outcomes

By the end of this lecture, students will be able to:

1. ✅ Distinguish between supervised, unsupervised, and reinforcement learning
2. ✅ Identify appropriate ML algorithms for AM problems
3. ✅ Understand data quality challenges in AM
4. ✅ Apply essential preprocessing techniques
5. ✅ Follow a systematic ML pipeline
6. ✅ Avoid common pitfalls (overfitting, data leakage)

## 🔧 Troubleshooting

### Issue: "Module not found" error
**Solution:** Ensure virtual environment is activated and requirements installed:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Issue: Plots not displaying
**Solution:** Scripts save plots to `outputs/` directory. Check there for PNG files.

### Issue: Data files not found
**Solution:** Run `python generate_data.py` first to create datasets.

## 📧 Contact

**Instructor:** Nazarii Mediukh, PhD  
**Email:** n.mediukh@ipms.kyiv.ua  
**Institution:** Institute for Problems of Materials Science, NASU

---

**Next:** Proceed to Lecture 2 (Process Parameter Optimization in AM) after completing this lecture.
