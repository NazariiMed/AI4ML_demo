# Lecture 3: Advanced Applications and Implementation Strategies

## Overview

This lecture covers advanced ML/AI applications in additive manufacturing and practical implementation strategies. Students will learn about multi-objective optimization, deep learning for defect detection, real-time process control, and real-world implementation challenges.

## Learning Objectives

By the end of this lecture, students will be able to:
1. Formulate and solve multi-objective optimization problems with conflicting goals
2. Apply deep learning (CNNs) for image-based defect detection
3. Design adaptive control systems for real-time process monitoring
4. Identify and address common implementation challenges
5. Learn from case studies of successful and failed implementations

## Topics Covered

### 1. Advanced Optimization Scenarios
- Multi-objective Bayesian optimization (MOBO)
- Pareto frontier concept and selection strategies
- Multi-material and multi-process optimization
- Handling process drift and equipment variation
- Domain adaptation techniques

### 2. Deep Learning in Additive Manufacturing
- Neural network architectures (CNN, LSTM, U-Net)
- Transfer learning for small datasets
- Convolutional Neural Networks for defect detection
- Semantic segmentation for process monitoring
- Time-series processing with LSTMs

### 3. Real-time Process Control
- In-situ monitoring technologies
- Control strategies: PID, MPC, Reinforcement Learning
- Feedback mechanisms and latency considerations
- Edge computing for real-time inference
- Adaptive layer height control case study

### 4. Implementation Challenges and Solutions
- Data quality issues and automated validation
- Data availability and synthetic data generation
- Sim-to-real gap and domain adaptation
- Computational requirements and model compression
- Integration with existing manufacturing systems
- Organizational and change management aspects

## Demonstrations

### Demo 1: Multi-Objective Bayesian Optimization
**File**: `01_multi_objective_optimization.py`

Demonstrates MOBO for SLM process optimization with three conflicting objectives:
- Minimize porosity (quality)
- Maximize build rate (speed)
- Minimize energy consumption (cost)

**Key Features**:
- Expected Hypervolume Improvement (EHVI) acquisition
- Pareto frontier visualization
- Solution selection strategies
- Trade-off analysis

**Run**:
```bash
python 01_multi_objective_optimization.py
```

**Expected Output**:
- Pareto frontier plots (2D and 3D)
- Hypervolume convergence
- Parallel coordinates visualization
- Table of Pareto-optimal solutions

---

### Demo 2: CNN for Defect Detection
**File**: `02_cnn_defect_detection.py`

Demonstrates transfer learning with ResNet-18 for classifying AM layer images into three categories:
- Normal (no defects)
- Porosity (gas pores)
- Crack (crack formation)

**Key Features**:
- Synthetic layer image generation
- Transfer learning from ImageNet
- Data augmentation
- Confusion matrix and ROC curves
- Training curve visualization

**Requirements**:
- PyTorch
- torchvision
- OpenCV (cv2)

**Run**:
```bash
python 02_cnn_defect_detection.py
```

**Expected Output**:
- Training curves (loss and accuracy)
- Confusion matrix
- Per-class accuracy
- ROC curves
- Example predictions

**Note**: Training on CPU takes ~10-15 minutes. GPU recommended for faster training.

---

### Demo 3: Adaptive Process Control
**File**: `03_adaptive_process_control.py`

Compares four control strategies for maintaining constant melt pool width:
1. Open-loop (baseline, no feedback)
2. PID control (classical feedback)
3. Model Predictive Control (MPC)
4. Reinforcement Learning (simplified Q-learning)

**Key Features**:
- Simplified SLM process simulator
- Realistic disturbances and noise
- Performance metrics (RMSE, MAE, defect rate)
- Control effort analysis

**Run**:
```bash
python 03_adaptive_process_control.py
```

**Expected Output**:
- Width tracking comparison
- Power command plots
- Error distributions
- Performance metrics table

---

## Installation

### Core Dependencies
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
```

### For Demo 2 (CNN) - Additional Requirements
```bash
pip install torch torchvision opencv-python Pillow
```

### Optional (for GPU acceleration in Demo 2)
- CUDA-enabled GPU
- PyTorch with CUDA support

## Directory Structure

```
lecture_03/
├── README.md                          # This file
├── 01_multi_objective_optimization.py # Demo 1 script
├── 02_cnn_defect_detection.py        # Demo 2 script
├── 03_adaptive_process_control.py    # Demo 3 script
├── data/                              # (empty, demos generate synthetic data)
└── outputs/                           # Generated visualizations
    ├── 01_multi_objective_optimization/
    ├── 02_cnn_defect_detection/
    └── 03_adaptive_process_control/
```

## Key Concepts

### Multi-Objective Optimization
- **Pareto Optimality**: Solution where you can't improve one objective without worsening another
- **Pareto Frontier**: Set of all Pareto-optimal solutions
- **EHVI**: Expected Hypervolume Improvement - measures quality of Pareto front
- **Selection Strategies**: Weighted sum, knee point, business constraints

### Deep Learning
- **Transfer Learning**: Reuse pre-trained models (ImageNet → AM defects)
- **Data Augmentation**: Rotation, flip, brightness → prevent overfitting
- **CNN**: Convolutional layers learn hierarchical features (edges → textures → objects)
- **Fine-tuning**: Adapt pre-trained model to new task with small dataset

### Control Systems
- **PID**: Simple, fast, tunable - industry standard
- **MPC**: Predictive, handles constraints - computationally expensive
- **RL**: Learns from experience - requires extensive training
- **Latency Budget**: Total loop time must be <10-50ms for real-time control

### Implementation Challenges
- **Data Quality**: Missing values, outliers, label noise
- **Data Availability**: Expensive experiments → synthetic data, transfer learning
- **Sim-to-Real Gap**: Domain adaptation, style transfer, fine-tuning
- **Computational**: Model compression (quantization, pruning, distillation)
- **Integration**: Legacy systems, communication protocols, organizational resistance

## Further Reading

### Papers
- Goh et al., "A Review on Machine Learning in 3D Printing" (2021)
- Qi et al., "Applying Neural-Network-Based Machine Learning to AM" (2019)
- Johnson et al., "ML for Materials Developments in Metals AM" (2021)
- Zhang et al., "Adaptive Layer Height Control" (2023)

### Books
- Goodfellow et al., "Deep Learning" (2016)
- Sutton & Barto, "Reinforcement Learning: An Introduction" (2018)
- Skogestad & Postlethwaite, "Multivariable Feedback Control" (2005)

### Online Courses
- Fast.ai for practical deep learning
- Stanford CS231n for CNNs
- DeepMind's RL course

### Software & Tools
- **PyTorch / TensorFlow**: Deep learning frameworks
- **Ax / BoTorch**: Multi-objective Bayesian optimization
- **OpenCV**: Computer vision and image processing
- **ONNX**: Model deployment and optimization
- **TensorFlow Lite**: Edge deployment

### Datasets
- NIST Additive Manufacturing Benchmark (AMB) series
- Open Additive Manufacturing Database (OpenAMD)

## Troubleshooting

### Demo 1 Issues
- **"differential_evolution failed"**: Increase maxiter or change seed
- **"Pareto front too small"**: Increase n_init or n_iter

### Demo 2 Issues
- **"Out of memory"**: Reduce batch_size or image_size
- **"Training too slow"**: Use GPU or reduce epochs
- **"Import error: cv2"**: Install opencv-python

### Demo 3 Issues
- **"MPC optimization slow"**: Reduce horizon or use coarser time step
- **"Unstable control"**: Adjust PID gains or reduce disturbances

## Contact

For questions or issues:
**Instructor:** Nazarii Mediukh, PhD  
**Email:** n.mediukh@ipms.kyiv.ua  
**Institution:** Institute for Problems of Materials Science, NASU

## License

Course materials are for educational use only.

---

**Previous**: [Lecture 2 - Bayesian Optimization](../lecture_02/README.md)
