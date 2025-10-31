# Machine Learning for Additive Manufacturing Course

A comprehensive 3-lecture course on applying artificial intelligence and machine learning techniques to additive manufacturing processes.

## 📖 Course Overview

**Duration:** 3 lectures × 1 hour each  
**Format:** Online with live demonstrations  
**Level:** Bachelor/Master students, engineers, researchers  
**ECTS Credits:** 0.5 (includes 15 hours independent study)

### Course Objectives

- Master ML fundamentals in manufacturing contexts
- Apply AI/ML across the AM workflow
- Understand technical challenges and opportunities
- Analyze real-world case studies
- Develop practical implementation skills

## 🗂️ Course Structure

### Lecture 1: Fundamentals of Machine Learning for Additive Manufacturing
**Topics:**
- Introduction to AI/ML in Manufacturing
- Supervised, Unsupervised, and Reinforcement Learning
- Data Collection and Quality in AM
- Machine Learning Pipeline for Manufacturing

**Demos:** Surface roughness prediction, defect clustering, data preprocessing, complete ML pipeline

📁 **[Go to Lecture 1 →](lecture_01/)**

---

### Lecture 2: Process Parameter Optimization in Additive Manufacturing
**Topics:**
- Fundamentals of Process Parameter Optimization
- Regression Techniques for AM
- Bayesian Optimization
- Multi-fidelity Optimization Approaches

**Demos:** Regression comparison, Gaussian process, Bayesian optimization, Multi-fidelity demo.
📁 **[Go to Lecture 2 →](lecture_02/)**

---

### Lecture 3: Advanced Applications and Implementation Strategies
**Topics:**
- Multi-objective Optimization
- Deep Learning in AM
- Real-time Process Control
- Implementation Challenges and Solutions

**Demos:** Multi-objective optimization, CNN defect detection, Adaptive process control
📁 **[Go to Lecture 3 →](lecture_03/)**

---

## 🚀 Quick Start

### Prerequisites

- Basic Python programming knowledge
- Understanding of manufacturing processes (helpful but not required)
- Familiarity with calculus and linear algebra (for deeper understanding)

### Installation

1. **Clone the repository:**

2. **Set up virtual environment or activate:**
```bash
source .venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Generate datasets and run demos:**
```bash
cd lecture_01
python generate_data.py
python 01_supervised_learning_regression.py
```

## 📦 Repository Structure

```
ml4am/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .venv/                            # Virtual environment (gitignored)
│
├── lecture_01/                       # Lecture 1 materials
│   ├── README.md                     # Lecture 1 guide
│   ├── LECTURE_01_SLIDES.md          # Slide content
│   ├── generate_data.py              # Dataset generator
│   ├── 01_supervised_learning_regression.py
│   ├── 02_unsupervised_clustering.py
│   ├── 03_data_preprocessing.py
│   ├── 04_complete_pipeline.py
│   ├── data/                         # Generated datasets
│   └── outputs/                      # Visualizations
│
├── lecture_02/                       # Same as lecture 01
│   └── ...
│
└── lecture_03/                       # Same as lecture 01
    └── ...
```

## 🎓 Learning Approach

This course combines:

1. **Theoretical Foundations** - Understanding ML concepts and algorithms
2. **Practical Demonstrations** - Live code execution with detailed logging
3. **Domain Application** - AM-specific examples and use cases
4. **Interactive Discussions** - Q&A and problem-solving sessions

## 📊 Technologies Used

**Core Libraries:**
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning algorithms
- `matplotlib` & `seaborn` - Visualization
- `scipy` - Scientific computing
- `torch` - NeuralNetworks
- `opencv` - image processing

## 🎯 Learning Outcomes

By the end of this course, students will be able to:

### Knowledge
- Explain ML fundamentals and their relevance to AM
- Identify appropriate ML techniques for different AM problems
- Understand the AM data pipeline from collection to deployment

### Skills
- Implement ML models for AM process optimization
- Preprocess and engineer features from AM sensor data
- Validate models properly and avoid overfitting
- Interpret results and communicate findings

### Competencies
- Design ML solutions for real AM problems
- Critically evaluate ML approaches in literature
- Implement end-to-end ML pipelines in Python
- Make data-driven decisions in AM process development

## 👨‍🏫 Instructor

**Nazarii Mediukh, PhD**  
Junior Research Scientist  
Institute for Problems of Materials Science  
National Academy of Sciences of Ukraine

**Contact:**
- Email: n.mediukh@ipms.kyiv.ua

**Research Focus:**
- Computational materials science
- AI/ML integration in materials research
- Additive manufacturing optimization
- Aluminum matrix composites

## 🎓 Target Audience

This course is designed for:

- **Materials and Manufacturing Engineers** seeking to apply ML in their work
- **Data Scientists** interested in manufacturing applications
- **Graduate Students** in materials science, mechanical engineering, or related fields
- **Industry Professionals** working in additive manufacturing
- **Researchers** exploring AI/ML for process optimization

## 📚 Recommended Prerequisites

### Essential
- Basic programming (Python preferred)
- Fundamental statistics (mean, variance, correlation)
- Basic calculus (derivatives, optimization concepts)

### Helpful
- Linear algebra (matrices, vectors)
- Manufacturing process knowledge
- Previous exposure to machine learning (any level)

## 🤝 Contributing

This is an educational repository. Suggestions for improvements are welcome:

1. Open an issue describing the proposed change
2. For code contributions, create a pull request
3. For content suggestions, email the instructor

## 📄 License

This course material is provided for educational purposes. Students and educators may use and modify for learning and teaching.

**Copyright © 2025 Nazarii Mediukh, Institute for Problems of Materials Science, NASU**

---

**Ready to start?** Head to [Lecture 1](lecture_01/) to begin your ML for AM journey! 🚀
