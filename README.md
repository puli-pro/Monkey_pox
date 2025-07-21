---

# 🦠 **Monkeypox Classification & Detection System**

**An Advanced Machine Learning Pipeline for Clinical Diagnosis Using Symptom Analysis**

---

## 📊 Project Overview

The **Monkeypox Classification & Detection System** is a comprehensive machine learning pipeline designed to assist healthcare professionals in the early detection and diagnosis of Monkeypox based on clinical symptoms and patient data. This system integrates various traditional machine learning algorithms and deep neural networks, leveraging advanced feature selection techniques and hyperparameter optimization to achieve optimal diagnostic accuracy.

**Key Innovation**: The system employs a multi-algorithm comparison strategy, combining models such as Random Forest, XGBoost, and Neural Networks, and integrates advanced methodologies like **feature selection**, **hyperparameter optimization**, and **cross-validation** to improve diagnostic precision.

---

## 🎯 Core Features

| **Feature Category**            | **Description**                                                                                         | **Implementation** |
| ------------------------------- | ------------------------------------------------------------------------------------------------------- | ------------------ |
| **Multi-Algorithm Comparison**  | 10+ ML algorithms including Random Forest, SVM, XGBoost, LightGBM. Comprehensive baseline establishment | ✅                  |
| **Deep Learning Integration**   | Neural networks with architecture search and hyperparameter tuning                                      | ✅                  |
| **Feature Selection Suite**     | Multiple feature selection algorithms for identifying key features                                      | ✅                  |
| **Clinical Validation**         | Cross-validation with medical performance metrics (e.g., sensitivity, specificity)                      | ✅                  |
| **Performance Analytics**       | ROC curves, confusion matrices, precision-recall analysis                                               | ✅                  |
| **Hyperparameter Optimization** | Automated tuning with Optuna and Keras Tuner. Grid and Bayesian optimization                            | ✅                  |

---

## 📁 Repository Structure

```
Monkey_pox/
├── 📊 Data Files
│   ├── DATA.csv                 # Primary dataset (25,000 records)
│   ├── Monkeypox.xlsx           # Original data format
│   ├── Monkeypox-checkpoint.csv # Data backup
│   └── Research Papers.xlsx     # Literature references
├── 🧠 Core Analysis Notebooks
│   ├── ANN.ipynb                # Neural Network Implementation
│   ├── Classification_algorithm.ipynb # ML Algorithms Comparison
│   ├── FS_algorithms.ipynb      # Feature Selection Methods
│   └── Hyperparameter_tuning.ipynb  # Parameter Optimization
├── 🔬 Specialized Analyses
│   ├── Monkeypox_ML_code.ipynb  # Primary ML pipeline
│   ├── ONLY_FS.ipynb            # Standalone feature selection
│   ├── RAMA (F).ipynb           # Advanced analysis methods
│   └── Krishna.ipynb            # Experimental implementations
├── 📂 Model Artifacts
│   ├── ann_tuner/ann_tuning/    # Neural network tuning results
│   ├── my_dir/                  # Hyperparameter search logs
│   ├── optuna_gb_tuning.log     # Gradient boosting optimization
│   └── optuna_tuning.log        # General optimization logs
└── 🗃️ Checkpoints & Backups
    └── .ipynb_checkpoints/      # Notebook versions
```

---

## 🚀 Getting Started

### Prerequisites

To run this project, ensure you have the following installed:

* Python >= 3.8
* Jupyter Notebook
* pandas >= 1.3.0
* numpy >= 1.21.0
* scikit-learn >= 1.0.0
* tensorflow >= 2.6.0
* xgboost >= 1.5.0
* lightgbm >= 3.3.0
* optuna >= 2.10.0
* keras-tuner >= 1.1.0
* matplotlib >= 3.5.0
* seaborn >= 0.11.0

### Installation & Setup

To get started, follow these steps:

```bash
# 1. Clone the repository
git clone https://github.com/puli-pro/Monkey_pox.git
cd Monkey_pox

# 2. Install dependencies
pip install pandas numpy scikit-learn tensorflow
pip install xgboost lightgbm optuna keras-tuner
pip install matplotlib seaborn jupyter

# 3. Launch Jupyter Notebook
jupyter notebook
```

---

## 🔍 Dataset Information

### **Dataset Characteristics**:

* **Size**: 25,000 patient records
* **Features**: 14 clinical symptoms and patient characteristics
* **Target Distribution**:

  * 63.64% Positive cases
  * 36.36% Negative cases
* **Data Quality**: Preprocessed with missing value handling and encoding.

### **Key Features**:

* **Patient Demographics**: Age, gender, and medical history.
* **Clinical Symptoms**: Fever, lesions, pain, etc.
* **Systemic Illness**: Fever, lymph nodes, muscle aches.
* **Binary Indicators**: Sore throat, HIV infection, STI history.

---

## 🧪 Usage Examples

### **Quick Start - Basic Classification**:

```python
# Load and explore the dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
data = pd.read_csv('DATA.csv')

# Prepare features and target
X = data.drop('MonkeyPox', axis=1)
y = data['MonkeyPox']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate performance
accuracy = rf_model.score(X_test, y_test)
print(f"Random Forest Accuracy: {accuracy:.4f}")
```

### **Advanced Neural Network Analysis**:

For deep learning approaches, open the **ANN.ipynb** notebook, where you can explore the entire pipeline, including hyperparameter tuning, neural network architecture search, and performance validation with multiple metrics.

### **Feature Selection Pipeline**:

```python
# Run feature selection algorithms
# Open FS_algorithms.ipynb for the complete workflow
```

---

## 📈 Model Performance

| **Algorithm**                 | **Accuracy** | **Precision** | **Recall** | **F1-Score** | **AUC-ROC** |
| ----------------------------- | ------------ | ------------- | ---------- | ------------ | ----------- |
| **LightGBM**                  | 70.18%       | 71.90%        | 88.76%     | 79.44%       | 69.34%      |
| **Gradient Boosting**         | 69.09%       | 70.55%        | 88.29%     | 78.43%       | 69.62%      |
| **Artificial Neural Network** | 70.34%       | 71.05%        | 90.97%     | 79.69%       | 69.97%      |
| **XGBoost**                   | 67.20%       | 70.11%        | 84.47%     | 76.62%       | 66.45%      |
| **AdaBoost**                  | 69.08%       | 71.00%        | 86.91%     | 78.16%       | 69.82%      |

### **Key Insights**:

* **Neural Networks** performed best with an accuracy of **70.34%**.
* **Feature Selection** improved the performance across all models.
* All models achieved **high recall (>84%)**, ensuring minimal false negatives, which is critical for disease screening.

---

## 🔧 Advanced Features

### **Hyperparameter Optimization**:

* **Keras Tuner**: Neural network architecture search.
* **Optuna**: Bayesian optimization for traditional ML algorithms.
* **Grid Search & Random Search**: Parameter tuning for optimization.

### **Feature Selection Methods**:

* **Statistical Tests**: Chi-square, ANOVA, F-test.
* **Recursive Feature Elimination**: Iterative ranking of features.
* **Permutation Importance**: Assessment of feature impact.
* **Correlation Analysis**: Detection of multicollinearity.

### **Model Validation**:

* **Stratified K-Fold**: Balanced cross-validation.
* **ROC Curve Analysis**: Threshold optimization for clinical workflows.
* **Confusion Matrix**: Detailed error analysis with clinical metrics.

---

## 🏥 Clinical Applications

### **Target Use Cases**:

* **Primary Screening**: Rapid initial assessment in clinical settings.
* **Epidemic Monitoring**: Population-level surveillance.
* **Resource Allocation**: Prioritizing high-risk patients for testing.
* **Decision Support**: Assisting healthcare providers in diagnosis.

### **Clinical Metrics Focus**:

* **High Recall (>84%)**: Minimizing false negatives.
* **Balanced Precision**: Reducing unnecessary confirmatory tests.
* **ROC Optimization**: Custom threshold tuning for clinical workflows.

---

## 🤝 Contributing

We welcome contributions to improve the diagnostic accuracy and clinical utility of this system:

### **Development Guidelines**:

* Fork the repository and create a feature branch.
* Follow **PEP 8** coding standards and add **docstrings**.
* Test your implementations with the provided dataset.
* Submit pull requests with clear descriptions of changes
