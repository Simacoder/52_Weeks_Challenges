# IBM HR Employee Attrition Analysis

A comprehensive machine learning project that demonstrates the application of Logistic Regression on IBM's HR dataset to predict employee attrition, with emphasis on understanding why accuracy can be misleading in imbalanced datasets.

## 📋 Project Overview

This project analyzes employee attrition using the IBM HR Analytics Employee Attrition dataset. The main goal is to predict which employees are likely to leave the company and demonstrate important concepts about model evaluation metrics, particularly why accuracy alone can be misleading.

### Key Learning Objectives
- Apply Logistic Regression for binary classification
- Generate probability predictions for employee attrition
- Understand the impact of classification thresholds
- Demonstrate why accuracy is misleading with imbalanced data
- Explore ROC curves and AUC as better evaluation metrics

## 🎯 Problem Statement

**Business Question**: Can we predict which employees are likely to leave the company based on their demographics, job satisfaction, compensation, and work environment factors?

**Data Challenge**: The dataset is imbalanced (~16% attrition rate), making traditional accuracy metrics misleading.

## 📊 Dataset Information

**Source**: IBM HR Analytics Employee Attrition & Performance Dataset

**Dataset Characteristics**:
- **Rows**: 1,470 employees
- **Features**: 35 attributes including demographics, job factors, and satisfaction scores
- **Target**: Attrition (Yes/No)
- **Class Distribution**: ~84% stayed, ~16% left (imbalanced)

### Key Features
- **Demographics**: Age, Gender, Marital Status
- **Job Information**: Job Role, Department, Job Level, Years at Company
- **Compensation**: Monthly Income, Hourly Rate, Stock Options
- **Satisfaction Metrics**: Job Satisfaction, Work-Life Balance, Environment Satisfaction
- **Work Patterns**: Overtime, Business Travel, Distance from Home

## 🛠️ Technical Implementation

### Libraries Used
```python
pandas              # Data manipulation
numpy              # Numerical operations
scikit-learn       # Machine learning algorithms and metrics
matplotlib         # Data visualization
seaborn           # Statistical visualizations
```

### Machine Learning Pipeline
1. **Data Preprocessing**
   - Handle categorical variables with Label Encoding
   - Feature selection and engineering
   - Train-test split with stratification
   - Feature scaling with StandardScaler

2. **Model Training**
   - Logistic Regression with default parameters
   - Probability prediction generation

3. **Model Evaluation**
   - Confusion Matrix analysis
   - Classification Report (Precision, Recall, F1-score)
   - ROC Curve and AUC analysis
   - Threshold optimization

## 📈 Key Results & Insights

### Model Performance
- **Accuracy**: ~86% (misleading due to class imbalance)
- **Recall**: ~34% (concerning - missing 66% of actual leavers)
- **AUC**: [Varies by run] - Better metric for imbalanced data
- **Precision**: [Varies by threshold]

### Why Accuracy is Misleading

**The Problem**: 86% accuracy sounds impressive, but 34% recall means we're missing 2 out of 3 employees who actually leave.

**Root Cause**:
- **Class Imbalance**: 84% employees stay, 16% leave
- **Majority Class Bias**: Model gets high accuracy by correctly predicting "stayed" for most employees
- **Business Impact**: Missing departing employees has high cost (replacement, knowledge loss)

**Better Metrics**:
- **Recall**: What percentage of actual leavers do we catch?
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Overall discriminative ability regardless of threshold
- **Business-specific metrics**: Cost of false negatives vs false positives

## 🔍 ROC Curve Analysis

The ROC (Receiver Operating Characteristic) curve provides crucial insights:

- **X-axis**: False Positive Rate (False alarms)
- **Y-axis**: True Positive Rate (Recall/Sensitivity)
- **AUC**: Area Under Curve - single metric for model quality
- **Threshold Selection**: Visual guide for business decision-making

**Key Advantage**: ROC/AUC is not misleading for imbalanced datasets, unlike accuracy.

## ⚙️ Usage Instructions

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Running the Analysis
1. **Prepare Data**: Place `HR-Employees.csv` in the project directory
2. **Run Analysis**: Execute the main Python script
3. **View Results**: Check console output and generated visualizations

### File Structure
```
project/
│
├── dataset/HREmployees.csv                 # Dataset (not included)
├── model.py       # Main analysis script
├── README.md                       # This file
└── outputs/
    ├── confusion_matrix.png        # Generated visualizations
    ├── roc_curve.png
    └── probability_distribution.png
```

## 📊 Visualizations Generated

1. **Confusion Matrix Heatmap**
   - Visual breakdown of prediction accuracy
   - Shows True/False Positives and Negatives

2. **ROC Curve**
   - Trade-off between True Positive Rate and False Positive Rate
   - Includes AUC score and optimal threshold identification

3. **Probability Distribution**
   - Histogram showing predicted probabilities for both classes
   - Demonstrates model's discriminative ability

## 🎛️ Threshold Optimization

The project explores different classification thresholds:

- **Default (0.5)**: Standard threshold, optimizes accuracy
- **Lower Thresholds (0.3-0.4)**: Better recall, more false alarms
- **Higher Thresholds (0.6-0.7)**: Higher precision, miss more leavers
- **Optimal Threshold**: Calculated using ROC curve analysis

### Business Considerations
- **HR Perspective**: Better to investigate happy employees than lose valuable talent
- **Cost Analysis**: Replacement costs (50-200% of salary) vs intervention costs
- **Threshold Selection**: Should align with business priorities and resource constraints

## 💼 Business Impact

### Current Model Performance
- **Missed Opportunities**: 66% of departing employees go undetected
- **Preventive Action**: Limited ability to intervene before employees leave
- **Resource Allocation**: HR time spent on lower-priority cases

### Improvement Strategies
1. **Threshold Adjustment**: Lower threshold for higher recall
2. **Feature Engineering**: Add more predictive variables
3. **Algorithm Comparison**: Try Random Forest, XGBoost, or Neural Networks
4. **Ensemble Methods**: Combine multiple models
5. **Cost-Sensitive Learning**: Penalize false negatives more heavily

## 🔄 Future Enhancements

### Technical Improvements
- [ ] Feature importance analysis and selection
- [ ] Hyperparameter tuning with cross-validation
- [ ] Multiple algorithm comparison
- [ ] Precision-Recall curve analysis
- [ ] Calibration plots for probability reliability

### Business Applications
- [ ] Real-time attrition risk scoring
- [ ] Automated alerts for high-risk employees
- [ ] Intervention strategy recommendations
- [ ] ROI analysis of retention programs

## 📚 Key Takeaways

1. **Accuracy is Misleading**: High accuracy doesn't guarantee good performance on minority classes
2. **Context Matters**: Choose metrics that align with business objectives
3. **Class Imbalance**: Requires special attention in model evaluation and threshold selection
4. **ROC/AUC**: Better evaluation metrics for imbalanced binary classification
5. **Business Understanding**: Domain knowledge crucial for interpreting results

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional evaluation metrics
- Alternative algorithms
- Feature engineering techniques
- Visualization enhancements
- Documentation improvements

## 📄 License

This project is for educational purposes. Dataset rights belong to IBM.

## 📞 Contact

For questions about this analysis or suggestions for improvement, please open an issue in this repository.

---

**Note**: This project demonstrates important machine learning concepts using a realistic business scenario. The emphasis on "misleading accuracy" serves as a crucial lesson for data scientists working with imbalanced datasets.