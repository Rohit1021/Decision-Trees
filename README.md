### Project Goal

This project aims to build a supervised classification model to predict in-hospital mortality, based on data from the MIMIC dataset, which includes physiological measurements for two patients. Using features such as heart rate (HR), respiratory rate (RESP), oxygen saturation (SpO2), systolic blood pressure (BP-S), diastolic blood pressure (BP-D), and pulse rate (PULSE), the model predicts the likelihood of in-hospital mortality. The target variable, "Anomaly," is binary: ‘1’ indicates the patient passed away in the hospital, and ‘0’ indicates discharge.

### Libraries Used

This project leverages several key libraries:

- **Pandas** and **NumPy**: For data handling, including loading, cleaning, and performing numerical operations.
- **Scikit-Learn**: For evaluation metrics, including accuracy, precision, recall, F1-score, and AUC-ROC, critical for understanding model performance.
- **TensorFlow**: Used for implementing advanced ensemble methods such as Random Forest and Gradient Boosted Trees, allowing a more robust and scalable approach.

### Algorithms Implemented

1. **Custom ID-3 Algorithm**: A custom implementation of the ID-3 decision tree algorithm was used to interpret each feature’s impact on mortality prediction. This algorithm works by iteratively selecting features with the highest information gain, aiming to reduce entropy at each level and achieve an optimal decision boundary.

2. **CART Algorithm**: Implementing the CART (Classification and Regression Trees) algorithm provided a way to explore binary splits based on the Gini index. This was useful for understanding how well each feature can divide the dataset into the target classes.

3. **Random Forest (TensorFlow)**: Using TensorFlow, I implemented a Random Forest classifier to aggregate multiple decision trees and enhance prediction accuracy. This ensemble approach helps mitigate overfitting and improves generalization.

4. **Gradient Boosted Trees**: Gradient Boosted Trees were also implemented to strengthen predictive power through boosting, where each tree is trained to correct the errors of the previous ones. This algorithm is particularly effective at handling complex relationships in the data.

### Summary

This project combines multiple decision tree-based approaches to optimize prediction of in-hospital mortality, providing insight into which clinical features most significantly contribute to patient outcomes.
