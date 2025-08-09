# Customer Churn Prediction Report

## Executive Summary

This report presents the development and evaluation of machine learning models to predict customer churn for a subscription-based service. Using historical customer data from the Churn_Modelling.csv dataset, we built and compared three different algorithms: Logistic Regression, Random Forest, and Gradient Boosting. The Gradient Boosting model achieved the best performance with an accuracy of 87% and an ROC-AUC score of 0.87, making it our recommended model for predicting customer churn.

## 1. Introduction

Customer churn, the rate at which customers stop doing business with a company, is a critical metric for subscription-based businesses. Predicting which customers are likely to churn allows companies to take proactive measures to retain valuable customers. This project aimed to develop a machine learning model that can accurately predict customer churn based on historical data.

## 2. Data Overview

The dataset contains 10,000 records with 14 features including:

- **Demographic information**: Geography, Gender, Age
- **Account information**: Credit Score, Tenure, Balance, Number of Products, Credit Card status, Active Member status, Estimated Salary
- **Target variable**: Exited (1 = churned, 0 = retained)

The dataset is imbalanced with approximately 20.37% of customers having churned.

## 3. Exploratory Data Analysis

### 3.1 Target Variable Distribution

The dataset shows an imbalance with only 20.37% of customers having churned. This imbalance was considered during model training and evaluation.

### 3.2 Categorical Features Analysis

- **Geography**: Customers from Germany and France have higher churn rates compared to Spain.
- **Gender**: Female customers have a higher churn rate than male customers.

### 3.3 Numerical Features Analysis

- **Age**: Customers who churn tend to be older (significant difference in median age).
- **Balance**: Customers who churn tend to have higher account balances.
- **IsActiveMember**: Active members are less likely to churn.
- **NumOfProducts**: Customers with fewer products are slightly more likely to churn.

### 3.4 Correlation Analysis

- Age has the strongest positive correlation with churn (0.29)
- IsActiveMember has a negative correlation with churn (-0.16)
- Balance has a positive correlation with churn (0.12)
- NumOfProducts has a slight negative correlation with churn (-0.05)

## 4. Data Preprocessing

The following preprocessing steps were applied:

1. **Removal of non-predictive columns**: RowNumber, CustomerId, and Surname were removed as they don't contribute to prediction.
2. **Handling categorical variables**: One-hot encoding was applied to Geography and Gender.
3. **Feature scaling**: Numerical features were standardized using StandardScaler.
4. **Train-test split**: Data was split into 80% training and 20% testing sets with stratification to maintain the class distribution.

## 5. Model Development and Evaluation

Three machine learning algorithms were implemented and compared:

### 5.1 Logistic Regression

- **Accuracy**: 80.80%
- **Precision**: 58.91%
- **Recall**: 18.67%
- **F1 Score**: 28.36%
- **ROC AUC**: 77.48%

### 5.2 Random Forest

- **Accuracy**: 86.40%
- **Precision**: 78.24%
- **Recall**: 45.95%
- **F1 Score**: 57.89%
- **ROC AUC**: 85.22%

### 5.3 Gradient Boosting

- **Accuracy**: 87.00%
- **Precision**: 79.28%
- **Recall**: 48.89%
- **F1 Score**: 60.49%
- **ROC AUC**: 87.08%

### 5.4 Hyperparameter Tuning

Hyperparameter tuning was performed for the Gradient Boosting model using GridSearchCV with 5-fold cross-validation. The best parameters were:
- n_estimators: 100
- learning_rate: 0.1
- max_depth: 3

The tuned model maintained the same performance metrics as the base Gradient Boosting model.

## 6. Feature Importance

The Gradient Boosting model identified the following features as most important for predicting churn:

1. **Age** (38.81%): The most influential feature, confirming our EDA finding that older customers are more likely to churn.
2. **NumOfProducts** (29.99%): The number of products a customer has is strongly related to churn probability.
3. **IsActiveMember** (11.39%): Active members are less likely to churn.
4. **Balance** (8.92%): Account balance has a moderate impact on churn.
5. **Geography** (6.91% combined): Customer location affects churn probability.

## 7. Conclusions and Recommendations

### 7.1 Model Selection

The **Gradient Boosting** model is recommended for deployment due to its superior performance across all metrics, particularly its high ROC-AUC score of 0.87.

### 7.2 Business Insights

Based on the feature importance and model results, we recommend the following strategies to reduce customer churn:

1. **Age-based retention strategies**: Develop targeted retention programs for older customers, who are more likely to churn.
2. **Product diversification**: Encourage customers to use more products, as having more products is associated with lower churn.
3. **Engagement initiatives**: Implement programs to increase customer activity and engagement, as active members are less likely to churn.
4. **Geography-specific approaches**: Develop tailored retention strategies for customers in Germany and France, where churn rates are higher.
5. **Balance monitoring**: Pay attention to customers with high balances, as they may be at higher risk of churning.

### 7.3 Future Work

1. **Address class imbalance**: Experiment with techniques like SMOTE or class weighting to improve model performance on the minority class.
2. **Feature engineering**: Create new features that might better capture customer behavior patterns.
3. **Model explainability**: Implement SHAP values or LIME to provide more detailed explanations for individual predictions.
4. **Deployment monitoring**: Set up a system to monitor model performance over time and retrain as needed.
5. **A/B testing**: Test the effectiveness of retention strategies informed by the model predictions.

## 8. Appendix

All code, visualizations, and detailed results are available in the project repository. The final model has been saved and can be loaded for making predictions on new data.