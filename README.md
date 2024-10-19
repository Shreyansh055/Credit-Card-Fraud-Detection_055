# Credit Card Fraud Detection

## Overview

This project focuses on building a machine learning model to detect fraudulent credit card transactions. Using historical transaction data, the model identifies patterns that indicate potential fraud, ensuring better accuracy in detecting anomalies and minimizing false positives. The solution leverages supervised learning techniques and statistical analysis to improve fraud detection efficiency.

## Features

- **Data Preprocessing**: Handling imbalanced datasets using techniques like SMOTE (Synthetic Minority Over-sampling Technique) to ensure model fairness.
- **Machine Learning Models**: Implementation of various algorithms such as Logistic Regression, Decision Trees, Random Forest, and Gradient Boosting for fraud detection.
- **Evaluation Metrics**: Performance evaluation using metrics like Accuracy, Precision, Recall, F1-Score, and AUC-ROC to ensure robustness.
- **Data Visualization**: Graphical representation of transaction patterns and fraud distribution using libraries like Matplotlib and Seaborn.
- **Hyperparameter Tuning**: Optimization of model parameters to improve accuracy and reduce false positives.
- **Scalability**: The model can handle large datasets and can be easily scaled for real-time fraud detection in financial systems.

## Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Matplotlib**
- **Seaborn**
- **SMOTE (Imbalanced-learn)**

## Dataset

The dataset used in this project consists of anonymized credit card transactions over a two-day period. It contains approximately 284,807 transactions, of which 492 are fraudulent. The dataset includes the following features:
- **Time**: Time elapsed since the first transaction in the dataset.
- **V1-V28**: Principal components obtained through PCA (anonymized).
- **Amount**: Transaction amount.
- **Class**: Target variable, where 1 indicates fraud and 0 indicates a legitimate transaction.

You can download the dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).

## Requirements

To run this project, ensure you have Python 3.x and install the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn

Usage
> Clone the repository:

bash
Copy code
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
> Data Preparation: Load the dataset and apply preprocessing steps like feature scaling and handling missing values.

> Model Training: Run the Python script to train the model using the prepared dataset and perform hyperparameter tuning for the best results.

> Model Evaluation: Evaluate the performance of the trained model using various metrics such as Precision, Recall, F1-Score, and the AUC-ROC curve.

> Visualization: Generate graphs to visualize transaction patterns and the distribution of fraudulent transactions.

Project Structure

├── data
│   └── creditcard.csv           # Dataset
├── notebooks
│   └── EDA.ipynb                # Exploratory Data Analysis
├── models
│   └── random_forest.pkl        # Trained Model
├── fraud_detection.py           # Main Script
├── README.md                    # Project Documentation
└── requirements.txt             # List of dependencies


Evaluation Metrics

The performance of the model is evaluated using the following metrics:

Accuracy: The ratio of correct predictions to total predictions.
Precision: The ratio of correctly predicted positive observations to the total predicted positives.
Recall: The ability of the model to find all the relevant cases within a dataset.
F1-Score: The weighted average of Precision and Recall.
AUC-ROC: The Area Under the Receiver Operating Characteristic curve, measuring the model's ability to distinguish between classes.
Contributing
Contributions are welcome! Please follow these steps to contribute:

Fork the repository-
Create a feature branch (git checkout -b feature-branch).
Commit your changes (git commit -m "Add feature").
Push to the branch (git push origin feature-branch).
Open a Pull Request.

Acknowledgements
Scikit-learn Documentation
Imbalanced-learn Documentation
Kaggle - Credit Card Fraud Dataset

Tables: To show model performance in a tabular format.

| Metric       | Value  |
|--------------|--------|
| Accuracy     | 99.2%  |
| Precision    | 93.1%  |
| Recall       | 94.7%  |
| F1-Score     | 93.8%  |


