# 01_CustomerBehavoiur_202401100300061
Customer Behavior Prediction
This project aims to classify customers as either Bargain Hunters or Premium Buyers based on their purchasing history. The goal is to understand customer behavior and help businesses develop targeted marketing strategies. This is a classification problem that utilizes machine learning techniques such as logistic regression, random forests, and support vector machines (SVM). Additionally, clustering techniques like K-Means are applied to segment customers based on their purchasing patterns.

Table of Contents
Project Overview

Dataset

Installation

Data Preprocessing

Modeling

Evaluation Metrics

Confusion Matrix & Heatmap

Segmentation & Clustering

Results

Usage

License

Project Overview
In this project, we classify customers into two categories based on their purchasing behavior:

Bargain Hunters: Customers who make frequent but low-value purchases.

Premium Buyers: Customers who make fewer but high-value purchases.

The model predicts whether a customer belongs to the "Bargain Hunter" or "Premium Buyer" category based on features derived from their purchase history.

Key Objectives:
Predict customer behavior using machine learning classification algorithms.

Evaluate model performance using metrics such as accuracy, precision, recall, and F1-score.

Segment customers based on purchasing patterns using clustering techniques.

Dataset
The dataset used for this project should contain the following features:

Customer ID: Unique identifier for each customer.

Total Spend: Total amount spent by the customer across all purchases.

Purchase Frequency: Number of transactions made by the customer in a given time period.

Recency: Time since the customer's last purchase.

Purchase Categories: Categories of products purchased (optional but useful for further insights).

Demographic Information (optional): Age, location, or other personal attributes.

The dataset should be cleaned and preprocessed before feeding it into the model. A sample dataset can be found in the data/ folder.

Installation
To set up the project locally, follow these steps:

Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/customer-behavior-prediction.git
cd customer-behavior-prediction
Create a virtual environment (optional but recommended):

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
Install the required dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Data Preprocessing
Data preprocessing is a critical part of this project, and it involves:

Handling missing values: Ensuring that no missing data remains in the dataset.

Feature engineering: Creating new features such as the average spend per transaction, the frequency of purchases, etc.

Scaling numerical features: Normalizing or standardizing continuous features like total spend and frequency of purchases to improve model performance.

Encoding categorical variables: One-hot encoding categorical variables like purchase categories.

The preprocessing steps are implemented in the preprocessing.py file.

Modeling
We implement several machine learning classification algorithms, including:

Logistic Regression: A basic but effective model for binary classification.

Random Forest Classifier: A powerful ensemble method that combines multiple decision trees.

XGBoost: An advanced gradient boosting technique.

Support Vector Machine (SVM): A classification algorithm that finds the hyperplane that best separates the classes.

Models are evaluated using cross-validation and hyperparameter tuning to select the best-performing model.

The code for model training and evaluation can be found in model.py.

Evaluation Metrics
To evaluate the performance of the classification models, we use the following metrics:

Accuracy: The proportion of correctly classified instances.

Precision: The proportion of true positives out of all predicted positives.

Recall: The proportion of true positives out of all actual positives.

F1-Score: The harmonic mean of precision and recall.

These metrics help assess how well the model predicts customer behavior.

The evaluation metrics are calculated in the evaluate_model.py file.

Confusion Matrix & Heatmap
We generate a confusion matrix to evaluate the performance of the classification model. The matrix is visualized as a heatmap to provide a clear representation of true positives, false positives, true negatives, and false negatives.

Example of confusion matrix code:
python
Copy
Edit
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Actual values and predicted values
y_true = [0, 1, 0, 1, 1, 0, 1]
y_pred = [0, 0, 0, 1, 1, 0, 1]

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bargain Hunter', 'Premium Buyer'], yticklabels=['Bargain Hunter', 'Premium Buyer'])
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
This will produce a confusion matrix heatmap for better model performance interpretation.
