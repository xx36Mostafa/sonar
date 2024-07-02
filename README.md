# Machine Learning Model Evaluation and Hyperparameter Truning

This repository contains a Python script to evaluate and tune various machine learning models using K-Fold cross-validation and Grid Search for hyperparameter optimization. The script supports different scaling ranges for feature normalization and provides performance metrics for each model.

## Models Included
- K-Nearest Neighbors (KNN)
- Support Vector Classifier (SVC)
- Decision Tree Classifier
- Random Forest Classifier

## Features
- Hyperparameter tuning using Grid Search
- Evaluation with K-Fold cross-validation
- Feature scaling using Min-Max Scaler with various ranges
- Performance metrics including accuracy and CPU time

## Prerequisites

Make sure you have the following libraries installed:

```bash
pip install pandas numpy scikit-learn
