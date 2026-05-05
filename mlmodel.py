# get kagglehub ready
!pip install kagglehub

# imports
import pandas as pd
import numpy as np
import kagglehub # to grab data from Kaggle
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import time
from IPython.display import display # for pretty pandas tables

# load data
print("Grabbing the dataset from Kaggle...")
try:
    data = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "altruistdelhite04/loan-prediction-problem-dataset",
        "train_u6lujuX_CVtuZ9i.csv"
    )
    print("Dataset loaded! Here's a peek:")
    display(data.head())
except Exception as e:
    print(f"Oops, couldn't load the dataset: {e}")
    raise # stops everything if data fails to load

# basic data exploration
print("\n-- Quick Data Check --")

print("\nInfo:")
data.info()

print("\nStats for numbers:")
display(data.describe())

print("\nWhat's up with Loan Status (our target)?")
display(data['Loan_Status'].value_counts())
print("\nLoan Status percentage:")
display(data['Loan_Status'].value_counts(normalize=True) * 100)

# clean data
print("\n-- Cleaning up the data --")

# get rid of Loan_ID, it's useless for prediction
data = data.drop("Loan_ID", axis=1)
print("Dropped 'Loan_ID' column.")

# change 'Y'/'N' to 1/0 for the target
data["Loan_Status"] = data["Loan_Status"].map({"Y": 1, "N": 0})
print("Converted 'Loan_Status' to 1s and 0s.")

# fill in missing values
print("Filling in missing values (mode for text, mean for numbers):")
for col in data.columns:
    if data[col].dtype == "object":
        if data[col].isnull().any():
            data[col].fillna(data[col].mode()[0], inplace=True)
    else:
        if data[col].isnull().any():
            data[col].fillna(data[col].mean(), inplace=True)

# turn text categories into numbers
print("Encoding categorical stuff:")
encoder = LabelEncoder()
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = encoder.fit_transform(data[col])

# prepare for training: split features (X) and target (y)
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]
print("\nSplit X and y.")

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into train (80%) and test (20%) sets.")

# scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled.")

# train models
# trying out a few classification models
models_to_try = {
    "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear'),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42) # need probability for ROC curves
}

results = {} # store performance metrics here
roc_info = {} # for ROC plots

print("\n-- Training and Checking Models --")

for name, model in models_to_try.items():
    print(f"\n--- {name} ---")

    # train it!
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    print(f"Training took: {train_time:.4f} seconds")

    # make predictions
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1] # probs for ROC

    # calculate common metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, preds)
    display(pd.DataFrame(cm,
                         index=['Actual N', 'Actual Y'],
                         columns=['Predicted N', 'Predicted Y']))

    # classification report
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    # ROC curve stuff
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc:.4f}")

    roc_info[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}

    # cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"Cross-val Accuracy (5 folds): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # save results
    results[name] = {
        'Accuracy': acc, 'Precision': prec, 'Recall': rec,
        'F1-Score': f1, 'ROC AUC': roc_auc, 'CV Accuracy': cv_scores.mean(),
        'Train Time (s)': train_time
    }

# compare models
print("\n-- ROC Curve Comparison --")
plt.figure(figsize=(10, 8))
for name, data_item in roc_info.items(): # Renamed 'data' to 'data_item' to avoid conflict with outer 'data' variable
    plt.plot(data_item['fpr'], data_item['tpr'], label=f"{name} (AUC = {data_item['auc']:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess (AUC = 0.50)')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for All Models")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

print("\n-- All Model Results --")
display(pd.DataFrame(results).T.sort_values(by='ROC AUC', ascending=False))

# hyperparameter tuning
# let's try to make the best model even better
best_model_name = max(results, key=lambda k: results[k]['ROC AUC'])
print(f"\n-- Tuning {best_model_name} --")

# sticking with Logistic Regression for tuning for now
tune_model = LogisticRegression(random_state=42)

# parameters to try
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga']
}

# run grid search
grid_search = GridSearchCV(tune_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
print("Starting grid search, might take a sec...")
grid_search.fit(X_train_scaled, y_train)

print(f"Best params for {best_model_name}: {grid_search.best_params_}")
print(f"Best cross-val ROC AUC: {grid_search.best_score_:.4f}")

best_tuned_model = grid_search.best_estimator_
print(f"Trained {best_model_name} again with the best settings.")

# final check on the tuned model
print(f"\n-- Final Check on Tuned {best_model_name} --")

preds_tuned = best_tuned_model.predict(X_test_scaled)
probs_tuned = best_tuned_model.predict_proba(X_test_scaled)[:, 1]

acc_tuned = accuracy_score(y_test, preds_tuned)
prec_tuned = precision_score(y_test, preds_tuned)
rec_tuned = recall_score(y_test, preds_tuned)
f1_tuned = f1_score(y_test, preds_tuned)
fpr_tuned, tpr_tuned, _ = roc_curve(y_test, probs_tuned)
roc_auc_tuned = auc(fpr_tuned, tpr_tuned)

print(f"Tuned Model Accuracy: {acc_tuned:.4f}")
print(f"Tuned Model Precision: {prec_tuned:.4f}")
print(f"Tuned Model Recall: {rec_tuned:.4f}")
print(f"Tuned Model F1-Score: {f1_tuned:.4f}")
print(f"Tuned Model ROC AUC: {roc_auc_tuned:.4f}")

print("\nConfusion Matrix for Tuned Model:")
cm_tuned = confusion_matrix(y_test, preds_tuned)
display(pd.DataFrame(cm_tuned,
                         index=['Actual N', 'Actual Y'],
                         columns=['Predicted N', 'Predicted Y']))

print("\nClassification Report for Tuned Model:")
print(classification_report(y_test, preds_tuned))

# plot ROC for the tuned model
plt.figure(figsize=(8, 6))
plt.plot(fpr_tuned, tpr_tuned, label=f"Tuned {best_model_name} (AUC = {roc_auc_tuned:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess (AUC = 0.50)')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve for Tuned {best_model_name}")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

