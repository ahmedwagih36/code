# Extracted and Organized Code from Wireless Sensor Network Project Notebook.pdf

# ====================================
# Step 1: Data Loading and Initial Inspection
# ====================================

# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab import drive # Note: This is specific to Google Colab
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder # OneHotEncoder is included though commented out later
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score

# Mount Google Drive (specific to Google Colab)
# drive.mount(
#     '/content/gdrive')

# Load the dataset
# Update the path to your dataset file
# data = pd.read_csv(
#     '/content/gdrive/MyDrive/Datasets/WSN-DS.csv')
# Assuming the data is loaded into a pandas DataFrame named 'data'
# For demonstration, let's assume 'data' is already loaded.
# Replace this with actual data loading if running outside the original context.
# Example placeholder if data isn't loaded:
# data = pd.DataFrame(np.random.rand(100, 19), columns=[f'feature_{i}' for i in range(18)] + ['label'])
# data['label'] = np.random.randint(0, 5, 100)

# Display first few rows
# data.head()

# Display basic information
# print(f"Number of Rows: {data.shape[0]}")
# print(f"Number of Columns: {data.shape[1]}")
# data.info()

# Display descriptive statistics
# data.describe()

# Display class distribution
# print("Class Distribution:\n", data.iloc[:, -1].value_counts())

# Check for missing values
# data.isnull().sum()

# ====================================
# Step 2: Data Preprocessing
# ====================================

# Remove Outliers (using IQR method)
# Select only numerical columns for outlier detection
# numerical_columns = data.select_dtypes(include=[
#     'float64', 'int64']).columns
# Calculate Q1 (25th percentile) and Q3 (75th percentile) for numerical columns only
# Q1 = data[numerical_columns].quantile(0.25)
# Q3 = data[numerical_columns].quantile(0.75)
# IQR = Q3 - Q1
# Filter out outliers from numerical columns
# data_no_outliers = data[~((data[numerical_columns] < (Q1 - 1.5 * IQR)) | (data[numerical_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]
# print(f"Number of Rows after outlier removal: {data_no_outliers.shape[0]}")
# data = data_no_outliers.copy() # Use data without outliers

# Handle Imbalanced Data (using SMOTE)
# Separate features and labels
# X = data.iloc[:, :-1]
# y = data.iloc[:, -1]

# Calculate Imbalance Ratio
# class_counts = y.value_counts()
# majority_class_count = class_counts.max()
# minority_class_count = class_counts.min()
# imbalance_ratio = majority_class_count / minority_class_count
# print(f"Imbalance Ratio: {imbalance_ratio}")

# Apply SMOTE to balance the classes
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X, y)

# Combine the resampled features and labels into a new DataFrame
# data_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=y.name)], axis=1)
# print("Resampled Data Shape:", data_resampled.shape)
# data = data_resampled.copy() # Use resampled data for subsequent steps

# ====================================
# Step 3: Exploratory Data Analysis (EDA) and Visualization
# ====================================
# Assuming 'data' is the preprocessed (outlier removed, resampled) DataFrame

# Plot the class distribution (Pie Chart)
# class_counts = data.iloc[:, -1].value_counts()
# plt.figure(figsize=(8, 8))
# plt.pie(class_counts, labels=class_counts.index, autopct=
#     "%1.1f%%", startangle=140, colors=sns.color_palette("tab10"))
# plt.title("Class Distribution")
# plt.show()

# Plot box plots for all numerical features
# plt.figure(figsize=(20, 15))
# for i, column in enumerate(data.columns[:-1], 1):
#     plt.subplot(5, 4, i) # Adjust grid size (5x4) based on number of features
#     sns.boxplot(x=data[column], color="cyan")
#     plt.title(f"Box Plot of {column}")
# plt.tight_layout()
# plt.show()

# Plot pair plots (for a subset of features)
# subset_features = data.columns[:5] # Adjust subset as needed
# subset_data = data[subset_features.tolist() + [data.columns[-1]]]
# sns.pairplot(subset_data, hue=subset_data.columns[-1], palette="tab10", markers=["o", "s", "D", "^", "+"])
# plt.suptitle("Pair Plots of Selected Features", y=1.02)
# plt.show()

# Correlation Matrix
# Calculate the correlation matrix
# corr_matrix = data.corr()
# Visualize the correlation matrix using a heatmap
# plt.figure(figsize=(12, 10))
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
# plt.title("Correlation Matrix")
# plt.show()

# ====================================
# Step 4: Feature Engineering
# ====================================
# Assuming 'data' is the preprocessed DataFrame

# Separate features (X) and labels (y)
# X = data.iloc[:, :-1]
# y = data.iloc[:, -1]

# Perform dimensionality reduction techniques (PCA)
# Apply PCA to reduce dimensions
# pca = PCA(n_components=10) # Adjust n_components as needed
# X_pca = pca.fit_transform(X)
# explained_variance = pca.explained_variance_ratio_
# print("Explained Variance by Each Principal Component:\n", explained_variance)
# X = pd.DataFrame(X_pca) # Use PCA-transformed features

# Encode Categorical Variables (Example - dataset seems numeric)
# If categorical features existed:
# X_categorical = data.select_dtypes(include=["object", "category"])
# onehotencoder = OneHotEncoder()
# X_encoded = onehotencoder.fit_transform(X_categorical).toarray()
# Combine encoded categorical features with numerical features
# X = pd.concat([pd.DataFrame(X_numerical_scaled), pd.DataFrame(X_encoded)], axis=1)

# Scale Numerical Features
# Using StandardScaler
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X = pd.DataFrame(X_scaled, columns=X.columns) # Use scaled features

# Or using MinMaxScaler
# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X)
# X = pd.DataFrame(X_scaled, columns=X.columns) # Use scaled features

# ====================================
# Step 5: Model Selection and Building
# ====================================
# Assuming X and y are prepared features and labels from previous steps
# Use X_scaled and y (or y_resampled if SMOTE was used)

# Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create a smaller subset for faster demonstration/tuning (optional)
# X_small, _, y_small, _ = train_test_split(X_train, y_train, test_size=0.99, random_state=42)
# X_train_small, X_test_small, y_train_small, y_test_small = train_test_split(X_small, y_small, test_size=0.25, random_state=42)

# Initialize models
# models = {
#     "Naive Bayes": GaussianNB(),
#     "Decision Tree": DecisionTreeClassifier(random_state=42),
#     "SVM": SVC(random_state=42),
#     "KNN": KNeighborsClassifier(),
#     "Random Forest": RandomForestClassifier(random_state=42)
# }

# Train and evaluate initial models (using the smaller dataset)
# print("--- Initial Model Training & Evaluation (Small Dataset) ---")
# for name, model in models.items():
#     model.fit(X_train_small, y_train_small)
#     y_pred_small = model.predict(X_test_small)
#     print(f"Model: {name}")
#     print(f"Accuracy: {accuracy_score(y_test_small, y_pred_small)}")
#     print(f"Classification Report:\n{classification_report(y_test_small, y_pred_small)}")
#     print("-" * 50)

# Optimize hyperparameters using GridSearchCV (using the smaller dataset)
# print("--- Hyperparameter Tuning (GridSearchCV - Small Dataset) ---")
# param_grids = {
#     "Naive Bayes": {},
#     "Decision Tree": {"max_depth": [None, 10, 20, 30], "min_samples_split": [2, 10, 20]},
#     "SVM": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
#     "KNN": {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]},
#     "Random Forest": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20], "min_samples_split": [2, 10, 20]}
# }

# best_models = {}
# for name, model in models.items():
#     if name in param_grids and param_grids[name]:
#         print(f"Tuning hyperparameters for {name}...")
#         grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring="accuracy", n_jobs=-1)
#         grid_search.fit(X_train_small, y_train_small)
#         best_models[name] = grid_search.best_estimator_
#         print(f"Best parameters for {name}: {grid_search.best_params_}")
#         print(f"Best cross-validation score for {name}: {grid_search.best_score_}")
#     else:
#         best_models[name] = model
#         print(f"No hyperparameters to tune for {name}. Using default model.")
#     print("-" * 50)

# ====================================
# Step 6: Ensemble Model Building (Stacking)
# ====================================
# Assuming 'best_models' dictionary contains the tuned models from GridSearchCV

# Define base models (using the best models found)
# base_models_list = [
#     ("nb", best_models["Naive Bayes"]),
#     ("dt", best_models["Decision Tree"]),
#     ("svm", make_pipeline(StandardScaler(), best_models["SVM"])), # Scale SVM input
#     ("knn", best_models["KNN"]),
#     ("rf", best_models["Random Forest"])
# ]

# Define the meta-model
# meta_model = LogisticRegression()

# Create the Stacking Classifier
# stacking_model = StackingClassifier(estimators=base_models_list, final_estimator=meta_model, cv=5)

# Train the ensemble model (using the smaller dataset)
# print("--- Training Stacking Classifier (Small Dataset) ---")
# stacking_model.fit(X_train_small, y_train_small)
# print("Stacking Classifier trained.")

# Tune the hyperparameters of the Stacking model's final estimator (optional)
# print("--- Tuning Stacking Classifier Meta-Model (Small Dataset) ---")
# param_grid_stacking = {
#     "final_estimator__C": [0.1, 1, 10],
#     "final_estimator__solver": ["lbfgs", "saga"]
# }
# grid_search_stacking = GridSearchCV(estimator=stacking_model, param_grid=param_grid_stacking, cv=3, scoring="accuracy", n_jobs=-1)
# grid_search_stacking.fit(X_train_small, y_train_small)
# best_stacking_model = grid_search_stacking.best_estimator_
# print("Best Parameters for Stacking Model:", grid_search_stacking.best_params_)

# ====================================
# Step 7: Final Model Evaluation
# ====================================
# Assuming 'best_models' and 'best_stacking_model' are trained and tuned

# Function to evaluate and print model performance on the test set
# def evaluate_final_model(name, model, X_test_set, y_test_set):
#     y_pred = model.predict(X_test_set)
#     accuracy = accuracy_score(y_test_set, y_pred)
#     report = classification_report(y_test_set, y_pred)
#     print(f"Model: {name}")
#     print(f"Accuracy on Test Set: {accuracy}")
#     print(f"Classification Report on Test Set:\n{report}")
#     print("-" * 50)

# Evaluate standalone classifiers on the original test set (X_test, y_test)
print("--- Final Standalone Model Performance (Test Set) ---")
for name, model in best_models.items():
    # Ensure models used for final evaluation are trained on the full training set if desired
    # model.fit(X_train, y_train) # Optional: Retrain on full training data
    evaluate_final_model(name, model, X_test, y_test)

# Evaluate the optimized stacking model on the original test set (X_test, y_test)
print("--- Final Optimized Stacking Model Performance (Test Set) ---")
# Ensure the stacking model is trained on the full training set if desired
# best_stacking_model.fit(X_train, y_train) # Optional: Retrain on full training data
# evaluate_final_model("Optimized Stacking Model", best_stacking_model, X_test, y_test)

print("Code extraction and organization complete. Please review and uncomment the sections you need.")



# Page 35 - Step 6: Model Evaluation (Additional Metrics)
# This section appears in the PDF after the initial comparison

from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay

# Function to evaluate and print model performance (potentially more detailed)
def evaluate_model_detailed(name, model, X_test_set, y_test_set):
    y_pred = model.predict(X_test_set)
    accuracy = accuracy_score(y_test_set, y_pred)
    report = classification_report(y_test_set, y_pred)
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")
    
    # Compute and print AUC-ROC score (for binary or multiclass using appropriate averaging)
    # Check if the model supports predict_proba
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_set)
        # For multiclass, use One-vs-Rest (OvR) or One-vs-One (OvO) strategy
        # Here, using OvR with macro averaging as an example
        try:
            auc_score = roc_auc_score(y_test_set, y_prob, multi_class=
                "ovr", average="macro")
            print(f"AUC-ROC Score (OvR, macro): {auc_score}")
        except ValueError as e:
            print(f"Could not compute AUC-ROC: {e}")
    else:
        print("AUC-ROC score calculation requires predict_proba method.")
        
    # Compute and display Confusion Matrix
    cm = confusion_matrix(y_test_set, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix for {name}")
    plt.show()
    
    print("-" * 50)

# Example usage (commented out, assumes models and data are ready)
# print("--- Detailed Model Evaluation (Test Set) ---")
# for name, model in best_models.items():
#     evaluate_model_detailed(name, model, X_test, y_test)
# evaluate_model_detailed("Optimized Stacking Model", best_stacking_model, X_test, y_test)


