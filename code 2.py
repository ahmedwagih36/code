import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from imblearn.over_sampling import SMOTE

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier

# Load dataset
df = pd.read_csv(r"C:\Users\Ahmed\Desktop\New folder\WSN-DS.csv")

# Convert 'label' column to numeric if it contains categorical values
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Basic info
print(df.info())
print(df.describe())

# Class distribution
print(df['label'].value_counts())
sns.countplot(x='label', data=df)
plt.title('Class Distribution')
plt.show()

# Box plots
df.drop(columns='label').plot(kind='box', subplots=True, layout=(6,3), figsize=(15,20), sharex=False, sharey=False)
plt.tight_layout()
plt.show()

# Correlation heatmap - using only numeric columns
df_numeric = df.select_dtypes(include=['number'])
plt.figure(figsize=(12, 10))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()

# Pairplot (sampled for performance)
sns.pairplot(df.sample(300), hue='label', diag_kind='kde')
plt.show()

# Prepare features and labels
X = df.drop(columns='label')
y = df['label']

# Balance data using SMOTE
smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_res, test_size=0.2, random_state=42)

# Define models with hyperparameter grid
models = {
    'Naive Bayes': (GaussianNB(), {}),
    'Decision Tree': (DecisionTreeClassifier(), {'max_depth': [3, 5, 10]}),
    'SVM': (SVC(probability=True), {'C': [1, 10], 'kernel': ['linear', 'rbf']}), 
    'KNN': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]}),
    'Random Forest': (RandomForestClassifier(), {'n_estimators': [50, 100], 'max_depth': [5, 10]})
}

best_models = {}
plt.figure(figsize=(10,8))

for i, (name, (model, params)) in enumerate(models.items()):
    grid = GridSearchCV(model, params, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_
    
    y_pred = grid.predict(X_test)
    y_prob = grid.predict_proba(X_test)
    
    print(f"--- {name} ---")
    print("Best Params:", grid.best_params_)
    print(classification_report(y_test, y_pred))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

    # Handle multiclass ROC curve by iterating over each class
    for i in range(y_prob.shape[1]):
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, i], pos_label=i)
        plt.plot(fpr, tpr, label=f"{name} - Class {i} (AUC = {roc_auc_score(y_test == i, y_prob[:, i]):.2f})")

# Final ROC Curve plot (for all models)
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for All Models")
plt.legend()
plt.show()

# Ensemble model (stacking)
estimators = [
    ('knn', best_models['KNN']),
    ('svm', best_models['SVM']),
    ('rf', best_models['Random Forest'])
]
stack_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stack_model.fit(X_train, y_train)

y_pred_stack = stack_model.predict(X_test)
y_prob_stack = stack_model.predict_proba(X_test)

print("--- Stacking Model ---")
print(classification_report(y_test, y_pred_stack))
sns.heatmap(confusion_matrix(y_test, y_pred_stack), annot=True, fmt='d')
plt.title("Confusion Matrix - Stacking")
plt.show()

# ROC for stacking (handle multiclass)
y_prob_stack_bin = stack_model.predict_proba(X_test)

# Calculate ROC for each class
for i in range(y_prob_stack_bin.shape[1]):
    fpr, tpr, _ = roc_curve(y_test, y_prob_stack_bin[:, i], pos_label=i)
    plt.plot(fpr, tpr, label=f"Stacking - Class {i} (AUC = {roc_auc_score(y_test == i, y_prob_stack_bin[:, i]):.2f})")

# Final ROC for stacking model
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Stacking Model")
plt.legend()
plt.show()