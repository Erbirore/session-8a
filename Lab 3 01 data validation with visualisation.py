# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 05:41:26 2025

@author: hongf
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from scipy.stats import ks_2samp

def load_titanic_dataset():
    df = pd.read_csv('train.csv')
    return df

df = load_titanic_dataset()

# Handling Missing Data
imputer = SimpleImputer(strategy='mean')
df['Age'] = imputer.fit_transform(df[['Age']])
df['Fare'] = imputer.fit_transform(df[['Fare']])

# Ensure 'Embarked' remains 1D after transformation
embarked_imputer = SimpleImputer(strategy='most_frequent')
df['Embarked'] = embarked_imputer.fit_transform(df[['Embarked']]).ravel()

# Visualization of Missing Data
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# Checking for Duplicate Data
df = df.drop_duplicates()

# Data Type Validation
df['Age'] = df['Age'].astype(float)
df['Pclass'] = df['Pclass'].astype(int)

# Visualization of Data Types
sns.countplot(x=df.dtypes, palette='pastel')
plt.title("Data Type Distribution")
plt.show()

# Handling Outliers
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]
outliers = detect_outliers(df, 'Fare')

# Visualization of Outliers
sns.boxplot(x=df['Fare'])
plt.title("Boxplot of Fare (Detecting Outliers)")
plt.show()

# Ensuring Consistency in Categorical Variables
df['Embarked'] = df['Embarked'].str.strip()

# Validating Data Ranges
df = df[(df['Age'] >= 0) & (df['Age'] <= 100)]

# Handling Imbalanced Data
X = df.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin'], errors='ignore')
y = df['Survived']
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X.select_dtypes(exclude=['object']), y)

# Visualization of Class Imbalance
sns.countplot(x=y, palette='pastel')
plt.title("Class Distribution Before SMOTE")
plt.show()

sns.countplot(x=y_resampled, palette='pastel')
plt.title("Class Distribution After SMOTE")
plt.show()

# Feature Engineering Validation
X_resampled['FamilySize'] = X_resampled['SibSp'] + X_resampled['Parch']

# Identifying and Removing Highly Correlated Features
corr_matrix = X_resampled.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
X_resampled.drop(columns=to_drop, inplace=True)

# Visualization of Correlation Matrix
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix Before Dropping Highly Correlated Features")
plt.show()

# Splitting Data Properly
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Encoding Categorical Variables
encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(df[['Embarked', 'Sex']])
df = df.drop(columns=['Embarked', 'Sex'], errors='ignore')
df = pd.concat([df.reset_index(drop=True), pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['Embarked', 'Sex']))], axis=1)

# Checking for Data Drift
ks_stat, p_value = ks_2samp(X_train['Age'], X_test['Age'])
print(f'KS Statistic: {ks_stat}, P-Value: {p_value}')

# Visualization of Data Drift
sns.kdeplot(X_train['Age'], label='Train Age', fill=True, alpha=0.5)
sns.kdeplot(X_test['Age'], label='Test Age', fill=True, alpha=0.5)
plt.title("Train vs Test Age Distribution")
plt.legend()
plt.show()

# Normalization & Standardization
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Visualization of Normalization & Standardization
sns.histplot(df['Age'], kde=True, bins=30)
plt.title("Standardized Age Distribution")
plt.show()

# Cross-Validation
model = RandomForestClassifier(random_state=42)
cv_scores = cross_val_score(model, X_resampled, y_resampled, cv=5)
print(f'Cross-Validation Scores: {cv_scores}')

# Visualization of Cross-Validation Scores
plt.bar(range(1, 6), cv_scores, color='royalblue')
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title("Cross-Validation Scores")
plt.show()

# Display Processed Data
df.head()
