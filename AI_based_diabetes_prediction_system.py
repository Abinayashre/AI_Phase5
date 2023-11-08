7321 - Nandha College Of Technology
Project by
ABINAYA P
_____________________________________________________

# Step 1: Import necessary libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings warnings.filterwarnings("ignore", category=UserWarning) from sklearn.model_selection 
import train_test_split from sklearn.preprocessing 
import StandardScaler from sklearn.svm 
import SVC from sklearn.metrics 
import accuracy_score, classification_report, confusion_matrix 
# Step 2: Load the dataset
df = pd.read_csv("/kaggle/input/diabetes-data-set/diabetes.csv") 
# Step 3: Data Cleaning
# Check for Missing Values 
missing_values = df.isnull().sum()
print("Missing Values:")
print(missing_values) 
# Handle missing values (if any) 
# For example, fill missing values with the mean of the column 
mean_fill = df.mean()
df.fillna(mean_fill, inplace=True) 
# Check for Duplicate Rows 
duplicate_rows = df[df.duplicated()] 
print("\nDuplicate Rows:") 
print(duplicate_rows) 
# Handle duplicate rows (if any) 
# For example, drop duplicate rows 
df.drop_duplicates(inplace=True)
# Step 4: Data Analysis 
# Summary Statistics
summary_stats = df.describe() 
print("\nSummary Statistics:")
print(summary_stats)
# Class Distribution (for binary classification problems) 
class_distribution = df['Outcome'].value_counts()
print("\nClass Distribution:")
print(class_distribution) 
# Step 5: Data Visualization 
sns.pairplot(df, hue='Outcome') 
plt.show() 
# Step 6: Support Vector Machine (SVM) Modeling 
# Separate features and target variable 
X = df.drop('Outcome', axis=1)
y = df['Outcome'] 
# Split the dataset into a training and testing set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test)
# Initialize and train the SVM model 
model = SVC(kernel='linear', random_state=42) 
model.fit(X_train, y_train) 
# Make predictions 
y_pred = model.predict(X_test) 
# Evaluate the model 
accuracy = accuracy_score(y_test, y_pred) 
print(f'Accuracy: {accuracy:.2f}') 
# Classification report and confusion matrix 
print(classification_report(y_test, y_pred)) 
cm = confusion_matrix(y_test, y_pred) 
sns.heatmap(cm, annot=True, fmt='d') 
plt.show()
