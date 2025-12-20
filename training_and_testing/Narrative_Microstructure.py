#Download libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#==========Load the data==================
#df = pd.read_csv('ml_sli_4_6_prod_combined_no_test.csv')
df = pd.read_csv('ml_sli_7_9_prod_combined_no_test.csv')

print(df.columns)
print(df.head(10))

#=========define_the_model==========
###Define model
#df_model = df[['label','Types_with_mazes','MLU_with_mazes']]
df_model = df[['label','Types_with_mazes','TNU']]

from sklearn.utils import class_weight
#Define feature matrix X and target variable y=labels
X = df_model.drop('label', axis = 1)
y = df_model.label.values

#check to make sure that the number of rows and labels are equal
#print (X)
#print(len(y))

#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Train / validation split (stratified)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Pipeline = scaling + model
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('clf', LogisticRegression(
        class_weight='balanced',
        C=1.0,
        solver='liblinear',
        random_state=42))
])

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_val)

print(len(y_val))
print(len(y_pred))
print(y_pred)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
conf_matrix = confusion_matrix(y_val, y_pred)
classification_rep = classification_report(y_val, y_pred)

# Print the results
print("Features: Narrative Microstructure")
#print("SLI vs ML TD, Age 4-6 years old")
#print("SLI vs ML TD, Age 7-9 years old")
print("SLI vs BL TD, Age 4-6 years old")
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)