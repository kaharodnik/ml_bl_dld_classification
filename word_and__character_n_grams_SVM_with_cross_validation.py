# -*- coding: utf-8 -*-
"""Word_and _Character_N-grams_SVM_3_17_24.ipynb

Original file is located at
    https://colab.research.google.com/drive/1qior2LsJ9RcMg9MQI7DnztHNZimHMmCO
"""

# Commented out IPython magic to ensure Python compatibility.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

###Load the data as dataframe
#===========================combined==================================
#df = pd.read_csv('bl_sli_4_6_balanced_translit_no_test_checked.csv', encoding = 'utf-8')
#df = pd.read_csv('bl_sli_7_9_translit_balanced_no_test_checked.csv', encoding = 'utf-8')
df = pd.read_csv('ml_sli_4_6_translit_combined_no_test_checked.csv', encoding = 'utf-8')
#df = pd.read_csv('ml_sli_7_9_pos_combined_no_test_checked.csv', encoding = 'utf-8')
#df = pd.read_csv('bl_sli_7_9_translit_balanced_no_test_checked.csv',encoding = 'utf-8')
#df = pd.read_csv('bl_sli_7_9_pos_balanced_no_test_checked.csv', encoding = 'utf-8')
#df = pd.read_csv('bl_sli_4_6_pos_balanced_no_test_checked.csv', encoding = 'utf-8')
#df = pd.read_csv('bl_sli_4_6_balanced_translit_no_test_checked.csv',encoding = 'utf-8')


print(df.columns)
#print(df.isnull().sum())

print(df.head())
#
sentences = df.text.values
labels = df.label.values
#
# Convert text documents into n-grams representations
# #CountVectorizer creates a sparse matrix where each row represents a file and each column represents an n-gram. The feature matrix is used to train a classifier.
vectorizer = CountVectorizer(ngram_range=(1,2))
#character n-grams
#vectorizer = CountVectorizer(analyzer='char',ngram_range=(1,4))
X = vectorizer.fit_transform(sentences)
y = labels
#
# # Convert the sparse matrix to a dense matrix and then to a DataFrame
feature_names = vectorizer.get_feature_names_out()
#print(feature_names)
# feature_matrix_df = pd.DataFrame(X.toarray(), columns=feature_names)
#
# # Print the DataFrame or send to an csv file
# print(feature_matrix_df)
# output_file = "ngram_feature_matrix.csv"
# feature_matrix_df.to_csv(output_file, index=False)
# print(feature_matrix_df)
# #print(f"Feature matrix saved to {output_file}")
#
# ### Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# ### Train the SVM classifier
svm_classifier = SVC(kernel='linear',C=1.0,class_weight='balanced')
svm_classifier.fit(X_train, y_train)
#
# ### Test - make predictions on the validation set
y_pred = svm_classifier.predict(X_test)
#
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#
# ### Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
#
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
#
# # Print the evaluation metrics
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')
#
#Cross_Validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, LeaveOneOut
#
#
model = SVC(kernel='linear',C=10.0,class_weight='balanced')
#
# Define the number of folds for cross-validation
kfold = KFold(n_splits=6, shuffle=True, random_state=42) # random_state ensures reproducible splits for all models
# #loo = LeaveOneOut()
#
# # Perform cross-validation
scores = cross_val_score(model, X, y, cv=kfold)
# #scores = cross_val_score(model, X, y, cv=loo)
#
# Print the cross-validation scores
print("Cross-validation scores:", scores)
print("Mean CV accuracy:", np.mean(scores))