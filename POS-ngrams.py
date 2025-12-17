import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#===========================combined POS==============================
df = pd.read_csv('ml_sli_4_6_pos_combined_no_test.csv', encoding = 'utf-8')
#df = pd.read_csv('ml_sli_7_9_pos_combined_no_test.csv', encoding = 'utf-8')
#df = pd.read_csv('bl_sli_7_9_pos_balanced_no_test.csv', encoding = 'utf-8')
#df = pd.read_csv('bl_sli_4_6_pos_balanced_no_test.csv', encoding = 'utf-8')
#label_list = [0,1]
#print(df.columns)

sentences = df.text.values
labels = df.label.values

# Convert text documents into n-grams representations
#CountVectorizer creates a sparse matrix where each row represents a feature
#Adjust n-gram range as needed for each n-gram type
vectorizer = CountVectorizer(ngram_range=(1,3))
X = vectorizer.fit_transform(sentences)
y = labels

# Convert the sparse matrix to a dense matrix and then to a DataFrame
feature_names = vectorizer.get_feature_names_out()
#print(feature_names)
feature_matrix_df = pd.DataFrame(X.toarray(), columns=feature_names)
# Print the DataFrame or send to an csv file if needed
#print(feature_matrix_df)
#output_file = "ngram_feature_matrix.csv"
#feature_matrix_df.to_csv(output_file, index=False)
#print(f"Feature matrix saved to {output_file}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
# Train the SVM classifier
svm_classifier = SVC(kernel='rbf',C=1.0,class_weight='balanced')
svm_classifier.fit(X_train, y_train)
# Make predictions on the test set
y_pred = svm_classifier.predict(X_test)
#print(X_train)
print(y_pred)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
# Print the evaluation metrics
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')

#=========================================================
# Cross_Validation and LOOCV
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score, LeaveOneOut

#model = SVC(kernel='linear',C=1.0,class_weight='balanced')


#kfold = KFold(n_splits=6, shuffle=True, random_state=42)
#loo = LeaveOneOut()

# Perform cross-validation
#scores = cross_val_score(svm_classifier, X_train_ngrams, y_train, cv=kfold)
#scores = cross_val_score(svm_classifier, X_train_ngrams, y_train, cv=loo)

# Print the cross-validation scores
#print("Cross-validation scores:", scores)
#print("Mean CV accuracy:", np.mean(scores))
