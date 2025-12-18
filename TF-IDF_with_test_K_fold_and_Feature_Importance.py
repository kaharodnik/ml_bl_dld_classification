import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

 ##Training data
train_data = pd.read_csv('bl_sli_4_6_balanced_translit_no_test.csv',encoding = 'utf-8')
X_train = train_data['text'].values
y_train = train_data['label'].values

#Test data
test_data = pd.read_csv('bl_sli_7_9_balanced_test_checked.csv', encoding = 'utf-8')
X_test = test_data['text'].values
y_test = test_data['label'].values
print (test_data.columns)

#Create TF-IDF vectors
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform the test data
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# SVM classification

clf = SVC(kernel='rbf',C=1,gamma="scale",class_weight='balanced',random_state=42,probability=True)

# Train the model on the training set
clf.fit(X_train_tfidf, y_train)

# Predict the labels for the test set
y_pred = clf.predict(X_test_tfidf)

# Evaluate the model on the test set
accuracy = accuracy_score(y_test, y_pred)
print("Test set accuracy: ", accuracy)

# Print a detailed classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Print the confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

#============= Cross_Validation ==================
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score, LeaveOneOut
import numpy as np

model = SVC(kernel='linear',C=1.0,class_weight='balanced')

# Define the number of folds for cross-validation
#StratifiedKFold ensures each fold has roughly the same proportion of 0s and 1s
kfold = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
#loo = LeaveOneOut()

# Perform cross-validation
scores = cross_val_score(clf, X_train_tfidf, y_train, cv=kfold)
#scores = cross_val_score(clf, X_train_tfidf, y_train, cv=loo)

# Print the cross-validation scores
print("Cross-validation scores:", scores)
print("Mean CV accuracy:", np.mean(scores))

#=========Feature Importance Scores==========
from sklearn.ensemble import RandomForestClassifier

# Random Forest
clf = RandomForestClassifier(random_state=42)
clf.fit(X_test_tfidf, y_test)

# Feature importance
feature_importance = pd.DataFrame(clf.feature_importances_, index=tfidf_vectorizer.get_feature_names_out(), columns=['Importance'])
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

print(feature_importance.head(10))

#====Plot 10 most important features =========

# Plot feature importance
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
feature_importance.head(10).plot(kind='bar')
plt.title('Top 10 Feature Importances, test set, POS TF-IDF, BL TD and ML DLD, 7-9 years old')
plt.ylabel('Importance Score')
plt.xlabel('Features')
plt.show()
