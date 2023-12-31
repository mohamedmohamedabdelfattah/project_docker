import pandas as pd 
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC



df = pd.read_csv("Bank_Deposite_Model.csv")
print(df)

# define X , y
X = df.iloc[:, :-1]
y = df.iloc[:, -1]


# Split the data into training and testing sets (80:20 ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




#1- LogiticRegressioin

# Initialize and train the Logistic Regression model
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# Predict using the test data
y_pred_lr = lr.predict(X_test)

# Evaluate the model accuracy
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f'Accuracy (Logistic Regression): {accuracy_lr * 100:.2f} %')

# Print the classification report
print('\nClassification Report:\n', classification_report(y_test, y_pred_lr))

# Print the confusion matrix
print('\nConfusion Matrix:\n', confusion_matrix(y_test, y_pred_lr))


############################################################################################

# 2- K-Nearest Neighboor 

knn = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors (k) as needed

# Train the model
knn.fit(X_train, y_train)
# Predict using the test data
y_pred = knn.predict(X_test)

# Evaluate the model accuracy
accuracy= accuracy_score(y_test, y_pred)
print(f'Accuracy (K-Nearest Neighbor): {accuracy * 100:.2f} %')

# Print the classification report
print('\nClassification Report:\n', classification_report(y_test, y_pred))

# Print the confusion matrix
print('\nConfusion Matrix:\n', confusion_matrix(y_test, y_pred))


############################################################################################

# 3 -Decision Tree 

# Decision Tree model
clf = DecisionTreeClassifier(random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Predict using the test data
y_pred = clf.predict(X_test)


# Evaluate the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy (Decision Tree Classifier): {accuracy * 100:.2f} %')

# Print the classification report
print('\nClassification Report:\n', classification_report(y_test, y_pred))

# Print the confusion matrix
print('\nConfusion Matrix:\n', confusion_matrix(y_test, y_pred))


#####################################################################################33

# 4-Random Forest Classifier 

# Random Forest model
RF_clf = RandomForestClassifier(n_estimators=1000, random_state=42)

# Train the model
RF_clf.fit(X_train, y_train)
# Predict using the test data
y_pred = RF_clf.predict(X_test)

# Evaluate the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy (Random Forest Classifier): {accuracy * 100:.2f} %')

# Print the classification report
print('\nClassification Report:\n', classification_report(y_test, y_pred))

# Print the confusion matrix
print('\nConfusion Matrix:\n', confusion_matrix(y_test, y_pred))


# 5- Liner_svc_classifier

# Initialize and train the Linear Support Vector Machine (SVM) classifier
linear_svc_classifier = LinearSVC()
linear_svc_classifier.fit(X_train, y_train)

# Predict using the test data
y_pred_linear_svc = linear_svc_classifier.predict(X_test)

# Evaluate the model accuracy
accuracy_linear_svc = accuracy_score(y_test, y_pred_linear_svc)
print(f'Accuracy (Linear SVM): {accuracy_linear_svc * 100:.2f} %')

# Print the classification report
print('\nClassification Report (Linear SVM):\n', classification_report(y_test, y_pred_linear_svc))

# Print the confusion matrix
print('\nConfusion Matrix (Linear SVM):\n', confusion_matrix(y_test, y_pred_linear_svc))

