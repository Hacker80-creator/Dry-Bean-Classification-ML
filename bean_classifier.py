# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split, learning_curve
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix
# import joblib

# # Load your CSV file into a DataFrame
# data = pd.read_csv(r'C:\Users\surij\OneDrive\Desktop\project_mush\project_mush\train_dataset.csv')

# # Split the data into features (X) and the target variable (y)
# X = data.drop(columns=['satisfaction'])
# y = data['satisfaction']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardize the features (scaling)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Create and train the Random Forest classifier
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_classifier.fit(X_train, y_train)

# # Save the trained model and scaler to files using joblib
# model_filename = 'rf_classifier_model.pkl'
# scaler_filename = 'scaler.pkl'
# joblib.dump(rf_classifier, model_filename)
# joblib.dump(scaler, scaler_filename)

# # Make predictions
# y_pred = rf_classifier.predict(X_test)

# # Calculate accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy:.2f}')

# # Create a confusion matrix
# conf_matrix = confusion_matrix(y_test, y_pred)
# print('Confusion Matrix:')
# print(conf_matrix)

# # Plot the learning curve
# train_sizes, train_scores, test_scores = learning_curve(
#     rf_classifier, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)

# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)
# test_mean = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)

# plt.figure(figsize=(10, 6))
# plt.plot(train_sizes, train_mean, label='Training accuracy', marker='o')
# plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
# plt.plot(train_sizes, test_mean, label='Validation accuracy', marker='o')
# plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15)
# plt.xlabel('Number of Training Samples')
# plt.ylabel('Accuracy')
# plt.title('Learning Curve')
# plt.legend()
# plt.grid(True)
# plt.show()



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


























import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier  # Import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Load your CSV file into a DataFrame
data = pd.read_csv('train_dataset.csv')  # Replace 'your_dataset.csv' with the actual path to your dataset

# Split the data into features (X) and the target variable (y)
X = data.drop(columns=['Class'])  # Replace 'Class' with your target column name
y = data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Standardize the features (scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the K-Nearest Neighbors classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors (k) as needed
knn_classifier.fit(X_train, y_train)

# Save the trained model and scaler to files using joblib
model_filename = 'knn_classifier_model.pkl'
scaler_filename = 'scaler.pkl'
joblib.dump(knn_classifier, model_filename)
joblib.dump(scaler, scaler_filename)

# Make predictions
y_pred = knn_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)


















# Plot the learning curve
train_sizes, train_scores, test_scores = learning_curve(
    knn_classifier, X_train, y_train, cv=5, scoring='accuracy', n_jobs=1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training accuracy', marker='o')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
plt.plot(train_sizes, test_mean, label='Validation accuracy', marker='o')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15)
plt.xlabel('Number of Training Samples')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
plt.show()

# Define class labels (modify as needed)
class_labels = data['Class'].unique()  # Get unique class labels from your dataset






def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Define class labels (modify as needed)
class_labels = ['True', 'False']

# Plot the confusion matrix
plot_confusion_matrix(conf_matrix, class_labels)
