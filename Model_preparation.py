# NAIVE BAYES ALGORITHM

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB  #we are using GaussianNB from Naive Bayes to make our model
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Function to read sequences from a file
def read_sequences(file_path):
    with open(file_path, 'r') as file:
        sequences = file.readlines()
    return sequences

# Path to your file
file_path = '/content/sequence (1).fasta'

# Read sequences from file
samples = read_sequences(file_path)

# Create a tokenizer, configured to only take into account the 1000 most common words
tokenizer = Tokenizer(num_words=1000)

# Build the word index
tokenizer.fit_on_texts(samples)

# Turn strings into lists of integer indices
sequences = tokenizer.texts_to_sequences(samples)

# You could also directly get the one-hot binary representations.
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

# Convert to float
X = one_hot_results.astype(float)

# Convert categorical variable into dummy/indicator variables.
le = LabelEncoder()
y = le.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize the GaussianNB classifier
clf = GaussianNB()

# Fit the model with the training data
clf.fit(X_train, y_train)

# Predict the target variable for the test data
y_pred = clf.predict(X_test)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Set random seed for reproducibility
np.random.seed(45)

# Plot the confusion matrix using seaborn
sns.set(font_scale=1.2)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

# SVM algorithm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Use Support Vector Machines from scikit-learn
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Function to read sequences from a file
def read_sequences(file_path):
    with open(file_path, 'r') as file:
        sequences = file.readlines()
    return sequences

# Path to your file
file_path = '/content/sequence (1).fasta'

# Read sequences from file
samples = read_sequences(file_path)

# Create a tokenizer, configured to only take into account the 1000 most common words
tokenizer = Tokenizer(num_words=1000)

# Build the word index
tokenizer.fit_on_texts(samples)

# Turn strings into lists of integer indices
sequences = tokenizer.texts_to_sequences(samples)

# You could also directly get the one-hot binary representations.
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

# Convert to float
X = one_hot_results.astype(float)

# Convert categorical variable into dummy/indicator variables.
le = LabelEncoder()
y = le.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

# Initialize the SVM classifier
clf = SVC(kernel='linear')  # You can choose different kernels like 'linear', 'rbf', 'poly', etc.


# Fit the model with the training data
clf.fit(X_train, y_train)

# Predict the target variable for the test data
y_pred = clf.predict(X_test)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Set random seed for reproducibility
np.random.seed(30)

# Plot the confusion matrix using seaborn
sns.set(font_scale=1.2)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)


# RANDOM FOREST CLASSIFIER
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Use Random Forest Classifier from scikit-learn
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Function to read sequences from a file
def read_sequences(file_path):
    with open(file_path, 'r') as file:
        sequences = file.readlines()
    return sequences

# Path to your file
file_path = '/content/sequence (1).fasta'

# Read sequences from file
samples = read_sequences(file_path)

# Create a tokenizer, configured to only take into account the 1000 most common words
tokenizer = Tokenizer(num_words=1000)

# Build the word index
tokenizer.fit_on_texts(samples)

# Turn strings into lists of integer indices
sequences = tokenizer.texts_to_sequences(labels)

# You could also directly get the one-hot binary representations.
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

# Convert to float
X = one_hot_results.astype(float)

# Convert categorical variable into dummy/indicator variables.
le = LabelEncoder()
y = le.fit_transform(samples)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

# Initialize the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=50, random_state=2)  # You can adjust the number of estimators as needed

# Fit the model with the training data
clf.fit(X_train, y_train)

# Predict the target variable for the test data
y_pred = clf.predict(X_test)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Set random seed for reproducibility
np.random.seed(30)

# Plot the confusion matrix using seaborn
sns.set(font_scale=1.2)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)


# KNN ALGORITHM
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.text import Tokenizer

# Function to read sequences from a file
def read_sequences(file_path):
    with open(file_path, 'r') as file:
        sequences = file.readlines()
    return sequences

# Path to your file
file_path = '/content/sequence (1).fasta'

# Read sequences from file
samples = read_sequences(file_path)

# Create a tokenizer, configured to only take into account the 1000 most common words
tokenizer = Tokenizer(num_words=1000)

# Build the word index
tokenizer.fit_on_texts(samples)

# Turn strings into lists of integer indices
sequences = tokenizer.texts_to_sequences(samples)

# You could also directly get the one-hot binary representations.
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

# Convert to float
X = one_hot_results.astype(float)

# Convert categorical variable into dummy/indicator variables.
le = LabelEncoder()
y = le.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Optimize the Choice of 'k'
param_grid = {'n_neighbors': range(1, 20)}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

best_k = grid_search.best_params_['n_neighbors']
clf = KNeighborsClassifier(n_neighbors=best_k)

# Fit the model with the training data
clf.fit(X_train_scaled, y_train)

# Predict the target variable for the test data
y_pred = clf.predict(X_test_scaled)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Set random seed for reproducibility
np.random.seed(45)

# Plot the confusion matrix using seaborn
sns.set(font_scale=1.2)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)
