#Random Forest 

# 1. Importing Necessary Libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



# 2. Loading and Preprocessing the Iris Dataset
# Load the Iris dataset

# data = pd.read_csv('data.csv')

# Display the first few rows and check for any missing values
# print("First 5 rows of the dataset:\n", data.head())
# print("\nShape of the dataset:", data.shape)

# Extract features (assuming all columns except the last one are features)
# X = data.iloc[:, :-1].values


# 2. Loading and Preprocessing the Iris Dataset
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Display the shape of the dataset
print("Shape of feature data:", X.shape)
print("Shape of target data:", y.shape)

# 3. Constructing the Random Forest Model
# Initialize Random Forest classifier
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
random_forest.fit(X_train, y_train)

# Predictions
y_pred = random_forest.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Random Forest Classifier: {accuracy:.4f}")

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))


#SVC


# 1. Importing Necessary Libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Loading and Preprocessing the Iris Dataset
# Load the Iris dataset

# data = pd.read_csv('data.csv')

# Display the first few rows and check for any missing values
# print("First 5 rows of the dataset:\n", data.head())
# print("\nShape of the dataset:", data.shape)

# Extract features (assuming all columns except the last one are features)
# X = data.iloc[:, :-1].values

# 2. Loading and Exploring the Dataset
iris = load_iris()
X = iris.data
y = iris.target

print(f"Shape of feature data: {X.shape}")
print(f"Shape of target data: {y.shape}")
print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)
print("\nFirst 5 rows of feature data:\n", X[:5])
print("\nFirst 5 target values:\n", y[:5])

# 3. Splitting the Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# 4. Training the Support Vector Machine
svm_clf = SVC(kernel='linear')
svm_clf.fit(X_train, y_train)

# 5. Evaluating the Model
y_pred = svm_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the SVM classifier: {accuracy * 100:.2f}%")

# 6. Visualizing the Decision Boundaries (for 2D)
def plot_decision_boundaries(X, y, model, title):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.show()

X_vis = X[:, :2]
X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(X_vis, y, test_size=0.3, random_state=42)
svm_clf_vis = SVC(kernel='linear')
svm_clf_vis.fit(X_train_vis, y_train_vis)

plot_decision_boundaries(X_train_vis, y_train_vis, svm_clf_vis, "SVM Decision Boundaries (Training Set)")
plot_decision_boundaries(X_test_vis, y_test_vis, svm_clf_vis, "SVM Decision Boundaries (Testing Set)")


#KNN

# 1. Importing Necessary Libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 2. Loading and Preprocessing the Iris Dataset
# Load the Iris dataset

# data = pd.read_csv('data.csv')

# Display the first few rows and check for any missing values
# print("First 5 rows of the dataset:\n", data.head())
# print("\nShape of the dataset:", data.shape)

# Extract features (assuming all columns except the last one are features)
# X = data.iloc[:, :-1].values

# 2. Loading and Exploring the Dataset
iris = load_iris()
X = iris.data
y = iris.target

print(f"Shape of feature data: {X.shape}")
print(f"Shape of target data: {y.shape}")
print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)
print("\nFirst 5 rows of feature data:\n", X[:5])
print("\nFirst 5 target values:\n", y[:5])

# 3. Splitting the Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# 4. Training the k-Nearest Neighbors Classifier
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)

# 5. Evaluating the Model and Printing Predictions
y_pred = knn_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the k-NN classifier: {accuracy * 100:.2f}%")

correct_predictions = []
wrong_predictions = []

for i in range(len(y_test)):
    if y_test[i] == y_pred[i]:
        correct_predictions.append((X_test[i], y_test[i], y_pred[i]))
    else:
        wrong_predictions.append((X_test[i], y_test[i], y_pred[i]))

print("\nCorrect Predictions:")
for sample, true_label, predicted_label in correct_predictions:
    print(f"Sample: {sample}, True Label: {true_label}, Predicted Label: {predicted_label}")

print("\nWrong Predictions:")
for sample, true_label, predicted_label in wrong_predictions:
    print(f"Sample: {sample}, True Label: {true_label}, Predicted Label: {predicted_label}")


#frozen lake

import numpy as np
import gym

# Initialize the FrozenLake environment
env = gym.make('FrozenLake-v1', is_slippery=False)

# Set up Q-table with all zeros
Q = np.zeros((env.observation_space.n, env.action_space.n))

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.99
min_epsilon = 0.01
episodes = 1000

# Training loop
for _ in range(episodes):
    state = env.reset()
    done = False
    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        next_state, reward, done, _ = env.step(action)
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        state = next_state
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

# Test the trained agent
state = env.reset()
done = False
while not done:
    action = np.argmax(Q[state])
    state, reward, done, _ = env.step(action)
    env.render()

print("Test finished.")

#WEB

from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask application
app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        data = request.get_json(force=True)
        
        # Preprocess data if necessary
        # Example: Convert data to numpy array
        input_data = np.array(data['input'])
        
        # Make predictions
        predictions = model.predict(input_data)
        
        # Prepare response
        response = {'predictions': predictions.tolist()}
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Run the application
    app.run(debug=True)

#ID3 


# 1. Importing Necessary Libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# 2. Loading and Preprocessing the Iris Dataset
# Load the Iris dataset

# data = pd.read_csv('data.csv')

# Display the first few rows and check for any missing values
# print("First 5 rows of the dataset:\n", data.head())
# print("\nShape of the dataset:", data.shape)

# Extract features (assuming all columns except the last one are features)
# X = data.iloc[:, :-1].values

# 2. Loading and Exploring the Dataset
iris = load_iris()
X = iris.data
y = iris.target

print(f"Shape of feature data: {X.shape}")
print(f"Shape of target data: {y.shape}")
print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)
print("\nFirst 5 rows of feature data:\n", X[:5])
print("\nFirst 5 target values:\n", y[:5])

# 3. Splitting the Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# 4. Training the Decision Tree Classifier Using ID3 Algorithm
clf = DecisionTreeClassifier(criterion="entropy")
clf.fit(X_train, y_train)

# 5. Evaluating the Model
y_pred = clf.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy of the decision tree classifier: {accuracy * 100:.2f}%")

# 6. Visualizing the Decision Tree
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree using ID3 Algorithm (Iris Dataset)")
plt.show()

# 7. Classifying a New Sample
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example sample from the Setosa class
new_prediction = clf.predict(new_sample)
print(f"Predicted class for the new sample: {iris.target_names[new_prediction][0]}")

#Naive Bayes

# 1. Importing Necessary Libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# 2. Loading and Preprocessing the Iris Dataset
# Load the Iris dataset

# data = pd.read_csv('data.csv')

# Display the first few rows and check for any missing values
# print("First 5 rows of the dataset:\n", data.head())
# print("\nShape of the dataset:", data.shape)

# Extract features (assuming all columns except the last one are features)
# X = data.iloc[:, :-1].values

# 2. Loading and Exploring the Dataset
iris = load_iris()
X = iris.data
y = iris.target

print(f"Shape of feature data: {X.shape}")
print(f"Shape of target data: {y.shape}")
print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)
print("\nFirst 5 rows of feature data:\n", X[:5])
print("\nFirst 5 target values:\n", y[:5])

# 3. Splitting the Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# 4. Training the Naive Bayes Classifier
nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)

# 5. Evaluating the Model
y_pred = nb_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Naive Bayes classifier: {accuracy * 100:.2f}%")

# 6. Printing Predictions
correct_predictions = []
wrong_predictions = []

for i in range(len(y_test)):
    if y_test[i] == y_pred[i]:
        correct_predictions.append((X_test[i], y_test[i], y_pred[i]))
    else:
        wrong_predictions.append((X_test[i], y_test[i], y_pred[i]))

print("\nCorrect Predictions:")
for sample, true_label, predicted_label in correct_predictions:
    print(f"Sample: {sample}, True Label: {true_label}, Predicted Label: {predicted_label}")

print("\nWrong Predictions:")
for sample, true_label, predicted_label in wrong_predictions:
    print(f"Sample: {sample}, True Label: {true_label}, Predicted Label: {predicted_label}")



#Adaboosting

# 1. Importing Necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 2. Loading and Preprocessing the Iris Dataset
# Load the Iris dataset

# data = pd.read_csv('data.csv')

# Display the first few rows and check for any missing values
# print("First 5 rows of the dataset:\n", data.head())
# print("\nShape of the dataset:", data.shape)

# Extract features (assuming all columns except the last one are features)
# X = data.iloc[:, :-1].values


# 2. Loading and Preprocessing the Iris Dataset
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Display the shape of the dataset
print("Shape of feature data:", X.shape)
print("Shape of target data:", y.shape)

# 3. Implementing AdaBoost Classifier
# Initialize AdaBoost classifier without base_estimator (default is DecisionTreeClassifier)
adaboost_clf = AdaBoostClassifier(n_estimators=50, random_state=42)

# Train the AdaBoost classifier
adaboost_clf.fit(X_train, y_train)

# Predictions
y_pred = adaboost_clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of AdaBoost Classifier: {accuracy:.4f}")

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))


#Bayesian NEtwork


# 1. Importing Necessary Libraries
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import VariableElimination

# 2. Loading and Preprocessing the Iris Dataset
# Load the Iris dataset

# data = pd.read_csv('data.csv')

# Display the first few rows and check for any missing values
# print("First 5 rows of the dataset:\n", data.head())
# print("\nShape of the dataset:", data.shape)

# Extract features (assuming all columns except the last one are features)
# X = data.iloc[:, :-1].values


# 2. Loading and Preparing the Dataset
# Sample dataset
data = pd.DataFrame(data={
    'A': ['T', 'T', 'F', 'T', 'F', 'F', 'T', 'F', 'T', 'T'],
    'B': ['T', 'T', 'F', 'F', 'T', 'T', 'F', 'F', 'T', 'T'],
    'C': ['F', 'T', 'F', 'T', 'F', 'F', 'T', 'F', 'F', 'T'],
    'D': ['T', 'T', 'F', 'T', 'F', 'T', 'T', 'F', 'T', 'F'],
    'E': ['T', 'T', 'F', 'T', 'F', 'T', 'F', 'F', 'T', 'F']
})

# 3. Defining the Bayesian Network Structure
model = BayesianNetwork([('A', 'C'), ('B', 'C'), ('C', 'D'), ('D', 'E')])

# 4. Training the Bayesian Network
model.fit(data, estimator=MaximumLikelihoodEstimator)

# 5. Making Predictions
infer = VariableElimination(model)

# Query the probability of different variables
result = infer.query(variables=['E'], evidence={'A': 'T', 'B': 'F', 'C': 'T', 'D': 'T'})
print(result)


#Ensemble Learning 


# 1. Importing Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

# 2. Loading and Preprocessing the Iris Dataset
# Load the Iris dataset

# data = pd.read_csv('data.csv')

# Display the first few rows and check for any missing values
# print("First 5 rows of the dataset:\n", data.head())
# print("\nShape of the dataset:", data.shape)

# Extract features (assuming all columns except the last one are features)
# X = data.iloc[:, :-1].values



iris = load_iris()
X = iris.data
y = iris.target

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to DataFrame for easier manipulation (not necessary for clustering)
iris_df = pd.DataFrame(data=X_scaled, columns=iris.feature_names)

# Display the first few rows of the dataset
print("First 5 rows of the Iris dataset:\n", iris_df.head())

# 3. Applying k-Means Algorithm
# Initialize and fit k-Means cluster
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Get cluster labels
kmeans_labels = kmeans.labels_

# Plotting k-Means results
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels, cmap='viridis', edgecolor='k')
plt.title('k-Means Clustering')
plt.xlabel('Sepal Length (scaled)')
plt.ylabel('Sepal Width (scaled)')
plt.show()

# 4. Applying EM Algorithm (Gaussian Mixture Model)
# Initialize and fit Gaussian Mixture Model (EM algorithm)
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X_scaled)

# Get cluster labels
gmm_labels = gmm.predict(X_scaled)

# Plotting EM algorithm results
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=gmm_labels, cmap='viridis', edgecolor='k')
plt.title('EM Algorithm (Gaussian Mixture Model)')
plt.xlabel('Sepal Length (scaled)')
plt.ylabel('Sepal Width (scaled)')
plt.show()

# 5. Comparing the Results
# Compare the results using Adjusted Rand Index (ARI)
ari_kmeans = adjusted_rand_score(y, kmeans_labels)
ari_gmm = adjusted_rand_score(y, gmm_labels)

print(f"Adjusted Rand Index (ARI) between k-Means and true labels: {ari_kmeans:.4f}")
print(f"Adjusted Rand Index (ARI) between EM Algorithm and true labels: {ari_gmm:.4f}")

