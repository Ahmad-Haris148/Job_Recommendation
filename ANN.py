import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Load datasets (jobs, users, interactions)
jobs_df = pd.read_csv("jobs.csv")  # Jobs dataset (contains job_id, job_title, skills, location)
users_df = pd.read_csv("users.csv")  # Users dataset (contains user_id, user_skills, user_location)
interactions_df = pd.read_csv("interactions.csv")  # Interactions dataset (contains user_id, job_id, interaction_score)

# Step 2: Merge datasets based on user-job interactions
merged_data = pd.merge(interactions_df, jobs_df, on='job_id', how='inner')  # Merging with job details
merged_data = pd.merge(merged_data, users_df, on='user_id', how='inner')  # Merging with user details

# Step 3: Preprocessing the data
label_encoder = LabelEncoder()

# Encode job titles (for simplicity, converting categorical values to integers)
merged_data['job_title_encoded'] = label_encoder.fit_transform(merged_data['job_title'])

# Encode skills (user and job skills) as simple count of skills for each (split by comma)
merged_data['skills_encoded'] = merged_data['skills'].apply(lambda x: len(x.split(',')))  # Count number of skills in the job
merged_data['user_skills_encoded'] = merged_data['user_skills'].apply(lambda x: len(x.split(',')))  # Count number of skills for user

# Encode locations (job and user locations)
merged_data['job_location_encoded'] = label_encoder.fit_transform(merged_data['location'])
merged_data['user_location_encoded'] = label_encoder.fit_transform(merged_data['user_location'])

# Step 4: Prepare the feature set (X) and target variable (Y)
X = merged_data[['job_title_encoded', 'skills_encoded', 'user_skills_encoded', 'job_location_encoded', 'user_location_encoded']]  # Features

# Target: interaction_score (assuming we want to predict how likely a user will interact with a job)
Y = merged_data['interaction_score']  # Target variable

# Step 5: Normalize the feature set (important for ANN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Normalize features to have zero mean and unit variance

# Step 6: Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Initialize parameters (weights) for a simple ANN
input_dim = X_train.shape[1]  # Number of input features
hidden_dim = 64  # Number of neurons in hidden layer
output_dim = 1  # Single output (interaction score prediction)

# Randomly initialize weights
np.random.seed(42)
W1 = np.random.randn(input_dim, hidden_dim)  # Weights from input to hidden layer
b1 = np.zeros((1, hidden_dim))  # Bias for hidden layer
W2 = np.random.randn(hidden_dim, output_dim)  # Weights from hidden to output layer
b2 = np.zeros((1, output_dim))  # Bias for output layer

# Activation functions
def relu(x):
    return np.maximum(0, x)  # ReLU activation function

def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # Sigmoid activation function (for output layer)

def sigmoid_derivative(x):
    return x * (1 - x)  # Derivative of sigmoid function for backpropagation

# Training Hyperparameters
epochs = 1000
learning_rate = 0.001

# Step 7: Training the network
for epoch in range(epochs):
    # Forward Propagation
    Z1 = np.dot(X_train, W1) + b1  # Linear transformation from input to hidden layer
    A1 = relu(Z1)  # Apply ReLU activation (hidden layer)
    Z2 = np.dot(A1, W2) + b2  # Linear transformation from hidden to output layer
    A2 = sigmoid(Z2)  # Apply sigmoid activation (output layer)

    # Compute the error (Loss)
    error = A2 - Y_train.values.reshape(-1, 1)  # Error is the difference between prediction and true value

    # Backward Propagation
    dZ2 = error * sigmoid_derivative(A2)  # Derivative of sigmoid for output layer
    dW2 = np.dot(A1.T, dZ2)  # Gradient of weights from hidden to output layer
    db2 = np.sum(dZ2, axis=0, keepdims=True)  # Gradient of biases for output layer

    dA1 = np.dot(dZ2, W2.T)  # Gradient of hidden layer activations
    dZ1 = dA1 * (Z1 > 0)  # ReLU derivative (1 if Z1 > 0 else 0)
    dW1 = np.dot(X_train.T, dZ1)  # Gradient of weights from input to hidden layer
    db1 = np.sum(dZ1, axis=0, keepdims=True)  # Gradient of biases for hidden layer

    # Update weights using gradient descent
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    # Print the loss (Mean Squared Error) every 100 epochs
    if epoch % 100 == 0:
        mse = np.mean(np.square(error))  # Mean Squared Error loss
        print(f"Epoch {epoch}/{epochs}, MSE: {mse:.4f}")

# Step 8: Evaluate the model on test data
# Forward pass on the test set
Z1_test = np.dot(X_test, W1) + b1
A1_test = relu(Z1_test)
Z2_test = np.dot(A1_test, W2) + b2
A2_test = sigmoid(Z2_test)

# Calculate the Mean Squared Error on the test set
mse_test = mean_squared_error(Y_test, A2_test)
print(f"Test MSE: {mse_test:.4f}")

# Step 9: Making predictions for a new user profile
new_user_profile = np.array([[2, 4, 3, 1, 2]])  # Example: 2 job skills, 4 user skills, job location encoded=1, user location encoded=2
new_user_profile_scaled = scaler.transform(new_user_profile)  # Normalize the new profile

# Forward pass for the new user
Z1_new = np.dot(new_user_profile_scaled, W1) + b1
A1_new = relu(Z1_new)
Z2_new = np.dot(A1_new, W2) + b2
A2_new = sigmoid(Z2_new)

print(f"Predicted interaction score for the new user: {A2_new[0][0]:.4f}")
