#### NMF 
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error

# Load the CSV files
users_df = pd.read_csv('users.csv')
interactions_df = pd.read_csv('interactions.csv')
jobs_df = pd.read_csv('jobs.csv')

# Create a user-job interaction matrix (user_id as rows and job_id as columns)
interaction_matrix = interactions_df.pivot(index='user_id', columns='job_id', values='interaction_score')

# Normalize the matrix (optional but can improve results)
normalized_matrix = normalize(interaction_matrix.fillna(0), axis=0)

# Create a function to calculate RMSE
def rmse(true_values, predicted_values):
    return np.sqrt(mean_squared_error(true_values, predicted_values))

# --- NMF --- 
def nmf_recommendation():
    nmf = NMF(n_components=3, random_state=42)
    user_topic_matrix = nmf.fit_transform(normalized_matrix)
    job_topic_matrix = nmf.components_

    # Reconstruct the matrix
    reconstructed_matrix_nmf = np.dot(user_topic_matrix, job_topic_matrix)

    return rmse(normalized_matrix, reconstructed_matrix_nmf), reconstructed_matrix_nmf

# Function to generate job recommendations for a user
def generate_recommendations(user_id, reconstructed_matrix, interactions_df):
    user_index = interaction_matrix.index.get_loc(user_id)
    predicted_scores = reconstructed_matrix[user_index]
    
    # Create a DataFrame for recommended jobs with predicted scores
    recommended_jobs = pd.DataFrame({
        'job_id': interaction_matrix.columns,
        'predicted_score': predicted_scores
    })
    
    # Sort by predicted score and return top 5 recommendations
    recommended_jobs = recommended_jobs.sort_values(by='predicted_score', ascending=False).head(5)
    
    # Merge with job titles
    recommended_jobs = recommended_jobs.merge(jobs_df[['job_id', 'job_title']], on='job_id')
    
    return recommended_jobs

# Running NMF recommendation and displaying results
nmf_accuracy, nmf_recommendations = nmf_recommendation()

# Calculate RMSE-based accuracy
max_m = max(max(row) for row in normalized_matrix)
min_m = min(min(row) for row in normalized_matrix)
nrmse = max_m - min_m
accuracy = (1 - (nmf_accuracy / nrmse)) * 100

# Display recommendations for each user
for _, user in users_df.iterrows():
    user_id = user['user_id']
    recommended_jobs = generate_recommendations(user_id, nmf_recommendations, interactions_df)
    
    print(f"User {user_id} - Accuracy: {accuracy:.1f}%, RMSE: {nmf_accuracy:.16f}")
    print("Recommended Jobs:")
    print(recommended_jobs[['job_id', 'job_title', 'predicted_score']])
    print()
