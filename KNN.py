#### KNN
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.metrics import mean_squared_error

# Load the datasets from CSV files
jobs_df = pd.read_csv('jobs.csv')
users_df = pd.read_csv('users.csv')
interactions_df = pd.read_csv('interactions.csv')

# Combine job skills and descriptions into a single column
jobs_df['Job_Skills_Description'] = jobs_df['skills'] + " " + jobs_df['location']

# Apply TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the job skills and descriptions to numerical vectors
job_vectors = vectorizer.fit_transform(jobs_df['Job_Skills_Description'])

# Function to recommend jobs based on user skills and job descriptions (cold start)
def recommend_jobs(user_skills, job_vectors, jobs_df, top_k=5):
    """
    Recommends jobs based on user skills and job descriptions using content-based filtering.
    """
    # Transform user skills into TF-IDF vector
    user_vector = vectorizer.transform([user_skills])

    # Compute cosine similarity between the user vector and all job vectors
    similarity_scores = cosine_similarity(user_vector, job_vectors)

    # Flatten the similarity scores array
    similarity_scores = similarity_scores.flatten()

    # Add similarity scores to the jobs DataFrame
    jobs_df['Similarity_Score'] = similarity_scores

    # Sort jobs by similarity score and return top-k recommended jobs (dropping duplicates)
    recommended_jobs = jobs_df.sort_values(by='Similarity_Score', ascending=False).drop_duplicates(subset='job_id').head(top_k)
    return recommended_jobs[['job_id', 'job_title', 'Similarity_Score']]

# Function to calculate accuracy and RMSE
def calculate_accuracy_rmse(user_id, recommended_jobs, interactions_df):
    """
    Calculate accuracy and RMSE for the recommendations.
    Accuracy: Percentage of recommended jobs that the user interacted with.
    RMSE: Mean Squared Error between the predicted and actual interaction scores.
    """
    # Merge recommended jobs with actual user interactions
    recommended_job_ids = recommended_jobs['job_id'].tolist()
    actual_interactions = interactions_df[interactions_df['user_id'] == user_id]

    # Get the actual interaction scores for the recommended jobs
    actual_scores = actual_interactions[actual_interactions['job_id'].isin(recommended_job_ids)]

    if len(actual_scores) == 0:
        return 0, 0  # No interaction found, return zero accuracy and RMSE

    # Calculate accuracy: Percentage of recommended jobs that the user interacted with
    matched_jobs = len(actual_scores)
    accuracy = (matched_jobs / len(recommended_jobs)) * 100

    # Calculate RMSE: Difference between predicted similarity scores and actual interaction scores
    predicted_scores = recommended_jobs.set_index('job_id').loc[actual_scores['job_id']]['Similarity_Score'].values
    actual_scores_values = actual_scores['interaction_score'].values
    rmse = np.sqrt(mean_squared_error(actual_scores_values, predicted_scores))

    return accuracy, rmse

# Function to recommend jobs for each user and calculate performance
def recommend_jobs_for_all_users(users_df, job_vectors, jobs_df, interactions_df, top_k=5):
    recommendations = []

    # Loop through each user to get job recommendations
    for _, user in users_df.iterrows():
        user_skills = user['user_skills']
        user_id = user['user_id']

        # Get recommendations for the specific user
        recommended_jobs = recommend_jobs(user_skills, job_vectors, jobs_df, top_k)

        # Calculate accuracy and RMSE for the recommended jobs
        accuracy, rmse = calculate_accuracy_rmse(user_id, recommended_jobs, interactions_df)

        # Store the recommendations along with accuracy and RMSE
        recommendations.append({
            'user_id': user_id,
            'recommended_jobs': recommended_jobs,
            'accuracy': accuracy,
            'rmse': rmse
        })

    return recommendations

# Example: Recommend jobs for all users and calculate accuracy and RMSE
recommendations = recommend_jobs_for_all_users(users_df, job_vectors, jobs_df, interactions_df)

# Display the results
for rec in recommendations:
    print(f"User {rec['user_id']} - Accuracy: {rec['accuracy']}%, RMSE: {rec['rmse']}")
    print("Recommended Jobs:")
    print(rec['recommended_jobs'])
    print()