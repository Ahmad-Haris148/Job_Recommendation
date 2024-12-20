import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

# Load the datasets
try:
    jobs_df = pd.read_csv('jobs.csv')
    users_df = pd.read_csv('users.csv')
    interactions_df = pd.read_csv('interactions.csv')
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# Combine job skills and descriptions into a single column for content-based filtering
jobs_df['skills'] = jobs_df['skills'].fillna('')  # Handle missing skills
jobs_df['location'] = jobs_df['location'].fillna('')  # Handle missing locations
jobs_df['Job_Skills_Description'] = jobs_df['skills'] + " " + jobs_df['location']

# Apply TF-IDF Vectorization for job descriptions
vectorizer = TfidfVectorizer(stop_words='english')
job_vectors = vectorizer.fit_transform(jobs_df['Job_Skills_Description'])

# Generate user-job interaction matrix
interaction_matrix = pd.pivot_table(
    interactions_df,
    index='user_id',
    columns='job_id',
    values='interaction_score',
    fill_value=0
)

# Objective function for GFO (optimize latent factors in NMF)
def objective_function(num_components):
    num_components = int(num_components)  # Convert to integer as NMF requires it
    nmf = NMF(n_components=num_components, random_state=42,max_iter=500, verbose=True)
    nmf_user = nmf.fit_transform(interaction_matrix)
    nmf_job = nmf.components_.T
    predictions = np.dot(nmf_user, nmf_job.T)

    # Calculate RMSE for the predictions
    actual = interaction_matrix.values.flatten()
    predicted = predictions.flatten()
    mask = actual > 0  # Only consider non-zero interactions
    rmse = np.sqrt(mean_squared_error(actual[mask], predicted[mask]))
    return rmse

# GFO Parameters
num_fish = 10            # Number of fish in the swarm
dimensions = 1           # Number of parameters to optimize
max_iterations = 20      # Maximum number of iterations
lb, ub = 2, 10           # Lower and upper bounds for latent factors

# Initialize population
population = np.random.uniform(lb, ub, (num_fish, dimensions))
fitness = np.array([objective_function(x[0]) for x in population])

# Grey Fish Optimization loop
for iteration in range(max_iterations):
    best_fitness = np.min(fitness)
    best_fish = population[np.argmin(fitness)]

    for i in range(num_fish):
        if np.random.rand() < 0.5:
            # Move towards the best fish
            direction = best_fish - population[i]
        else:
            # Random exploration
            direction = np.random.uniform(-1, 1, dimensions)

        # Update position
        step_size = np.random.rand()
        new_position = population[i] + step_size * direction
        new_position = np.clip(new_position, lb, ub)  # Clip to bounds

        # Evaluate fitness of the new position
        new_fitness = objective_function(new_position[0])
        if new_fitness < fitness[i]:
            population[i] = new_position
            fitness[i] = new_fitness

    print(f"Iteration {iteration+1}/{max_iterations}, Best RMSE: {best_fitness}")

# Use the best number of latent factors from GFO
optimal_components = int(best_fish[0])
print("Optimal number of components:", optimal_components)

# Final NMF model with optimized components
nmf = NMF(n_components=optimal_components, random_state=42)
nmf_user = nmf.fit_transform(interaction_matrix)
nmf_job = nmf.components_.T
nmf_predictions = np.dot(nmf_user, nmf_job.T)

# Function to recommend jobs
def recommend_jobs(user_skills, job_vectors, jobs_df, top_k=5):
    user_vector = vectorizer.transform([user_skills])
    similarity_scores = cosine_similarity(user_vector, job_vectors).flatten()

    jobs_df['Similarity_Score'] = similarity_scores
    recommended_jobs = jobs_df.sort_values(by='Similarity_Score', ascending=False).drop_duplicates(subset='job_id').head(top_k)
    return recommended_jobs[['job_id', 'job_title', 'Similarity_Score']]

# Function to calculate RMSE and accuracy
def calculate_accuracy_rmse(user_id, recommended_jobs, interactions_df):
    recommended_job_ids = recommended_jobs['job_id'].tolist()
    actual_interactions = interactions_df[interactions_df['user_id'] == user_id]
    actual_scores = actual_interactions[actual_interactions['job_id'].isin(recommended_job_ids)]

    if len(actual_scores) == 0:
        return 0, 0  # No interaction found

    matched_jobs = len(actual_scores)
    accuracy = (matched_jobs / len(recommended_jobs)) * 100

    predicted_scores = recommended_jobs.set_index('job_id').loc[actual_scores['job_id']]['Similarity_Score'].values
    actual_scores_values = actual_scores['interaction_score'].values
    rmse = np.sqrt(mean_squared_error(actual_scores_values, predicted_scores))

    return accuracy, rmse

# Recommend jobs for all users
def recommend_jobs_for_all_users(users_df, job_vectors, jobs_df, interactions_df, nmf_predictions, top_k=5):
    recommendations = []

    for _, user in users_df.iterrows():
        user_skills = user['user_skills']
        user_id = user['user_id']

        if user_id not in interaction_matrix.index:
            recommended_jobs = recommend_jobs(user_skills, job_vectors, jobs_df, top_k)
        else:
            user_index = interaction_matrix.index.get_loc(user_id)
            user_predictions = pd.Series(nmf_predictions[user_index], index=interaction_matrix.columns)
            top_n_jobs = user_predictions.sort_values(ascending=False).head(top_k)

            recommended_jobs = jobs_df[jobs_df['job_id'].isin(top_n_jobs.index)]
            recommended_jobs = recommended_jobs.merge(
                pd.DataFrame({'job_id': top_n_jobs.index, 'Similarity_Score': top_n_jobs.values}),
                on='job_id'
            )

        accuracy, rmse = calculate_accuracy_rmse(user_id, recommended_jobs, interactions_df)
        recommendations.append({
            'user_id': user_id,
            'recommended_jobs': recommended_jobs,
            'accuracy': accuracy,
            'rmse': rmse
        })

    return recommendations

# Run recommendations for all users
recommendations = recommend_jobs_for_all_users(users_df, job_vectors, jobs_df, interactions_df, nmf_predictions)

# Display recommendations
for rec in recommendations:
    print(f"User {rec['user_id']} - Accuracy: {rec['accuracy']}%, RMSE: {rec['rmse']}")
    print("Recommended Jobs:")
    print(rec['recommended_jobs'][['job_id', 'job_title', 'Similarity_Score']])
    print()