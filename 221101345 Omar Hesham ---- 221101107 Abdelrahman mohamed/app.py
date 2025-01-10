import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

# Load data
@st.cache_data  # Cache data to improve performance
def load_data():
    # Define file paths (update these paths as needed)
    ratings_file_path = r'D:\Downloads\ml-10m\ml-10M100K\ratings.dat'
    tags_file_path = r'D:\Downloads\ml-10m\ml-10M100K\tags.dat'
    movies_file_path = r'D:\Downloads\ml-10m\ml-10M100K\movies.dat'

    # Load data into DataFrames
    ratings = pd.read_csv(ratings_file_path, sep="::", engine="python", names=["UserID", "MovieID", "Rating", "Timestamp"])
    tags = pd.read_csv(tags_file_path, sep="::", engine="python", names=["UserID", "MovieID", "Tag", "Timestamp"])
    movies = pd.read_csv(movies_file_path, sep="::", engine="python", names=["MovieID", "Title", "Genres"])

    return ratings, tags, movies

# Preprocess data
def preprocess_data(ratings, tags, movies):
    # Remove duplicates
    ratings = ratings.drop_duplicates()
    tags = tags.drop_duplicates()
    movies = movies.drop_duplicates()

    # Convert timestamps to datetime
    ratings['Timestamp'] = pd.to_datetime(ratings['Timestamp'], unit='s')
    tags['Timestamp'] = pd.to_datetime(tags['Timestamp'], unit='s')

    # Create user-item matrix
    user_item_matrix = ratings.pivot(index='UserID', columns='MovieID', values='Rating').fillna(0)

    # Create user-tag matrix
    user_tag_matrix = tags.groupby(['UserID', 'Tag']).size().unstack(fill_value=0)

    # Apply SVD for dimensionality reduction
    svd = TruncatedSVD(n_components=100, random_state=42)
    svd_matrix = svd.fit_transform(user_item_matrix)

    return user_item_matrix, user_tag_matrix, svd_matrix, movies

# Get popular movies
def get_popular_movies(ratings, movies, n=10):
    # Calculate the average rating for each movie
    avg_ratings = ratings.groupby('MovieID')['Rating'].mean()
    num_ratings = ratings.groupby('MovieID')['Rating'].count()

    # Sort movies by average rating (descending) and number of ratings (descending)
    popular_movies = avg_ratings.sort_values(ascending=False).head(n)

    # Get movie titles for the top N popular movies
    movie_titles = movies[movies['MovieID'].isin(popular_movies.index)]['Title'].values
    return pd.Series(movie_titles)

# Tag-based recommendation
def tag_based_recommendation(user_id, user_tag_matrix, ratings, movies, n_recommendations=10):
    # Initialize k-NN for tag-based recommendation
    knn = NearestNeighbors(n_neighbors=n_recommendations, metric='cosine')
    knn.fit(user_tag_matrix.values)

    # Find the nearest neighbors based on user_tag_profiles
    distances, indices = knn.kneighbors(user_tag_matrix.loc[user_id].values.reshape(1, -1), n_neighbors=n_recommendations)

    # Get movie recommendations based on the nearest neighbors' ratings
    recommended_movie_ids = np.array([])  # To hold the recommended movie IDs
    for idx in indices.flatten():
        similar_user_ratings = ratings[ratings['UserID'] == user_tag_matrix.index[idx]]
        recommended_movie_ids = np.append(recommended_movie_ids, similar_user_ratings['MovieID'].values)

    # Deduplicate and return top N recommended movies
    recommended_movie_ids = np.unique(recommended_movie_ids)
    recommended_movie_titles = movies.loc[movies['MovieID'].isin(recommended_movie_ids), 'Title'].values
    return pd.Series(recommended_movie_titles[:n_recommendations])

# Collaborative filtering-based recommendation
def cf_based_recommendation(user_id, svd_matrix, movies, n_recommendations=10):
    # Predict ratings using SVD
    cf_scores = np.dot(svd_matrix[user_id, :], svd.components_)

    # Recommend top N movies based on predicted ratings
    recommended_movie_ids = np.argsort(cf_scores)[-n_recommendations:][::-1]

    # Convert movie IDs to movie titles
    recommended_movie_titles = movies.loc[movies['MovieID'].isin(recommended_movie_ids), 'Title'].values
    return pd.Series(recommended_movie_titles)

# Hybrid-based recommendation
def hybrid_based_recommendation(user_id, user_item_matrix, user_tag_matrix, svd_matrix, movies, n_recommendations=10):
    # Collaborative filtering scores
    cf_scores = np.dot(svd_matrix[user_id, :], svd.components_)

    # Tag-based scores using k-NN
    knn = NearestNeighbors(n_neighbors=n_recommendations, metric='cosine')
    knn.fit(user_tag_matrix.values)
    distances, indices = knn.kneighbors(user_tag_matrix.loc[user_id].values.reshape(1, -1), n_neighbors=n_recommendations)
    tag_scores = np.mean(user_item_matrix.iloc[indices.flatten(), :], axis=0)

    # Combine CF and tag-based scores
    cf_weight = 0.7
    tag_weight = 0.3
    hybrid_scores = cf_weight * cf_scores + tag_weight * tag_scores

    # Recommend top N movies based on hybrid scores
    recommended_movie_ids = np.argsort(hybrid_scores)[-n_recommendations:][::-1]

    # Convert movie IDs to movie titles
    recommended_movie_titles = movies.loc[movies['MovieID'].isin(recommended_movie_ids), 'Title'].values
    return pd.Series(recommended_movie_titles)

# Hybrid recommendation function
def hybrid_recommendation(user_id, user_item_matrix, user_tag_matrix, svd_matrix, ratings, movies, n_recommendations=10):
    # Case 1: Cold-start users (no ratings and no tags)
    if user_id not in user_item_matrix.index and user_id not in user_tag_matrix.index:
        st.warning(f"User {user_id} is a cold-start user. Recommending popular movies.")
        return get_popular_movies(ratings, movies, n_recommendations)

    # Case 2: User has only tag-based data (no ratings)
    elif user_id not in user_item_matrix.index and user_id in user_tag_matrix.index:
        st.warning(f"User {user_id} has no ratings. Using tag-based recommendation.")
        return tag_based_recommendation(user_id, user_tag_matrix, ratings, movies, n_recommendations)

    # Case 3: User has only ratings data (no tag-based data)
    elif user_id in user_item_matrix.index and user_id not in user_tag_matrix.index:
        st.warning(f"User {user_id} has no tags. Using collaborative filtering recommendation.")
        return cf_based_recommendation(user_id, svd_matrix, movies, n_recommendations)

    # Case 4: User has both ratings and tag-based data (use hybrid recommendation)
    elif user_id in user_item_matrix.index and user_id in user_tag_matrix.index:
        st.success(f"User {user_id} has both ratings and tags. Using hybrid recommendation.")
        return hybrid_based_recommendation(user_id, user_item_matrix, user_tag_matrix, svd_matrix, movies, n_recommendations)

# Streamlit app
def main():
    st.title("🎬 Movie Recommendation System")
    st.write("Welcome to Our Social Tagging Movie Recommendation System!")

    # Load data
    ratings, tags, movies = load_data()

    # Preprocess data
    user_item_matrix, user_tag_matrix, svd_matrix, movies = preprocess_data(ratings, tags, movies)

    # User input
    user_id = st.number_input("Enter your User ID:", min_value=1, max_value=1000000, value=1, step=1)
    n_recommendations = st.slider("Number of recommendations:", min_value=1, max_value=20, value=10)

    # Get recommendations
    if st.button("Get Recommendations"):
        recommendations = hybrid_recommendation(user_id, user_item_matrix, user_tag_matrix, svd_matrix, ratings, movies, n_recommendations)

        # Display recommendations
        st.subheader("Recommended Movies:")
        for i, movie in enumerate(recommendations, 1):
            st.write(f"{i}. {movie}")

# Run the app
if __name__ == "__main__":
    main()