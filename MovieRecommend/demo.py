#!/usr/bin/env python3
"""
Demo script for the Movie Recommender System
This script demonstrates the core functionality without the Streamlit UI
"""

import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    """Load the movies data"""
    movies_list = pickle.load(open('movies.pkl','rb'))
    return movies_list

def create_similarity_matrix():
    """Create similarity matrix for recommendations"""
    movies_list = load_data()
    
    # Create CountVectorizer to convert text to vectors
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies_list['tags']).toarray()
    
    # Calculate cosine similarity
    similarity = cosine_similarity(vectors)
    return similarity, movies_list

def recommend_movies(movie_title, similarity_matrix, movies_list, num_recommendations=5):
    """Get movie recommendations"""
    try:
        # Find the index of the selected movie
        movie_index = movies_list[movies_list['title'] == movie_title].index[0]
        
        # Get similarity scores for this movie
        distances = similarity_matrix[movie_index]
        
        # Get top similar movies (excluding the movie itself)
        movies_list_sorted = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:num_recommendations+1]
        
        recommended_movies = []
        for i in movies_list_sorted:
            recommended_movies.append({
                'title': movies_list.iloc[i[0]]['title'],
                'similarity_score': i[1],
                'movie_id': movies_list.iloc[i[0]]['movie_id']
            })
        
        return recommended_movies
    except IndexError:
        return []

def main():
    print("🎬 Movie Recommender System Demo")
    print("=" * 50)
    
    # Load data and create similarity matrix
    print("Loading data and creating similarity matrix...")
    similarity_matrix, movies_list = create_similarity_matrix()
    print(f"✅ Loaded {len(movies_list)} movies")
    
    # Show some sample movies
    print("\n📋 Sample Movies in Dataset:")
    sample_movies = movies_list['title'].head(10).tolist()
    for i, movie in enumerate(sample_movies, 1):
        print(f"{i:2d}. {movie}")
    
    # Demo recommendations
    print("\n🎯 Demo Recommendations:")
    print("-" * 30)
    
    # Test with a popular movie
    test_movie = "Avatar"
    print(f"\n🔍 Finding movies similar to '{test_movie}'...")
    
    recommendations = recommend_movies(test_movie, similarity_matrix, movies_list, 5)
    
    if recommendations:
        print(f"\n✨ Top 5 movies similar to '{test_movie}':")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['title']} (Similarity: {rec['similarity_score']:.3f})")
    else:
        print(f"❌ No recommendations found for '{test_movie}'")
    
    # Test with another movie
    test_movie2 = "The Dark Knight"
    print(f"\n🔍 Finding movies similar to '{test_movie2}'...")
    
    recommendations2 = recommend_movies(test_movie2, similarity_matrix, movies_list, 3)
    
    if recommendations2:
        print(f"\n✨ Top 3 movies similar to '{test_movie2}':")
        for i, rec in enumerate(recommendations2, 1):
            print(f"{i}. {rec['title']} (Similarity: {rec['similarity_score']:.3f})")
    else:
        print(f"❌ No recommendations found for '{test_movie2}'")
    
    print("\n" + "=" * 50)
    print("🎉 Demo completed! Run 'streamlit run app.py' to use the full web interface.")
    print("=" * 50)

if __name__ == "__main__":
    main()

