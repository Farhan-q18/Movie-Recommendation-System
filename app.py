import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_poster(movie_id, title=None, size='w342'):
    try:
        api_key = "821dbaa37545506755e488a1dfe93f56"
        response = requests.get(
            "https://api.themoviedb.org/3/movie/{}?api_key={}&language=en-US".format(movie_id, api_key),
            timeout=8
        )
        data = response.json()
        path = data.get('poster_path')
        if not path:
            # Fallback: try searching by title if provided
            if title:
                try:
                    search = requests.get(
                        'https://api.themoviedb.org/3/search/movie',
                        params={'api_key': api_key, 'query': title, 'language': 'en-US', 'include_adult': 'false'},
                        timeout=8
                    )
                    sdata = search.json()
                    results = sdata.get('results') or []
                    # Prefer exact (case-insensitive) match that has a poster
                    for res in results:
                        if res.get('poster_path') and (res.get('title', '').strip().lower() == title.strip().lower()):
                            pp = res['poster_path'].lstrip('/')
                            return f"https://image.tmdb.org/t/p/{size}/{pp}"
                    # Else, first result that has a poster
                    for res in results:
                        if res.get('poster_path'):
                            pp = res['poster_path'].lstrip('/')
                            return f"https://image.tmdb.org/t/p/{size}/{pp}"
                except Exception:
                    pass
            return "https://via.placeholder.com/300x450/667eea/ffffff?text=No+Poster"
        pp = str(path).lstrip('/')
        return f"https://image.tmdb.org/t/p/{size}/{pp}"
    except Exception:
        return "https://via.placeholder.com/300x450/667eea/ffffff?text=No+Poster"

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_image_bytes(url: str) -> bytes:
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.content
    except Exception:
        pass
    # fallback placeholder
    try:
        resp = requests.get("https://via.placeholder.com/300x450/667eea/ffffff?text=No+Poster", timeout=10)
        if resp.status_code == 200:
            return resp.content
    except Exception:
        pass
    return b""
# Page configuration
st.set_page_config(
    page_title="🎬 Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .movie-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    /* removed pink cards; using plain captions under posters */
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stSelectbox > div > div {
        background-color: #f0f2f6;
        color: white !important;
    }
    .stSelectbox label {
        color: #ffffff !important;
        font-weight: bold;
    }
    .sidebar .stSelectbox label {
        color: #ffffff !important;
        font-weight: bold;
    }
    .stSelectbox div[data-baseweb="select"] > div {
        color: #000000 !important;
    }
    .stSelectbox div[data-baseweb="select"] > div > div {
        color: #000000 !important;
    }
    .stSelectbox div[data-baseweb="select"] > div > div > div {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the movies data"""
    movies_list = pickle.load(open('movies.pkl','rb'))
    return movies_list

@st.cache_data
def create_similarity_matrix(movies_df):
    """Create and cache the similarity matrix"""
    # movies_list = load_data()
    
    # Create CountVectorizer to convert text to vectors
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies_df['tags']).toarray()
    
    # Calculate cosine similarity
    similarity = cosine_similarity(vectors)
    return similarity

def get_movie_poster(movie_title):
    """Get movie poster from TMDB API (placeholder for now)"""
    # This would require TMDB API key in a real implementation
    return "https://via.placeholder.com/300x450/667eea/ffffff?text=🎬"

def recommend_movies(movie_title, similarity_matrix, movies_list, num_recommendations=5):
    """Get movie recommendations based on similarity"""
    try:
        # Find the index of the selected movie
        movie_index = movies_list[movies_list['title'] == movie_title].index[0]
        
        # Get similarity scores for this movie
        distances = similarity_matrix[movie_index]
        
        # Get top similar movies (excluding the movie itself)
        movies_list_sorted = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:num_recommendations+1]
        
        recommended_movies = []
        for i in movies_list_sorted:
            recommended_movies.append(movies_list.iloc[i[0]]['title'])
        
        return recommended_movies
    except IndexError:
        return []

def get_movie_details(movie_title, movies_list):
    """Get movie details from the dataset"""
    try:
        movie_data = movies_list[movies_list['title'] == movie_title].iloc[0]
        return {
            'title': movie_data['title'],
            'movie_id': movie_data['movie_id'],
            'tags': movie_data['tags']
        }
    except IndexError:
        return None

def main():
    # Load data
    movies_list = load_data()
    similarity_matrix = create_similarity_matrix(movies_list)
    
    # Main header
    st.markdown('<h1 class="main-header">🎬 Movie Recommender System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 Choose Your Movie")
        st.markdown("Select a movie you like to get personalized recommendations!")
        
        # Movie selection
        selected_movie = st.selectbox(
            'Select a movie:',
            movies_list['title'].values,
            help="Choose a movie to get similar recommendations"
        )
        
        # Number of recommendations
        num_recommendations = st.slider(
            "Number of recommendations:",
            min_value=3,
            max_value=10,
            value=5,
            help="How many movies would you like to see?"
        )
        
        # Recommendation type
        recommendation_type = st.selectbox(
            "Recommendation type:",
            ["Similar Movies", "Popular Movies", "Random Movies"],
            help="Choose the type of recommendations"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Dataset Info")
        st.info(f"Total Movies: {len(movies_list)}")
        st.info(f"Selected: {selected_movie}")
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎬 Selected Movie")
        movie_details = get_movie_details(selected_movie, movies_list)
        
        if movie_details:
            try:
                poster_url = fetch_poster(movie_details['movie_id'], title=movie_details['title'], size='w500')
                img_bytes = fetch_image_bytes(poster_url)
                if img_bytes:
                    st.image(img_bytes, width=220)
            except Exception:
                pass
            st.markdown(f"""
            <div class="movie-card">
                <h3>{movie_details['title']}</h3>
                <p><strong>Movie ID:</strong> {movie_details['movie_id']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎯 Recommendations")
        
        if recommendation_type == "Similar Movies":
            recommended_movies = recommend_movies(selected_movie, similarity_matrix, movies_list, num_recommendations)
            
            if recommended_movies:
                st.markdown(f"**Movies similar to '{selected_movie}':**")
                rec_cols = st.columns(5)
                for i, movie in enumerate(recommended_movies, 1):
                    movie_details = get_movie_details(movie, movies_list)
                    if movie_details:
                        with rec_cols[(i-1) % 5]:
                            try:
                                poster_url = fetch_poster(movie_details['movie_id'], title=movie_details['title'], size='w342')
                                img_bytes = fetch_image_bytes(poster_url)
                                if img_bytes:
                                    st.image(img_bytes, width=160)
                            except Exception:
                                pass
                            st.caption(movie_details['title'])
            else:
                st.error("No recommendations found for this movie.")
        
        elif recommendation_type == "Popular Movies":
            # Show most popular movies (based on movie_id as a proxy)
            popular_movies = movies_list.nlargest(num_recommendations, 'movie_id')['title'].tolist()
            st.markdown("**Popular Movies:**")
            for i, movie in enumerate(popular_movies, 1):
                movie_details = get_movie_details(movie, movies_list)
                if movie_details:
                    try:
                        poster_url = fetch_poster(movie_details['movie_id'], title=movie_details['title'], size='w342')
                        img_bytes = fetch_image_bytes(poster_url)
                        if img_bytes:
                            st.image(img_bytes, width=160)
                    except Exception:
                        pass
                    st.caption(movie_details['title'])
        
        else:  # Random Movies
            random_movies = movies_list.sample(n=num_recommendations)['title'].tolist()
            st.markdown("**Random Movie Selection:**")
            for i, movie in enumerate(random_movies, 1):
                movie_details = get_movie_details(movie, movies_list)
                if movie_details:
                    try:
                        poster_url = fetch_poster(movie_details['movie_id'], title=movie_details['title'], size='w342')
                        img_bytes = fetch_image_bytes(poster_url)
                        if img_bytes:
                            st.image(img_bytes, width=160)
                    except Exception:
                        pass
                    st.caption(movie_details['title'])
    
    # Search functionality
    st.markdown("---")
    st.markdown("### 🔍 Search Movies")
    
    search_term = st.text_input("Search for a movie:", placeholder="Type movie name...")
    
    if search_term:
        # Filter movies based on search term
        filtered_movies = movies_list[movies_list['title'].str.contains(search_term, case=False, na=False)]
        
        if not filtered_movies.empty:
            st.markdown(f"**Found {len(filtered_movies)} movies matching '{search_term}':**")
            
            # Display search results in columns
            cols = st.columns(3)
            for idx, (_, movie) in enumerate(filtered_movies.iterrows()):
                with cols[idx % 3]:
                    movie_details = get_movie_details(movie['title'], movies_list)
                    if movie_details:
                        st.markdown(f"""
                        <div class="movie-card">
                            <h4>{movie_details['title']}</h4>
                            <p><strong>ID:</strong> {movie_details['movie_id']}</p>
                            <p>{movie_details['tags'][:100]}...</p>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.warning(f"No movies found matching '{search_term}'")
    
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🎬 Movie Recommender System | Built with Streamlit & Machine Learning</p>
        <p>Powered by Content-Based Filtering using Cosine Similarity</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

# 
