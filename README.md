# 🎬 Movie Recommender System

A comprehensive movie recommendation system built with Streamlit and machine learning techniques.

## ✨ Features

- **Content-Based Filtering**: Uses cosine similarity to recommend movies based on content tags
- **Multiple Recommendation Types**: 
  - Similar Movies (based on content similarity)
  - Popular Movies (based on movie ID as popularity proxy)
  - Random Movie Selection
- **Interactive UI**: Beautiful, responsive interface with gradient styling
- **Search Functionality**: Search and filter movies by title
- **Movie Details**: Display comprehensive movie information
- **Customizable Recommendations**: Choose the number of recommendations (3-10)

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

3. **Open in Browser**: The app will automatically open at `http://localhost:8501`

## 📊 Dataset

The system uses a dataset containing:
- **4,806 movies** with detailed information
- **Movie ID, Title, and Tags** for each movie
- **Content-based features** extracted from movie descriptions and tags

## 🧠 How It Works

1. **Data Processing**: Movie tags are converted to numerical vectors using CountVectorizer
2. **Similarity Calculation**: Cosine similarity is computed between all movie vectors
3. **Recommendation Generation**: For a selected movie, the system finds the most similar movies
4. **User Interface**: Interactive Streamlit interface allows users to explore recommendations

## 🎯 Usage

1. **Select a Movie**: Choose any movie from the dropdown in the sidebar
2. **Choose Recommendation Type**: 
   - Similar Movies: Content-based recommendations
   - Popular Movies: Most popular movies in the dataset
   - Random Movies: Random selection for discovery
3. **Adjust Settings**: Use the slider to choose how many recommendations to display
4. **Search Movies**: Use the search bar to find specific movies
5. **Explore Results**: Click through different movies to see various recommendations

## 🛠️ Technical Details

- **Framework**: Streamlit for web interface
- **ML Library**: Scikit-learn for vectorization and similarity calculation
- **Data Processing**: Pandas for data manipulation
- **Caching**: Streamlit caching for optimal performance
- **Styling**: Custom CSS with gradient backgrounds and responsive design

## 📁 Project Structure

```
MovieRecommend/
├── app.py              # Main Streamlit application
├── movies.pkl          # Movie dataset (pickle format)
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🔧 Customization

- **Add More Features**: Extend with user ratings, collaborative filtering, or external APIs
- **Improve Styling**: Modify the CSS in the app.py file
- **Add Data Sources**: Integrate with TMDB API for movie posters and additional details
- **Enhance Algorithms**: Implement more sophisticated recommendation algorithms

## 🎨 Features Showcase

- **Beautiful UI**: Gradient backgrounds, card-based layout, responsive design
- **Real-time Search**: Instant movie search and filtering
- **Multiple Views**: Different recommendation types for various use cases
- **Performance Optimized**: Cached computations for fast loading
- **User-Friendly**: Intuitive interface with helpful tooltips and descriptions

## 🚀 Future Enhancements

- [ ] User authentication and personal profiles
- [ ] Movie rating system
- [ ] Collaborative filtering recommendations
- [ ] Integration with movie databases (TMDB, OMDB)
- [ ] Movie poster display
- [ ] Advanced filtering options (genre, year, rating)
- [ ] Export recommendations functionality
- [ ] Mobile-responsive improvements

---

**Built with ❤️ using Streamlit and Machine Learning**

