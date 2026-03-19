# Import libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------- Load Data -----------------
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Preprocess genres
movies['genres'] = movies['genres'].str.replace('|', ' ')

# TF-IDF vectorization for content-based similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create index mapping
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# ----------------- Recommendation Function -----------------
def recommend_movies(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5 similar movies
    movie_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]
    result = pd.DataFrame({
        "Recommended Movie": movies['title'].iloc[movie_indices],
        "Similarity Score": scores
    })
    return result

# ----------------- Streamlit UI -----------------
st.title("🎬 Interactive Movie Recommendation System")
st.write("Select a movie to get recommendations and see interactive visualizations.")

# User selects a movie
selected_movie = st.selectbox("Select a Movie", movies['title'].values)

# Button to show recommendations
if st.button("Show Recommendations"):
    recommendations = recommend_movies(selected_movie)
    st.subheader("Top 5 Recommended Movies")
    st.table(recommendations)

# ----------------- Interactive Plots -----------------
# Ratings for selected movie
st.subheader(f"📊 Rating Distribution for '{selected_movie}'")
movie_ratings = ratings[ratings['movieId'] == movies[movies['title'] == selected_movie]['movieId'].values[0]]

if not movie_ratings.empty:
    fig, ax = plt.subplots(figsize=(8,4))
    sns.histplot(movie_ratings['rating'], bins=5, kde=True, ax=ax)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    st.pyplot(fig)
else:
    st.write("No ratings available for this movie.")

# Genres of selected movie
st.subheader(f"🎭 Genres of '{selected_movie}'")
genres = movies[movies['title'] == selected_movie]['genres'].values[0].split()
genre_counts = pd.Series(genres).value_counts()
fig2, ax2 = plt.subplots(figsize=(6,3))
genre_counts.plot(kind='bar', color='skyblue', ax=ax2)
ax2.set_ylabel("Count")
st.pyplot(fig2)