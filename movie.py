# movie.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------- Load Data -----------------
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# ----------------- Preprocess Data -----------------
movies['genres'] = movies['genres'].str.replace('|', ' ')

# ----------------- TF-IDF & Similarity -----------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Map movie titles to indices
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# ----------------- Recommendation Function -----------------
def recommend_movies(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5 similar movies
    movie_indices = [i[0] for i in sim_scores]
    scores = [float(i[1]) for i in sim_scores]  # Convert to Python float
    titles = [str(movies['title'].iloc[i]) for i in movie_indices]  # Convert to string
    result = pd.DataFrame({
        "Recommended Movie": titles,
        "Similarity Score": scores
    })
    return result

# ----------------- Streamlit UI -----------------
st.title("🎬 Interactive Movie Recommendation System")
st.write("Select a movie to get recommendations and explore interactive visualizations.")

# Movie selection
selected_movie = st.selectbox("Select a Movie", movies['title'].values)

if st.button("Show Recommendations"):
    recommendations = recommend_movies(selected_movie)
    st.subheader("Top 5 Recommended Movies")
    st.table(recommendations)

# ----------------- EDA & Visualizations -----------------
st.subheader(f"📊 Rating Distribution for '{selected_movie}'")
movie_id = movies[movies['title'] == selected_movie]['movieId'].values[0]
movie_ratings = ratings[ratings['movieId'] == movie_id]

if not movie_ratings.empty:
    fig, ax = plt.subplots(figsize=(8,4))
    sns.histplot(movie_ratings['rating'], bins=5, kde=True, ax=ax, color='skyblue')
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
genre_counts.plot(kind='bar', color='orange', ax=ax2)
ax2.set_ylabel("Count")
st.pyplot(fig2)

# Top 10 genres overall
st.subheader("🔥 Top 10 Movie Genres Overall")
genre_series = movies['genres'].str.split(expand=True).stack()
top_genres = genre_series.value_counts().head(10)
fig3, ax3 = plt.subplots(figsize=(8,4))
top_genres.plot(kind='bar', color='green', ax=ax3)
ax3.set_ylabel("Count")
st.pyplot(fig3)

# Top 10 movies by average rating (with at least 50 ratings)
st.subheader("🏆 Top 10 Movies by Average Rating")
rating_counts = ratings.groupby('movieId').count()['rating']
rating_avgs = ratings.groupby('movieId').mean()['rating']
popular_movies = rating_avgs[rating_counts >= 50].sort_values(ascending=False).head(10)
popular_titles = movies.set_index('movieId').loc[popular_movies.index]['title']
fig4, ax4 = plt.subplots(figsize=(8,4))
sns.barplot(x=popular_movies.values, y=popular_titles.values, ax=ax4, palette='coolwarm')
ax4.set_xlabel("Average Rating")
ax4.set_ylabel("Movie")
st.pyplot(fig4)
