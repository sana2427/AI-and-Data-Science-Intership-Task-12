# Task 2: Content-Based Movie Recommendation System with Streamlit GUI

## Overview
This repository contains the solution for Task 2: Building a **content-based movie recommendation system** using the **MovieLens dataset**.  
The system recommends movies based on genre similarity and provides an **interactive Streamlit GUI** for users to explore recommendations and visual insights.

---

## Objective
The objective of this task is to build a recommendation system that:

- Suggests the **top 5 movies similar** to a selected movie using **content-based filtering**.
- Leverages **TF-IDF vectorization** to represent movie genres numerically.
- Computes **cosine similarity** between movies to determine similarity.
- Provides an **interactive Streamlit dashboard** for user-friendly recommendations and visualizations.

---

## Dataset
**Dataset Name:** MovieLens Dataset  
**Type:** Structured CSV  

### Dataset Description
1. **movies.csv**  
   - `movieId`: Unique identifier for each movie  
   - `title`: Movie title  
   - `genres`: Genres of the movie (separated by `|`)  

2. **ratings.csv**  
   - `userId`: Unique identifier for each user  
   - `movieId`: Movie identifier  
   - `rating`: User rating for the movie  
   - `timestamp`: When the rating was given  

Movies can belong to multiple genres, which are used as features for content-based similarity.  
User ratings provide additional insights into popularity and distribution.

---

## Tools & Technologies Used
- Python  
- pandas & NumPy  
- scikit-learn (TF-IDF Vectorizer, Cosine Similarity)  
- Matplotlib & Seaborn  
- Streamlit  
- Jupyter Notebook / Google Colab  

---

## Approach

### 1️⃣ Data Loading
- Loaded **movies.csv** and **ratings.csv** using `pandas.read_csv()`.
- Converted CSV data into pandas DataFrames for easy manipulation.

### 2️⃣ Data Cleaning & Preprocessing
- Checked for **missing values** using `.isnull().sum()`.
- Replaced the `|` separator in `genres` with spaces to treat genres as **text features**.
- Prepared the data for TF-IDF vectorization.

### 3️⃣ Exploratory Data Analysis (EDA)
- **Rating Distribution:** Visualized using histograms.  
- **Top Genres:** Identified most frequent genres using bar charts.  
- **Additional EDA Graphs (Optional):**
  - Average rating per genre  
  - Movie count per genre  

### 4️⃣ Model Building
- **Feature Representation:** Used TF-IDF to convert movie genres into numerical vectors.  
- **Similarity Calculation:** Applied **cosine similarity** between genre vectors.  
- **Recommendation Function:**  
  - Input a movie title → retrieves top 5 most similar movies based on genre similarity.  

### 5️⃣ Testing the Recommendation System
- Verified functionality using `"Toy Story (1995)"` as input.
- Ensured top 5 recommended movies match the genre profile.

### 6️⃣ Streamlit GUI Implementation
- Users select a movie from a dropdown menu.
- Click **“Show Recommendations”** to view top 5 similar movies with similarity scores.
- Interactive visualizations:
  - Rating distribution histogram  
  - Genre bar chart  

---

## Results & Insights
- **Genre-based similarity works effectively:** Movies with similar genres are grouped together.  
- **TF-IDF improves feature representation:** Highlights important genres and downplays common ones.  
- **Cosine similarity is efficient:** Simple and effective for content-based filtering.  
- **Interactive visualization adds value:** Helps users understand rating patterns and genre distribution.  
- **User-friendly GUI:** Streamlit interface allows non-technical users to explore recommendations easily.

---

## Conclusion
A **content-based movie recommendation system** was successfully built using the MovieLens dataset.  
- TF-IDF vectorization transformed movie genres into numerical features.  
- Cosine similarity computed similarities between movies to generate recommendations.  
- The Streamlit GUI provides a clean and interactive way to explore movie recommendations and visual insights.  

This project demonstrates how **simple text features** and **machine learning techniques** can create an effective, user-friendly recommendation system.

---

## Skills Gained
- Data preprocessing and cleaning with pandas  
- TF-IDF feature extraction for text data  
- Cosine similarity for content-based recommendation  
- Exploratory Data Analysis (EDA) with visualizations  
- Interactive dashboard creation using Streamlit