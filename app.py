import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load dataset
df = pd.read_csv("tmdb_5000_movies.csv")
df = df[['title', 'overview']].dropna()

# Load model and create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
movie_embeddings = model.encode(df['overview'].tolist(), show_progress_bar=True)

# Recommendation function
def recommend_movies(user_input, top_n=5):
    user_embedding = model.encode([user_input])
    similarities = cosine_similarity(user_embedding, movie_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_n]
    return df['title'].iloc[top_indices].tolist()

# Streamlit UI
st.title("🎬 GenAI Movie Recommender")

user_input = st.text_input("Describe the kind of movie you want to watch:")

if user_input:
    recommendations = recommend_movies(user_input)
    st.subheader("Top Recommendations:")
    for movie in recommendations:
        st.write(f"👉 {movie}")
