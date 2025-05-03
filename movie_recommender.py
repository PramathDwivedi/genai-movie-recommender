from sentence_transformers import SentenceTransformer
import pandas as pd

# Step 1: Load your real dataset
df = pd.read_csv("tmdb_5000_movies.csv")

# Step 2: Keep only the columns we need
df = df[['title', 'overview']]

# Step 3: Drop missing data (some movies have no overview)
df.dropna(inplace=True)

# Step 4: Load the embedding model (this will download the model once)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 5: Create embeddings from the overviews
movie_overviews = df['overview'].tolist()
movie_embeddings = model.encode(movie_overviews, show_progress_bar=True)

# Step 6: Preview
print("1st Movie:", df['title'][0])
print("1st Overview:", df['overview'][0])
print("1st Embedding:", movie_embeddings[0])
from sklearn.metrics.pairwise import cosine_similarity

def recommend_movies(user_input, top_n=5):
    # Step 1: Embed the user input (same way as we did for overviews)
    user_embedding = model.encode([user_input])

    # Step 2: Calculate cosine similarity with all movie embeddings
    similarities = cosine_similarity(user_embedding, movie_embeddings)[0]  # [0] because output is [[...]]
    
    # Step 3: Sort and get top N similar movies
    top_indices = similarities.argsort()[::-1][:top_n]

    # Step 4: Get movie titles
    recommended_titles = df['title'].iloc[top_indices].tolist()
    
    return recommended_titles
if __name__ == "__main__":
    query = input("Describe the kind of movie you want to watch: ")
    recommendations = recommend_movies(query)

    print("\nTop Movie Recommendations:")
    for i, title in enumerate(recommendations, 1):
        print(f"{i}. {title}")
