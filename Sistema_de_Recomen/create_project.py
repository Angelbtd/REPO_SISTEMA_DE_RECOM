import os

# Definir la estructura de carpetas y archivos
project_name = "Sistema_de_Recomendacion_Personalizada"
directories = [
    "data",
    "src",
]

# Archivos dentro de 'data'
data_files = [
    ("data/movies.csv", "# Este archivo contiene el dataset de películas de MovieLens."),
    ("data/ratings.csv", "# Este archivo contiene el dataset de calificaciones de los usuarios.")
]

# Archivos dentro de 'src'
src_files = {
    "src/__init__.py": "",
    "src/data_preprocessing.py": '''import pandas as pd

def load_and_merge_data(movies_path, ratings_path):
    \"\"\"Cargar y fusionar los datos de las películas y las calificaciones.\"\"\"
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    movie_data = pd.merge(ratings, movies, on='movieId')
    return movie_data

if __name__ == "__main__":
    movie_data = load_and_merge_data('../data/movies.csv', '../data/ratings.csv')
    print(movie_data.head())''',
    
    "src/collaborative_filtering.py": '''from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
import pandas as pd

def collaborative_filtering(movie_data):
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(movie_data[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.25)

    model = SVD()
    model.fit(trainset)

    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    print(f"RMSE: {rmse}")
    return model

if __name__ == "__main__":
    movie_data = pd.read_csv('../data/movies.csv')
    collaborative_filtering(movie_data)''',
    
    "src/content_based_recommendation.py": '''import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def content_based_recommendation(movie_data, movie_title):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movie_data['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    idx = movie_data.index[movie_data['title'] == movie_title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[1:6]]  # Top 5 recomendados

    return movie_data['title'].iloc[movie_indices]

if __name__ == "__main__":
    movie_data = pd.read_csv('../data/movies.csv')
    print(content_based_recommendation(movie_data, 'Toy Story (1995)'))''',

    "src/main.py": '''import pandas as pd
from src.data_preprocessing import load_and_merge_data
from src.collaborative_filtering import collaborative_filtering
from src.content_based_recommendation import content_based_recommendation

if __name__ == "__main__":
    movie_data = load_and_merge_data('../data/movies.csv', '../data/ratings.csv')
    print("Filtrado Colaborativo:")
    collaborative_filtering(movie_data)
    
    print("\nRecomendación basada en contenido:")
    recommendations = content_based_recommendation(movie_data, 'Toy Story (1995)')
    print(recommendations)''',
    
    "src/recommendation_api.py": '''from flask import Flask, jsonify, request
import pandas as pd
from src.content_based_recommendation import content_based_recommendation

app = Flask(__name__)

@app.route('/recommend', methods=['GET'])
def recommend():
    movie_title = request.args.get('title')
    movie_data = pd.read_csv('../data/movies.csv')
    recommendations = content_based_recommendation(movie_data, movie_title)
    return jsonify(recommendations.tolist())

if __name__ == "__main__":
    app.run(debug=True)''',
    
    "src/utils.py": '''import matplotlib.pyplot as plt
import seaborn as sns

def plot_ratings_distribution(movie_data):
    plt.figure(figsize=(10, 6))
    sns.histplot(movie_data['rating'], kde=True)
    plt.title("Distribución de Calificaciones")
    plt.xlabel("Calificación")
    plt.ylabel("Frecuencia")
    plt.show()''',

    "config.py": '''import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MOVIES_FILE = os.path.join(DATA_DIR, 'movies.csv')
RATINGS_FILE = os.path.join(DATA_DIR, 'ratings.csv')'''
}

# Crear directorios principales
for directory in directories:
    os.makedirs(directory, exist_ok=True)

# Crear archivos dentro de 'data'
for file, content in data_files:
    with open(file, 'w') as f:
        f.write(content)

# Crear archivos dentro de 'src'
for file, content in src_files.items():
    with open(file, 'w') as f:
        f.write(content)

print(f"Proyecto '{project_name}' ha sido creado exitosamente.")
