import pandas as pd

def load_and_merge_data(movies_path, ratings_path):
    """Cargar y fusionar los datos de las películas y las calificaciones."""
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    movie_data = pd.merge(ratings, movies, on='movieId')
    return movie_data

if __name__ == "__main__":
    movie_data = load_and_merge_data('../data/movies.csv', '../data/ratings.csv')
    print(movie_data.head())