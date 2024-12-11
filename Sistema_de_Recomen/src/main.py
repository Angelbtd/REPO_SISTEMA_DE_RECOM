import pandas as pd
from src.data_preprocessing import load_and_merge_data
from src.collaborative_filtering import collaborative_filtering
from src.content_based_recommendation import content_based_recommendation

if __name__ == "__main__":
    movie_data = load_and_merge_data('../data/movies.csv', '../data/ratings.csv')
    print("Filtrado Colaborativo:")
    collaborative_filtering(movie_data)
    
    print("
Recomendación basada en contenido:")
    recommendations = content_based_recommendation(movie_data, 'Toy Story (1995)')
    print(recommendations)