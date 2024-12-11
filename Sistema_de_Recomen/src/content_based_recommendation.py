import pandas as pd
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
    print(content_based_recommendation(movie_data, 'Toy Story (1995)'))