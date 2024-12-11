from surprise import SVD, Dataset, Reader
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
    collaborative_filtering(movie_data)