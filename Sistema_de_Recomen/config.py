import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MOVIES_FILE = os.path.join(DATA_DIR, 'movies.csv')
RATINGS_FILE = os.path.join(DATA_DIR, 'ratings.csv')