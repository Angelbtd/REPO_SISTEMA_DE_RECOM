from flask import Flask, jsonify, request
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
    app.run(debug=True)