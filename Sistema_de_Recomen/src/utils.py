import matplotlib.pyplot as plt
import seaborn as sns

def plot_ratings_distribution(movie_data):
    plt.figure(figsize=(10, 6))
    sns.histplot(movie_data['rating'], kde=True)
    plt.title("Distribución de Calificaciones")
    plt.xlabel("Calificación")
    plt.ylabel("Frecuencia")
    plt.show()