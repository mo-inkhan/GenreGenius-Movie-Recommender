import pandas as pd

def load_data():
    """
    Load movie data from a CSV file into a pandas DataFrame.

    :return movies: pd.DataFrame - DataFrame containing movie information (movieId, title, genres)
    """

    return pd.read_csv("movies.csv")