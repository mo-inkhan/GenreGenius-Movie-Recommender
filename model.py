import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def create_similarity_matrix(movies):
    """
    Create a similarity matrix using the genres of the movies.

    :param movies: pd.DataFrame - DataFrame containing movie information (movieId, title, genres)
    :return cosine_sim: np.ndarray - Cosine similarity matrix between movies based on their genres
    """

    count_vectorizer = CountVectorizer(stop_words="english")
    count_matrix = count_vectorizer.fit_transform(movies["genres"])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    return cosine_sim


def get_recommendations(title, movies, cosine_sim):
    """
    Get movie recommendations based on a given title.

    :param title: str - Movie title to base recommendations on
    :param movies: pd.DataFrame - DataFrame containing movie information (movieId, title, genres)
    :param cosine_sim: np.ndarray - Cosine similarity matrix between movies based on their genres
    :return recommendations: pd.DataFrame - DataFrame containing the top 10 most similar movie titles
    """

    title = title.lower()
    indices = pd.Series(movies.index, index=movies["title"].apply(
        lambda x: x.lower())).drop_duplicates()

    if title not in indices:
        return pd.DataFrame(columns=['title'])

    index = indices[title]
    similarity_scores = list(enumerate(cosine_sim[index]))
    similarity_scores = sorted(
        similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:11]

    movie_indices = [i[0] for i in similarity_scores]
    return movies["title"].iloc[movie_indices].reset_index(drop=True)
