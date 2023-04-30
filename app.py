import streamlit as st
from loadData import load_data
from model import create_similarity_matrix, get_recommendations

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon=None
)

st.title("Movie Recommendation System")

movies = load_data()
cosine_sim = create_similarity_matrix(movies)

user_input = st.text_input("Enter a movie title", "")
recommendations = get_recommendations(user_input, movies, cosine_sim)

if recommendations.empty:
    st.write("Example: Arrival (2016)")
    st.write("No recommendations found.")
else:
    st.write(recommendations)
