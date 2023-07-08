import streamlit as st
import pickle
import pandas as pd
import requests

def fetch_poster(movie_id):
    respose = requests.get(url.format(movie_id))
    data = respose.json()
    return "https://image.tmdb.orgg/t/p/w500" + data['poster_path']

def recommend(movie):
    movie_index = movies[movies['title']==movie].index[0]
    distance = similarity[movie_index]
    movie_list = sorted(list(enumerate(distance)),reverse=True,key=lambda x:x[1])[1:6]

    recommended_movies = []
    recommended_movies_poster = []
    for i in movie_list:
        movie_id=movies.iloc[i[0]].movie.id

        recommended_movies.append(movies.iloc[i[0]].title)
        #fetch poster from api
        recommended_movies_poster.append(fetch_poster(movie_id))
    return recommended_movies

movies_dict = pickle.load(open('movies_dict.pkl','rb'))
movies = pd.DataFrame(movies_dict)

similarity = pickle.load(open('similarity.pkl','rb'))
st.title('Movie Recommender System')

selected_movie_name = st.selectbox(
'How would you like to be contacted',
movies['title'].values)

if st.button('Recommend'):
    recommended_movie_names,recommended_movie_posters = recommend(selected_movie_name)
    col1, col2, col3, col4, col5 = st.beta_columns(5)
    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])
    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])

    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])
