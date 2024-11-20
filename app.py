import streamlit as st
import pandas as pd
from music_recommendation_system import recommend_songs, find_song


st.title("Music Recommendation System")
st.write("This app recommends songs based on a song you like! Enter a song name and release year to get started.")

# Input fields for song name and year
song_name = st.text_input("Enter Song Name")
song_year = st.number_input("Enter Song Year", min_value=1900, max_value=2023, step=1)

spotify_data = pd.read_csv("data.csv")  

if st.button("Get Recommendations"):
    if song_name and song_year:
        # Find the song in the dataset
        song_info = find_song(song_name, song_year)
        if song_info:
            # Get recommendations
            recommendations = recommend_songs([song_info], spotify_data, n_songs=10)
            st.write("### Recommended Songs:")
            for i, song in enumerate(recommendations, 1):
                st.write(f"{i}. **{song['name']}** by {song['artists']} ({song['year']})")
        else:
            st.error("Song not found. Please check the name and year and try again.")
    else:
        st.warning("Please enter both a song name and year.")
import streamlit as st
from music_recommendation_system import recommend_songs, find_song
import pandas as pd

# Title and description
st.title("Music Recommendation System")
st.write("This app recommends songs based on a song you like! Enter a song name and release year to get started.")

# Input fields for song name and year
song_name = st.text_input("Enter Song Name")
song_year = st.number_input("Enter Song Year", min_value=1900, max_value=2023, step=1)

# Load dataset (if required, update the path if it's not in the same directory)
# Uncomment the line below and replace with the correct path if necessary
# spotify_data = pd.read_csv("path_to_spotify_data.csv")

# Button to generate recommendations
if st.button("Get Recommendations"):
    if song_name and song_year:
        # Find the song in the dataset
        song_info = find_song(song_name, song_year)
        if song_info:
            # Get recommendations
            recommendations = recommend_songs([song_info], spotify_data, n_songs=10)
            st.write("### Recommended Songs:")
            for i, song in enumerate(recommendations, 1):
                st.write(f"{i}. **{song['name']}** by {song['artists']} ({song['year']})")
        else:
            st.error("Song not found. Please check the name and year and try again.")
    else:
        st.warning("Please enter both a song name and year.")
