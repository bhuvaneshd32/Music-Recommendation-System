from model import recommend_songs

import streamlit as st

st.set_page_config(page_title="Music Recommendation System", layout="centered")

if "rows" not in st.session_state:
    st.session_state.rows = [{"name": "", "year": ""}]  # Initialize with one row

def add_row():
    st.session_state.rows.append({"name": "", "year": ""})

def display_rows():
    for i, row in enumerate(st.session_state.rows):
        col1, col2 = st.columns([3, 1])  # Column ratios
        with col1:
            st.session_state.rows[i]["name"] = st.text_input(
                f"Song Title {i + 1}",
                value=row["name"],
                key=f"name_{i}"
            )
        with col2:
            st.session_state.rows[i]["year"] = st.text_input(
                f"Year {i + 1}",
                value=row["year"],
                key=f"year_{i}"
            )

st.title("Music Recommendation System")
st.write("Enter song titles and release years")

display_rows()

if st.button("Add Row"):
    add_row()

if st.button("Submit"):
    st.write("Recommended Songs:")
    st.write(st.session_state.rows)
    st.write(recommend_songs(st.session_state.rows))
