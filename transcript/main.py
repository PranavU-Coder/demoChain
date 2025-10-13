import streamlit as st
import youtube_enjoyer as yt
import textwrap

st.title("YouTube Assistant")

with st.sidebar:
    with st.form(key='my_form'):
        youtube_url = st.sidebar.text_area(
            label = 'what is the youtube url?',
            max_chars=50
        )
        query = st.sidebar.text_area(
            label = "ask me about the video",
            max_chars=50,
            key='query'
        )

        submit_button = st.form_submit_button(label='submit')

if query and youtube_url:
    db = yt.create_vector_deb_youtube_url(youtube_url)
    response = yt.get_response_query(db, query=query)
    st.subheader("Response: ")
    st.text(textwrap.fill(response,width=80))