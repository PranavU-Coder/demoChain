import langchain_helper as lch
import streamlit as st

st.title('Pet Name Generator')

animal_type = st.sidebar.selectbox('what is your pet?',{"cat","dog","cow","hamster"})

if animal_type == "cat":
    pet_color = st.sidebar.text_area(label="what color is your cat", max_chars=15)

if animal_type == "dog":
    pet_color = st.sidebar.text_area(label="what color is your dog", max_chars=15)

if animal_type == "cow":
    pet_color = st.sidebar.text_area(label="what color is your cow", max_chars=15)

if animal_type == "hamster":
    pet_color = st.sidebar.text_area(label="what color is your hamster", max_chars=15)    

if pet_color:
    response = lch.generate_pet_name(animal_type=animal_type,pet_color=pet_color)
    st.text(response)