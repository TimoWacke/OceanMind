import streamlit as st
import os

# Import dependen
import constants

os.environ["OPENAI_API_KEY"] = constants.API_KEY

#Title 
st.title("GPT-3 Playground")

query = st.text_input("Query: ", "What is the meaning of life?")

import openai
openai.api_key = os.environ["OPENAI_API_KEY"]

response = openai.Completion.create(