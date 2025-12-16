import os

import streamlit as st

from RAG_utility import process_documents_to_chroma_db, answer_question

#set working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

st.title("Document Answering RAG")

uploaded_file = st.file_uploader("Upload a pdf file", type=["pdf"])

if uploaded_file is not None:
    # save the file in a directory
    save_path = os.path.join(working_dir, uploaded_file.name)
    # save the file
    with open(save_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    process_documents = process_documents_to_chroma_db(uploaded_file.name)
    st.info("document process successfully")

# get user input

user_question = st.text_input("Ask your question")
if st.button("Answer"):

    answer = answer_question(user_question)
    st.markdown("GROQ LLM Answer")
    st.markdown(answer)