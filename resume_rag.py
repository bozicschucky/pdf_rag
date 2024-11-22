import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions
import requests
import json
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')


@st.cache_resource
def init_chroma():
    client = chromadb.Client()
    collection_name = "resume_data"
    try:
        collection = client.create_collection(collection_name)
    except Exception:
        collection = client.get_collection(collection_name)
    return client, collection


client, collection = init_chroma()


def read_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def chunk_text(text, chunk_size=100):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size])
              for i in range(0, len(words), chunk_size)]
    return chunks


resume_text = read_pdf(os.getenv('DOCUMENT_PATH'))
chunks = chunk_text(resume_text)


embedding_model = embedding_functions.DefaultEmbeddingFunction()


for i, chunk in enumerate(chunks):
    embedding = embedding_model([chunk])
    if embedding is None:
        print(f"Embedding for chunk {i} is None")
    else:
        # Ensure embeddings are in the correct format
        embedding_list = embedding[0].tolist() if isinstance(
            embedding, list) and len(embedding) > 0 else embedding
        print(f"Embedding for chunk {i}:", embedding_list)
        collection.upsert(
            ids=[str(i)],
            documents=[chunk],
            embeddings=[embedding_list]
        )


def query_chroma_db(query):
    query_embedding = embedding_model([query])
    if query_embedding is None:
        print("Query embedding is None")
        return
    query_embedding_list = query_embedding[0].tolist() if isinstance(
        query_embedding, list) and len(query_embedding) > 0 else query_embedding
    results = collection.query(
        query_embeddings=[query_embedding_list], n_results=5)
    documents = results['documents']
    print("Query Results:", documents)

    if not documents:
        print("No documents found")
        return None

    return documents


def generate_response(prompt):
    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={
        api_key}'
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()


def answer_query(query):
    retrieved_text = query_chroma_db(query)

    prompt = f'''
    You are a smart answering assistant that provides concise and straight to the point answers based on the information provided,

    The following resume information is provided: {retrieved_text}, answer the question: {query}

    Provide concise and brief answers.
    Avoid adding unnecessary information like Based on your resume, or As per your resume, etc.
    return the answer in markdown format so that it is displayed properly in the Streamlit app.
    '''
    response = generate_response(prompt)
    markdown_response = response['candidates'][0]['content']['parts'][0]['text']
    return markdown_response


# Streamlit app
st.title("Resume QA Application")

query = st.text_input("Enter your query:")

if st.button("Submit"):
    if query:
        response = answer_query(query)
        st.markdown(response)
    else:
        st.error("Please enter a query.")
