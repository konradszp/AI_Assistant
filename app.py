import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

load_dotenv()

st.set_page_config(page_title="AI Assistant", page_icon="🤖")
st.title("AI Assistant - Test 3")
st.caption("Testing Google API Connection & Embeddings Generation")

with st.sidebar:
    st.header("1. Document")
    uploaded_file = st.file_uploader("Add PDF file: ", type=["pdf"])

def extract_text(pdf_file) -> str:
    # Extract all text content from the pdf file
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
        return text




if uploaded_file != None:
    st.success(f"Uploading done: {uploaded_file.name}")

    with st.spinner("Reading PDF content..."):
        pdf_text = extract_text(uploaded_file)

    if pdf_text.strip():
        st.subheader("Testing Google Gemini API Connection")

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("api key not found in env file")
        else:
            st.info("API Key was found.")

            if st.button("Generate test embedding"):
                with st.spinner("Sending request to Google API..."):
                    try:
                        text = pdf_text[:300]
                        # Get valid embedding model
                        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
                        vector = embeddings.embed_query(text)

                        st.success("Success! Got vector from Google API")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(label="Vector's length", value=len(vector))
                        with col2:
                            st.metric(label="Model", value="gemini-embedding-001")
                        with st.expander("View of the first 10 values of Vector"):
                            st.write(vector[:10])

                    except Exception as e:
                        st.error(f"Error connecting to Google API: {e}")

    else:
        st.warning("Could not read document.")

else:
    st.info("Insert PDF file to test")
    


