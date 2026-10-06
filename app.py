import streamlit as st
from pypdf import PdfReader

st.set_page_config(page_title="AI Assistant", page_icon="🤖")
st.title("AI Assistant - Test 2")
st.caption("Testing PDF file text extraction")

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
        st.subheader("Extracted text:")
        with st.expander("View Content"):
            st.text_area("",value=pdf_text, height = 400)
    else:
        st.warning("Could not read document.")
else:
    st.info("Insert PDF file to test")


