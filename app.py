import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS

load_dotenv()

st.set_page_config(page_title="AI Assistant", page_icon="🤖")
st.title("AI Assistant - Test 5")
st.caption("Smart Onboarding Implementation")

with st.sidebar:
    st.header("Document")
    pdf_file = st.file_uploader("Add PDF file: ", type=["pdf"])

    if st.button("Clean chat history"):
        st.session_state.messages = []
        st.rerun()

# Extract all text content from the pdf file
def extract_text(pdf_file) -> str:
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

@st.cache_resource(show_spinner=False)
def build_rag_chain(pdf_file):
    pdf_text = extract_text(pdf_file)
    if not pdf_text.strip():
        return None, 0


    #Chunking Data
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    chunks = text_splitter.split_text(pdf_text)

    # Embedding model & FAISS
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", max_retries=5)
    vector_store = FAISS.from_texts(texts=chunks, embedding=embedding_model)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # Prompt
    system_prompt = (
        "You are professional corporate knowledge assistant. "
        "Answer only from the document context below. "
        "If there is no answer, say that it does not contain this information. "
        "Do not make up facts or do not take knowledge from outside the document"
        "Document context:\n{context}"
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    # Model
    chat_model = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.2)

    def format_documents(documents):
        return "\n\n".join(document.page_content for document in documents)

    rag_chain = (
        {"context": retriever | format_documents, "question": RunnablePassthrough()}
        | prompt_template
        | chat_model
        | StrOutputParser()
    )

    return rag_chain, len(chunks)

chat_model = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.2)
def smart_onboard(text: str):
    sample_text = text[:10000].strip()
    prompt = ChatPromptTemplate.from_template(
        "Based on the following document excerpt, generate:\n"
        "1. A 3-bullet-point executive summary in English.\n"
        "2. Exactly 3 relevant, interesting questions a user might ask about this document.\n\n"
        "Format output EXACTLY like this:\n"
        "SUMMARY:\n"
        "- Bullet 1\n"
        "- Bullet 2\n"
        "- Bullet 3\n"
        "QUESTIONS:\n"
        "1. Question one?\n"
        "2. Question two?\n"
        "3. Question three?\n\n"
        "Document excerpt:\n{context}"
    )

    chain = prompt | chat_model
    response = chain.invoke({"context": sample_text})

    content = response.content
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    
    return content.strip()

if pdf_file is not None and "onboarding_done" not in st.session_state:
    with st.spinner("Generating summary and suggested questions..."):
        try:
            pdf_text = extract_text(pdf_file)
            if pdf_text.strip():
                onboarding_text = smart_onboard(pdf_text)
                st.session_state["onboarding_text"] = onboarding_text
            else:
                st.session_state["onboarding_text"] = "Couldn't read PDF text"
            st.session_state["onboarding_done"] = True
        except Exception as e:
            st.error(f"Error generating onboarding: {e}")
            st.session_state["onboarding_done"] = True

if "onboarding_text" in st.session_state:
    st.markdown("### 💡 About the document (Smart Onboarding)")
    output = st.session_state["onboarding_text"]
    if isinstance(output, list):
        output = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in output
        )
    
    output = str(output)

    if "QUESTIONS:" in output.upper():
        parts = output.upper().split("QUESTIONS:")
        idx = output.upper().find("QUESTIONS:")
        summary_part = output[:idx]
        questions_part = output[idx + len("QUESTIONS:"):]
        st.info(summary_part.replace("SUMMARY:", "").replace("Summary:", "").strip())
        
        st.markdown("**Suggested questions:**")
        questions = []
        for line in questions_part.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            if line[0].isdigit():
                cleaned = line.split(".", 1)[-1].split(")", 1)[-1].strip()
                if cleaned:
                    questions.append(cleaned)
            elif line.startswith("-"):
                questions.append(line.lstrip("- ").strip())
        
        if questions:
            cols = st.columns(min(len(questions), 3))
            for idx, q in enumerate(questions[:3]):
                if cols[idx].button(q, key=f"q_btn_{idx}"):
                    st.session_state["user_question"] = q
                    st.rerun()
        else:
            st.write(questions_part)
    else:
        st.info(output)

if "messages" not in st.session_state:
    st.session_state.messages = []


if pdf_file != None:

    with st.spinner("Document processing"):
        try:
            rag_chain , num_chunks = build_rag_chain(pdf_file)
            if rag_chain is None:
                st.error("PDF file is empty or does not contain readable text")
            else:
                st.sidebar.success(f"Indexed {num_chunks} chunks")
        except Exception as e:
            st.error(f"Error creating knowledge base: {e}")
            rag_chain=None

    if rag_chain != None:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        pending_question = st.session_state.pop("user_question", None)

        chat_question = st.chat_input("Ask question about the document")

        question = pending_question or chat_question

        if question:
            st.session_state.messages.append({"role": "user", "content": question})

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Generating response..."):
                    try:
                        response = rag_chain.invoke(question)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error generating response by Gemini AI: {e}")
else:
    st.info("Insert PDF file to test")
    


