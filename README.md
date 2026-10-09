## Project Overview

Enterprise Knowledge Copilot is a simple RAG (Retrieval-Augmented Generation) web application built with Python and Streamlit. It allows users to upload PDF documents and ask questions about their content.

The app uses LangChain to split PDFs into smaller text chunks, converts them into embeddings with Google Gemini, and saves them in a FAISS vector database. When a user asks a question, the app finds the most relevant document parts and passes them to the Gemini model to generate an answer. The prompt is designed so the model only uses the uploaded document, creating a fully functional RAG pipeline without hallucinations.

--------------------------------------------------

### Features

* **Fact-Based Answers:** The AI only answers using information from the PDF file and declines to answer if the information does not exist inside it.
* **Vector Search:** Uses FAISS and Google Gemini embedding model to search for matching text.
* **Streamlit UI:** Tidy and user friendly chat interface.
* **Docker Support:** Runs inside a Docker container using Docker Compose so it works on any machine.

--------------------------------------------------

## Setup & Running with Docker

### Prerequisites

* Docker Desktop installed
* Google Gemini API key

### How to Run

1. **Clone the project**

   ```bash
   git clone git@github.com:konradszp/AI_Assistant.git
   cd AI_Assistant
   ```

2. **Add API key**

    Make a .env file and add API key generated on Google AI Studio

    ```bash
    GOOGLE_API_KEY=google_gemini_api_key
    ```

3. **Start with docker compose**

    ```bash
    docker-compose up --build
    ```

4. **Open in browser**

    Go to http://localhost:8501 to test the app

