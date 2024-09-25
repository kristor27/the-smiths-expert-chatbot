# The Smiths Expert

The Smiths Expert is an AI-powered chatbot application designed to answer questions about the iconic band, The Smiths. Built using Langgraph, this application leverages advanced language models and a sophisticated routing system to provide accurate and context-rich responses.

## Features

- **AI-Powered Chatbot**: Utilizes OpenAI's GPT-3.5 Turbo to generate responses.
- **LangGraph Workflow**: Manages the workflow of queries using a graph-based approach.
- **Query Routing**: Directs queries to the appropriate tool based on the question type.
  - **Wiki Search**: Fetches and parses Wikipedia pages for context.
  - **Retrieve (RAG)**: Retrieves song lyrics from a vector database for context.
- **Lyrics Scraping**: Scrapes and stores all lyrics of The Smiths' songs in a vector database (Astra DB).
- **Contextual Responses**: Provides answers using either Wikipedia content or song lyrics, depending on the query.

## How It Works

1. **User Interaction**: Users interact with the chatbot through a simple Streamlit interface.
2. **Query Routing**: The application routes the query to one of two tools:
   - **Wiki Search**: If the query is general or historical, the application fetches the relevant Wikipedia page, parses it, and uses it as context for the response.
   - **Retrieve (RAG)**: If the query is related to song lyrics, the application retrieves the top K (K=1) relevant documents from the vector database and uses them as context.
3. **Response Generation**: The selected context is fed into the OpenAI GPT-3.5 Turbo model to generate a comprehensive answer.

## Setup and Installation

1. **Clone the Repository**:
git clone https://github.com/kristor27/the-smiths-expert-chatbot.git


2. **Install Dependencies**:
pip install -r requirements.txt

3. **Run the Application**:
streamlit run app.py

4. **Environment Variables**:
   - Set up your OpenAI API key.
   - Configure access to your Astra DB vector-db for vector storage and retrieval.


## Contact and collaborations

Feel free to contact me if you want collaboration on llm-based ml engineering projects. my email: hareb.idir@gmail.com.
