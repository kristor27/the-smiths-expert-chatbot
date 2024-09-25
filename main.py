import streamlit as st
from langgraph_implementation import create_app, run_app, initialize_cassandra
from utils import scrape_lyrics_to_langchain_documents, store_documents_in_astradb
from PIL import Image
from langchain_openai import ChatOpenAI
import random

# Streamlit app configuration (must be the first Streamlit command)
st.set_page_config(page_title="The Smiths Expert", layout="wide")

# Smiths quotes for random display
smiths_quotes = [
    "Panic on the streets of London...",
    "How soon is now?",
    "I am human and I need to be loved...",
    "There is a light that never goes out...",
    "Heaven knows I'm miserable now..."
]

# Custom CSS for Smiths-inspired theme
st.markdown("""
<style>
    /* Smiths-inspired color palette */
    :root {
        --primary-color: #1a1a1a;
        --secondary-color: #f1c40f;
        --text-color: #ecf0f1;
        --background-color: #2c3e50;
    }
    
    /* Global styles */
    body {
        color: var(--text-color);
        background-color: var(--background-color);
    }
    
    /* Headings */
    h1, h2, h3 {
        color: var(--secondary-color);
        font-family: 'Georgia', serif;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: var(--secondary-color);
        color: var(--primary-color);
        font-weight: bold;
    }
    
    /* Chat messages */
    .stTextInput>div>div>input {
        background-color: var(--primary-color);
        color: var(--text-color);
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: var(--primary-color);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'db_connected' not in st.session_state:
    st.session_state.db_connected = False
if 'openai_connected' not in st.session_state:
    st.session_state.openai_connected = False
if 'docs' not in st.session_state:
    st.session_state.docs = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to test OpenAI API connection
def test_openai_connection(openai_api_key):
    chat = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")
    try:
        response = chat([{"role": "user", "content": "Hello"}])
        print(response)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

# Sidebar
st.sidebar.image("im/the-smiths.jpg", use_column_width=True)
st.sidebar.title("The Smiths Expert")
st.sidebar.markdown(f"*{random.choice(smiths_quotes)}*")

# API Key inputs with improved styling
with st.sidebar.expander("API Configuration", expanded=False):
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    astra_db_token = st.text_input("Astra DB Token", type="password")
    astra_db_id = st.text_input("Astra DB ID")

# Connection buttons with improved styling
col1, col2 = st.sidebar.columns(2)
if col1.button("Connect to DB"):
    if astra_db_token and astra_db_id:
        try:
            initialize_cassandra(astra_db_token, astra_db_id)
            st.session_state.db_connected = True
            st.sidebar.success("Connected to AstraDB!")
        except Exception as e:
            st.sidebar.error(f"Failed to connect: {str(e)}")
    else:
        st.sidebar.warning("Enter Astra DB Token and ID")

if col2.button("Test OpenAI"):
    if openai_api_key:
        if test_openai_connection(openai_api_key):
            st.session_state.openai_connected = True
            st.sidebar.success("Connected to OpenAI API!")
        else:
            st.sidebar.error("Failed to connect. Check API key.")

# Data scraping and AstraDB population buttons
if st.sidebar.button("Scrape Lyrics", disabled=not st.session_state.db_connected):
    with st.spinner("Scraping The Smiths lyrics..."):
        st.session_state.docs = scrape_lyrics_to_langchain_documents()
        st.sidebar.success(f"Scraped {len(st.session_state.docs)} songs")

if st.sidebar.button("Populate AstraDB", disabled=not st.session_state.db_connected or st.session_state.docs is None):
    with st.spinner("Populating AstraDB with Smiths wisdom..."):
        inserted_count = store_documents_in_astradb(st.session_state.docs)
        st.sidebar.success(f"Inserted {inserted_count} Smiths documents")

# Display LangGraph
with st.sidebar.expander("LangGraph Visualization", expanded=False):
    graph_image = Image.open("im/langgraph.jpeg")
    st.image(graph_image, use_column_width=True)

# Main chat interface
st.title("Chat with The Smiths Expert")
st.markdown("*Ask me anything about The Smiths, their music, or lyrics!*")

# Display chat messages with custom styling
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input with Smiths-themed placeholder
if prompt := st.chat_input("How soon is now? Ask your question..."):
    if not st.session_state.db_connected or not st.session_state.openai_connected:
        st.error("Please connect to both AstraDB and OpenAI API before chatting.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run the LangGraph app
        with st.spinner("The Smiths Expert is cooking...Let Him Cook!"):
            app = create_app(openai_api_key)
            response = run_app(app, prompt, openai_api_key)

        # Display tool message
        with st.expander("Behind the Scenes", expanded=False):
            st.write(f"Tool name: {response['tool_name']}")
            st.code(response["tool_response"])

        # Display AI response with Smiths-themed styling
        with st.chat_message("assistant"):
            st.markdown(f"🎸 *{response['final_answer']}*")
        st.session_state.messages.append({"role": "assistant", "content": response["final_answer"]})


# Easter egg: Hidden Morrissey
if st.sidebar.button("🎤 Click here and win!", key="easter_egg"):
    st.balloons()
    st.sidebar.image("im/morrissey.jpg", caption="Surprise! It's Morrissey!")