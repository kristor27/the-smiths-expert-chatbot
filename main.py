import streamlit as st
from langgraph_implementation import create_app, run_app, initialize_cassandra
from utils import scrape_lyrics_to_langchain_documents, store_documents_in_astradb
from PIL import Image
from langchain_openai import ChatOpenAI

# Streamlit app configuration
st.set_page_config(page_title="The Smiths Expert", layout="wide")

# Sidebar
st.sidebar.title("The Smiths Expert")
st.sidebar.image("im/the-smiths.jpg", use_column_width=True)

# API Key inputs
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
astra_db_token = st.sidebar.text_input("Astra DB Token", type="password")
astra_db_id = st.sidebar.text_input("Astra DB ID")

# Initialize session state variables
if 'db_connected' not in st.session_state:
  st.session_state.db_connected = False
if 'openai_connected' not in st.session_state:
  st.session_state.openai_connected = False
if 'docs' not in st.session_state:
  st.session_state.docs = None

# Function to test OpenAI API connection
def test_openai_connection(openai_api_key):
    chat = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")
    try:
        # Send a message to the model and get a response
        response = chat([{"role": "user", "content": "Hello"}])
        print(response)  # Output the response for debugging
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


# Connect to DB button
if st.sidebar.button("Connect to DB"):
  if astra_db_token and astra_db_id:
      try:
          initialize_cassandra(astra_db_token, astra_db_id)
          st.session_state.db_connected = True
          st.sidebar.success("Successfully connected to AstraDB!")
      except Exception as e:
          st.sidebar.error(f"Failed to connect to AstraDB: {str(e)}")
  else:
      st.sidebar.warning("Please enter Astra DB Token and ID")

# Test OpenAI API connection
if openai_api_key:
  if test_openai_connection(openai_api_key):
      st.session_state.openai_connected = True
      st.sidebar.success("Successfully connected to OpenAI API!")
  else:
      st.sidebar.error("Failed to connect to OpenAI API. Please check your API key.")

# Data scraping and AstraDB population buttons
if st.sidebar.button("Scrape Lyrics", disabled=not st.session_state.db_connected):
  with st.spinner("Scraping lyrics..."):
      st.session_state.docs = scrape_lyrics_to_langchain_documents()
      st.sidebar.success(f"Scraped {len(st.session_state.docs)} songs")

if st.sidebar.button("Populate AstraDB", disabled=not st.session_state.db_connected or st.session_state.docs is None):
  with st.spinner("Populating AstraDB..."):
      inserted_count = store_documents_in_astradb(st.session_state.docs)
      st.sidebar.success(f"Successfully inserted {inserted_count} documents into AstraDB")

# Display LangGraph
st.sidebar.subheader("LangGraph Visualization")
graph_image = Image.open("im/langgraph.jpeg")
st.sidebar.image(graph_image, use_column_width=True)

# Main chat interface
st.title("Chat with The Smiths Expert")

# Initialize chat history
if "messages" not in st.session_state:
  st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
      st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about The Smiths"):
  if not st.session_state.db_connected or not st.session_state.openai_connected:
      st.error("Please connect to both AstraDB and OpenAI API before chatting.")
  else:
      st.session_state.messages.append({"role": "user", "content": prompt})
      with st.chat_message("user"):
          st.markdown(prompt)

      # Run the LangGraph app
      app = create_app(openai_api_key)
      response = run_app(app, prompt, openai_api_key)  # Add openai_api_key here

      # Display tool message
      with st.expander("Tool Message", expanded=False):
          st.code(response["tool_response"])

      # Display AI response
      with st.chat_message("assistant"):
          st.markdown(response["final_answer"])
      st.session_state.messages.append({"role": "assistant", "content": response["final_answer"]})

# Footer
st.markdown("---")
st.markdown("Powered by LangChain and Streamlit")