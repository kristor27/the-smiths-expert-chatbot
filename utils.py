import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.cassandra import Cassandra
from langchain_huggingface import HuggingFaceEmbeddings


# Base URL for the website
base_url = "http://www.passionsjustlikemine.com/"

# Function to scrape lyrics from a given song page URL
def scrape_lyrics(song_url):
  response = requests.get(song_url)
  soup = BeautifulSoup(response.text, 'html.parser')
  lyrics_block = soup.find('blockquote')
  if lyrics_block:
      lyrics = lyrics_block.get_text(separator="\n").strip()
      return lyrics
  return None

# Function to get all song links from the main lyrics page
def get_song_links():
  lyrics_page_url = base_url + "lyrics-smiths.htm"
  response = requests.get(lyrics_page_url)
  soup = BeautifulSoup(response.text, 'html.parser')
  song_links = []
  menu_div = soup.find('div', {'id': 'mmenu'})
  if menu_div:
      for a_tag in menu_div.find_all('a', href=True):
          song_links.append(base_url + a_tag['href'])
  return song_links

# Main function to scrape all lyrics and convert them to Langchain documents
def scrape_lyrics_to_langchain_documents():
  song_links = get_song_links()
  documents = []
  for song_url in song_links:
      song_name = song_url.split('/')[-1].replace('smiths-', '').replace('.htm', '').replace('-', ' ').title()
      lyrics = scrape_lyrics(song_url)
      if lyrics:
          song_document = Document(
              page_content=lyrics,
              metadata={"song_name": song_name, "url": song_url}
          )
          documents.append(song_document)
          print(f"Processed song: {song_name}")
      else:
          print(f"Could not scrape lyrics for: {song_name}")
  return documents

def store_documents_in_astradb(documents):
  # Initialize Cassandra connection
  # Note: We don't need to initialize here as it's done in the main app now

  # Split the documents if needed
  text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
      chunk_size=500, chunk_overlap=0
  )
  doc_splits = text_splitter.split_documents(documents)

  # Store documents in AstraDB
  embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
  astra_vector_store = Cassandra(
      embedding=embeddings,
      table_name="smiths_songs_all",
      session=None,
      keyspace=None
  )
  astra_vector_store.add_documents(doc_splits)
  return len(doc_splits)  # Return the number of inserted documents