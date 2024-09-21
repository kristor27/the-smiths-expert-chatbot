import cassio
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import END, StateGraph, START
from typing import List, Literal
from typing_extensions import TypedDict
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings


# Global variables
astra_vector_store = None
astra_vector_index = None
retriever = None

def initialize_cassandra(token, database_id):
  global astra_vector_store, astra_vector_index, retriever
  cassio.init(token=token, database_id=database_id)
  
  # Initialize vector store after Cassandra connection is established
  embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
  astra_vector_store = Cassandra(
      embedding=embeddings,
      table_name="smiths_lyrics_new",
      session=None,
      keyspace=None
  )
  astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
  retriever = astra_vector_store.as_retriever()

# Set up Wikipedia search
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

# Data model for routing
class RouteQuery(BaseModel):
  datasource: Literal["vectorstore", "wiki_search"] = Field(
      ...,
      description="Route the query to the appropriate datasource."
  )

# Prompt for routing
system = """You are an expert at routing a user question to a vectorstore or Wikipedia.
The vectorstore contains lyrics of The Smiths. Use the vectorstore for questions about lyrics or song names.
Use Wikipedia for questions about the background of The Smiths."""
route_prompt = ChatPromptTemplate.from_messages(
  [
      ("system", system),
      ("human", "{question}"),
  ]
)

# Function to retrieve documents
def retrieve(state):
  global astra_vector_index
  if astra_vector_index is None:
      raise ValueError("Vector store not initialized. Please connect to the database first.")
  question = state["question"]
  documents = astra_vector_index.vectorstore.similarity_search(question)
  return {"documents": documents, "question": question}

def wiki_search(state):
  question = state["question"]
  docs = wiki.invoke({"query": question})
  wiki_results = [Document(page_content=docs, metadata={"source": "wikipedia"})]
  return {"documents": wiki_results, "question": question}

def route_question(state):
  question = state["question"]
  openai_api_key = state["openai_api_key"]
  llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-3.5-turbo")
  structured_llm_router = llm.with_structured_output(RouteQuery)
  question_router = route_prompt | structured_llm_router
  source = question_router.invoke({"question": question})
  if source.datasource == "wiki_search":
      return "wiki_search"
  elif source.datasource == "vectorstore":
      return "retrieve"

def generate_final_answer(state):
  openai_api_key = state["openai_api_key"]
  llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-3.5-turbo")
  prompt = PromptTemplate.from_template(
      "Using the following context, answer the initial question: {context}\n\nInitial Question: {question}"
  )
  context = "\n".join([
      doc.page_content if isinstance(doc, Document) else doc[2]
      for doc in state["documents"]
  ])
  chain = prompt | llm
  response = chain.invoke({"context": context, "question": state["question"]})
  return {
      "question": state["question"],
      "generation": response.content,
      "documents": state["documents"]
  }

class GraphState(TypedDict):
  question: str
  generation: str
  documents: List[str]
  openai_api_key: str

def create_app(openai_api_key):
  # Build the graph
  workflow = StateGraph(GraphState)
  workflow.add_node("wiki_search", wiki_search)
  workflow.add_node("retrieve", retrieve)
  workflow.add_node("generate_final_answer", generate_final_answer)
  workflow.add_conditional_edges(
      START,
      route_question,
      {
          "wiki_search": "wiki_search",
          "retrieve": "retrieve",
      },
  )
  workflow.add_edge("retrieve", "generate_final_answer")
  workflow.add_edge("wiki_search", "generate_final_answer")
  workflow.add_edge("generate_final_answer", END)

  # Compile the app
  return workflow.compile()

def run_app(app, question, openai_api_key):
    inputs = {"question": question, "openai_api_key": openai_api_key}
    tool_response = ""
    final_answer = ""
    for output in app.stream(inputs):
      for key, value in output.items():
          if key in ["retrieve", "wiki_search"]:
              tool_response = "\n".join([doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in value['documents']])
          elif key == "generate_final_answer":
              final_answer = value['generation']     
    return {"tool_response": tool_response, "final_answer": final_answer}