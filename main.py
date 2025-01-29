from typing import Annotated, List
from typing_extensions import TypedDict
import os
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import ToolMessage
from langchain_core.messages import HumanMessage
import groq


# --- Define State ---
class State(TypedDict):
    messages: Annotated[List, add_messages]
    document_text: str
    chunks: List[str]
    embeddings: List
    relevant_context: str

# Initialize StateGraph
graph_builder = StateGraph(State)

# 1. Document Upload Node
def upload_document(state: State) -> dict:
    last_message = state["messages"][-1]

    if isinstance(last_message, HumanMessage):
        file_path = last_message.content
    else:
        raise ValueError("Invalid message format. Expected a HumanMessage object.")

    # Check if it's actually a file path (avoid processing queries)
    if not os.path.exists(file_path):
        print(f"Skipping upload. Not a valid file path: {file_path}")  # Debugging
        return {}  # Prevent further processing

    print(f"Processing file: {file_path}")  # Debugging
    
    with open(file_path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        document_text = "".join([page.extract_text() for page in reader.pages])
    
    return {"document_text": document_text}



# 2. Document Parsing Node
def parse_document(state: State) -> dict:
    document_text = state["document_text"]
    chunk_size = 500
    chunks = [document_text[i:i + chunk_size] for i in range(0, len(document_text), chunk_size)]
    return {"chunks": chunks}



# 3. Text-to-Embedding Node
def text_to_embeddings(state: State) -> dict:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Convert numpy arrays to lists for serialization
    embeddings = model.encode(state["chunks"], show_progress_bar=True).tolist()
    
    return {"embeddings": embeddings}



# 4. Embedding-to-VectorDB Node
class VectorDB:
    def __init__(self, vector_dim):
        self.index = faiss.IndexFlatL2(vector_dim)

    def store(self, embeddings: list):
        embeddings = np.array(embeddings).astype("float32")
        self.index.add(embeddings)

    def query(self, query_embedding: list, top_k: int = 5):
        query_embedding = np.array(query_embedding).astype("float32").reshape(1, -1)
        distances, indices = self.index.search(query_embedding, top_k)
        return distances, indices

vector_db = VectorDB(vector_dim=384)


def store_embeddings(state: State) -> dict:
    embeddings = np.array(state["embeddings"]).astype("float32")  # Convert back to numpy for faiss
    vector_db.store(embeddings)
    return {}



# 5. Query VectorDB Node

def query_embeddings(state: State) -> dict:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    last_message = state["messages"][-1]

    if isinstance(last_message, HumanMessage):
        query_text = last_message.content
    else:
        raise ValueError("Invalid message format. Expected a HumanMessage object.")

    query_embedding = model.encode([query_text])
    distances, indices = vector_db.query(query_embedding[0])

    # Debugging: Print retrieved indices and distances
    print(f"Query: {query_text}")
    print(f"Retrieved Indices: {indices}")
    print(f"Distances: {distances}")

    # Handle case where no matches are found
    if len(indices) == 0 or len(indices[0]) == 0:
        print("No relevant document chunks found for the query.")
        return {"relevant_context": "No relevant information found in the document."}

    relevant_context = " ".join([state["chunks"][i] for i in indices[0] if i < len(state["chunks"])])
    return {"relevant_context": relevant_context}







#groq key = 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




def query_llm(state: State) -> dict[str, str]:
    if not state.messages: 
        return {"response": "No messages found to process."}

    query = state.messages[-1].content
    trimmed_context = state.relevant_context[:1000]

    client = groq.Client(api_key="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")

    response = client.chat.completions.create(
        model="gemma2-9b-It",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {trimmed_context}\n\nQuery: {query}"}
        ],
        max_tokens=200
    )

    if response and response.choices:
        return {"response": response.choices[0].message.content}

    return {"response": "I'm sorry, but I couldn't generate an answer."}










graph_builder.add_node("store_embeddings", store_embeddings)
graph_builder.add_node("query_vector_db", query_embeddings)
graph_builder.add_node("text_to_embeddings", text_to_embeddings)
graph_builder.add_node("parse_document", parse_document)
graph_builder.add_node("upload_document", upload_document)
graph_builder.add_node("query_llm", query_llm)

graph_builder.add_edge(START, "upload_document")
graph_builder.add_edge("upload_document", "parse_document")
graph_builder.add_edge("parse_document", "text_to_embeddings")
graph_builder.add_edge("text_to_embeddings", "store_embeddings")
graph_builder.add_edge("text_to_embeddings", "query_vector_db")
graph_builder.add_edge("query_vector_db", "query_llm")
graph_builder.add_edge("query_llm", END)


graph = graph_builder.compile(checkpointer=memory)

# Execute the Workflow
if __name__ == "__main__":
    file_path = "C:/Users/trive/OneDrive/Desktop/Learn the basics.pdf"  # Ensure this file exists

    upload_inputs = {"messages": [{"role": "user", "content": file_path}]}
    config = {"configurable": {"thread_id": "1"}}
    
    print(f"Uploading and processing: {file_path}")
    graph.invoke(upload_inputs, config)

    query_input = {"messages": [{"role": "user", "content": "What is the main topic of the document?"}]}
    
    print("Querying the document...")
    query_output = graph.invoke(query_input, config)

    answer = query_output.get("response", "No answer was generated.")
    print("Answer:", answer)

    
