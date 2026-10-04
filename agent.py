from dotenv import load_dotenv, find_dotenv
import os
import time
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableSequence
from pinecone import Pinecone, ServerlessSpec
from langchain_core.tracers.context import tracing_v2_enabled

# Configure logging
logging.basicConfig(level=logging.INFO)

#Load the environment variables from the .env file
load_dotenv(find_dotenv(".env"))

#Acess the environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

#Ensure that the environment variables are loaded correctly
if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

if PINECONE_API_KEY is None:
    raise ValueError("PINECONE_API_KEY is not set in the environment variables.")       

print("Environment variables loaded successfully.")

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
os.environ["LANGSMITH_TRACING_V2"] = os.getenv("LANGSMITH_TRACING_V2")
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'

# print(os.environ["OPENAI_API_KEY"])

# Indexing the document(Static Indexing)
# Initializing Pinecone client and logging

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = 'rag-agent-index'

# Check if the index already exists if not create it
def ensure_index():
    existing_indexes = [index["name"] for index in pc.list_indexes()]
    if index_name in existing_indexes:
        logging.info(f"Index '{index_name}' already exists. Skipping creation.")
    else:
        logging.info(f"Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        time.sleep(5)  # Ensure index is ready

# index = pc.Index(index_name)
# print(index.describe_index_stats())

# Function to load and split documents  
def load_and_split_documents(filepath):
    logging.info("Loading document...")
    loader = PyPDFLoader(filepath)
    docs = loader.load()

    logging.info("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    splits = text_splitter.split_documents(docs)
    return {"all_splits": splits, "total_Splits:": len(splits), "message": "Documents loaded and split successfully!"}

# Function to embed documents
def embed_documents(inputs):
    splits = inputs["all_splits"]
    logging.info("Generating embeddings...")
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
    embeddings = embeddings_model.embed_documents([split.page_content for split in splits])

    # Compute norms (to ensure embeddings aren't garbage)
    norms = [sum(e[:5]) for e in embeddings[:5]]  # Get first 5 embeddings' norm
    
    return {"all_embeddings": embeddings, "norms": norms, "message": "Embeddings generated successfully!"}

# Function to upsert embeddings into Pinecone
def upsert_embeddings(data):  # This function expects a dictionary
    splits = data["splits"]["all_splits"]
    embeddings = data["embeddings"]["all_embeddings"]
    logging.info(f"Upserting {len(embeddings)} documents into Pinecone...")
    index = pc.Index(index_name)

    vectors = [
        {
            "id": f"doc_{split.metadata.get('source')}_{i}_{split.metadata.get('page_label', 'no_label')}",  # Ensure ID is valid and unique(because vectors of same Id get overwritten by the latest vector)
            "values": emb,
            "metadata": {"text": split.page_content}
        }
        for i, (split, emb) in enumerate(zip(splits, embeddings)) if len(emb) > 0
    ]

    BATCH_SIZE = 100  # Recommended batch size
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i:i + BATCH_SIZE]
        index.upsert(vectors=batch, namespace='ns1')
        logging.info(f"Upserted batch {i // BATCH_SIZE + 1} of {len(vectors) // BATCH_SIZE + 1}")
    
    logging.info(f"Upserted {len(vectors)} vectors into the vector store.")
    return f"Upserted {len(vectors)} vectors into the vector store."

# Using Langchain's RunnableSequence to create a pipeline for document processing
# Turn Functions into Runnables
load_split_runnable = RunnableLambda(load_and_split_documents)
embed_runnable = RunnableLambda(embed_documents)
upsert_runnable = RunnableLambda(upsert_embeddings)

# Using LangChain's RunnableSequence to create the Indexing chain
# indexing_chain = RunnableSequence(load_split_runnable, embed_runnable, upsert_runnable)


# Using LangChain's | Operator for an Indexing Chain
indexing_chain = (
    load_split_runnable 
    | {
        "splits": RunnablePassthrough(),
        "embeddings": embed_runnable
    }
    | upsert_runnable
)

# Run the indexing pipeline
def run_indexing_pipeline(filepath):
    ensure_index()

    with tracing_v2_enabled(project_name=os.getenv("LANGSMITH_PROJECT", "default")):
        indexing_chain.invoke(filepath)
    
    logging.info("Indexing pipeline completed successfully!")

run_indexing_pipeline("./data/BOFA_safedepositbox_disclosures.pdf")

# Retrieval and Generation (RAG) Pipeline

# Defining the embedding  model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
index = pc.Index(index_name)

# Creating the Retriever
def retriever(question):
    # always include the question in []. because embed_documents expects a list. 
    # If its not a list, then each character will be treated as a separate document that needs to be embedded seperately.
    embeddedQuestion = embeddings_model.embed_documents([question]) 
    similar_docs = index.query(vector=embeddedQuestion, top_k=3, namespace="ns1", include_metadata=True)
    return similar_docs



