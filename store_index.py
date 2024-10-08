from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

directory_path = r"D:\END-TO-END-MEDICAL-CHATBOT-USING-LLAMA2\Data\Medical-book.pdf"

extracted_pdf = load_pdf(directory_path)
text_chunks = text_split(extracted_pdf)
embeddings = download_hugging_face_embeddings()


# Initialize Pinecone using the Pinecone class
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY"),  # Or use your PINECONE_API_KEY variable
    environment=os.environ.get("PINECONE_API_ENV")  # Or use your PINECONE_API_ENV variable
)

# Now, check for existing indexes or create a new one if needed
if 'my-index' not in pc.list_indexes().names():
    pc.create_index(
        index_name='medical-chatbot',
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            # Try using a different region that is supported by the free plan
            region='us-east-1'
        )
    )


index_name="medical-chatbot"

#Creating Embeddings for Each of The Text Chunks & storing
docsearch = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
