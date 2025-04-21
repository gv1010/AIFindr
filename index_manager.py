import os
import nltk
import pandas as pd
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from vector_db import VectorDB

class IndexManager:
    def __init__(self, docs_dir: str = "./documents", chunk_size: int = 1000, csv_file = "/home/gv/work/LOCAL_RAG (copy)/documents/data.csv", chunk_overlap: int = 0):
        """Initialize index manager with document directory, splitter, and optional CSV file."""
        self.docs_dir = docs_dir
        self.csv_file = csv_file if csv_file else os.path.join(docs_dir, "data.csv")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_and_index_documents(self, vector_db: VectorDB):
        """Load documents and CSV, split into chunks with metadata, and index into vector DB."""
        # Ensure documents directory exists
        if not os.path.exists(self.docs_dir):
            os.makedirs(self.docs_dir)
            # Create sample CSV
            sample_data = pd.DataFrame({
                "id": ["sample_1"],
                "text": ["Sample document for RAG. This is a test document to demonstrate the RAG system. It contains some basic information about LangChain and Ollama."]
            })
            sample_data.to_csv(self.csv_file, index=False)
            print(f"Created sample CSV in {self.csv_file}")

        texts = []

        # Load CSV if it exists
        if os.path.exists(self.csv_file):
            try:
                df = pd.read_csv(self.csv_file)
                if df.shape[1] < 2:
                    raise ValueError("CSV must have at least two columns: metadata and text.")
                for _, row in df.iterrows():
                    metadata_value = str(row.iloc[0])  # First column as metadata
                    text = str(row.iloc[1])  # Second column as text
                    # Create a Document for the text
                    doc = Document(
                        page_content=text,
                        metadata={"source": self.csv_file, "csv_metadata": metadata_value}
                    )
                    # Split into chunks
                    chunks = self.text_splitter.split_documents([doc])
                    # Add chunk-wise metadata
                    for i, chunk in enumerate(chunks):
                        chunk_metadata = chunk.metadata.copy()
                        chunk_metadata.update({
                            "chunk_index": i,
                            "chunk_id": f"{os.path.basename(self.csv_file)}_row_{row.name}_chunk_{i}",
                            "name": row.name
                        })
                        texts.append(Document(page_content=chunk.page_content, metadata=chunk_metadata))
            except Exception as e:
                print(f"Error processing CSV {self.csv_file}: {e}")
                raise SystemExit("Failed to process CSV. Ensure it has valid data with at least two columns.")

        # # Load .txt documents
        # try:
        #     loader = DirectoryLoader(self.docs_dir, glob="*.txt", loader_cls=TextLoader)
        #     documents = loader.load()
        # except Exception as e:
        #     print(f"Error loading .txt documents: {e}")
            # documents = []  # Continue with CSV data if .txt fails

        # # Split .txt documents and add metadata
        # for doc in documents:
        #     chunks = self.text_splitter.split_documents([doc])
        #     for i, chunk in enumerate(chunks):
        #         chunk_metadata = chunk.metadata.copy() if chunk.metadata else {}
        #         chunk_metadata.update({
        #             "source": doc.metadata.get("source", "unknown"),
        #             "chunk_index": i,
        #             "chunk_id": f"{os.path.basename(doc.metadata.get('source', 'unknown'))}_chunk_{i}"
        #         })
        #         texts.append(Document(page_content=chunk.page_content, metadata=chunk_metadata))

        if not texts:
            raise SystemExit("No valid documents or CSV data found to index.")

        # Index documents into vector DB
        vector_db.load_or_create_vectorstore(texts)