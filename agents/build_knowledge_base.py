import chromadb
from resume_examples_data import RESUME_EXAMPLES

# This creates a persistent database stored on disk, inside a folder
# called "chroma_db" - so it only needs to be built once, not every run
client = chromadb.PersistentClient(path="./chroma_db")

# A "collection" is like a table - a named group of related vectors
collection = client.get_or_create_collection(name="resume_examples")

# Prepare the data for insertion
documents = [item["bullet"] for item in RESUME_EXAMPLES]
metadatas = [{"role": item["role"]} for item in RESUME_EXAMPLES]
ids = [f"example_{i}" for i in range(len(RESUME_EXAMPLES))]

# Add everything to the collection - Chroma automatically embeds 
# the text using its default embedding model under the hood
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print(f"Successfully added {len(documents)} resume examples to the knowledge base.")
print(f"Collection now contains {collection.count()} items.")