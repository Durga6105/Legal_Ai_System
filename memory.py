import chromadb
from chromadb.config import Settings
from datetime import datetime

client = chromadb.Client(Settings(persist_directory="./chroma_db"))
collection = client.get_or_create_collection(name="legal_memory")

def create_memory(text, output, summary):
    importance = 5 if "missing" in output.lower() or "risky" in output.lower() else 1
    collection.add(
        documents=[text],
        metadatas=[{
            "analysis": output,
            "summary": summary,
            "importance": importance,
            "time": str(datetime.now())
        }],
        ids=[str(datetime.now().timestamp())]
    )

def retrieve_memories(query):
    results = collection.query(query_texts=[query], n_results=5)
    memories = results["metadatas"][0] if results["metadatas"] else []
    memories = sorted(memories, key=lambda x: x.get("importance", 1), reverse=True)
    return memories[:3]

def get_all_memories():
    return collection.get()