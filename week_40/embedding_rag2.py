import os
import numpy as np
import faiss
from langchain_community.chat_models import ChatHuggingFace
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ========== MODEL SETUP ==========
model_id = "microsoft/Phi-3-mini-4k-instruct"  

print("Loading model... (this may take a minute)")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=500, temperature=0.3)
llm = ChatHuggingFace(pipeline=pipe)

# ========== EMBEDDINGS SETUP ==========
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ========== DOCUMENT LOADING ==========
text_folder = "RAG files"
documents = []
for filename in os.listdir(text_folder):
    if filename.lower().endswith(".txt"):
        file_path = os.path.join(text_folder, filename)
        loader = TextLoader(file_path)
        documents.extend(loader.load())

# ========== CHUNKING ==========
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = []
for doc in documents:
    chunks = splitter.split_text(doc.page_content)
    for chunk in chunks:
        split_docs.append(Document(page_content=chunk))
documents = split_docs

# ========== NORMALIZATION FUNCTION ==========
def normalize(vectors):
    vectors = np.array(vectors)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

# ========== EMBED + NORMALIZE DOCUMENTS ==========
doc_texts = [doc.page_content for doc in documents]
doc_embeddings = embeddings.embed_documents(doc_texts)
doc_embeddings = normalize(doc_embeddings)

# ========== CREATE FAISS INDEX (INNER PRODUCT = COSINE SIM) ==========
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product index
index.add(doc_embeddings)

# Simple docstore mapping
docstore = {i: doc for i, doc in enumerate(documents)}

# ========== INTERACTIVE LOOP ==========
def main():
    print("Welcome to the RAG Assistant (Cosine Similarity Edition). Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Exiting…")
            break

        # Embed + normalize query
        query_embedding = embeddings.embed_query(user_input)
        query_embedding = normalize([query_embedding])

        # Retrieve top k chunks (you can change k)
        k = 5
        D, I = index.search(query_embedding, k)

        # Get relevant documents
        relevant_docs = [docstore[i] for i in I[0]]
        retrieved_context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Print retrieved chunks with cosine scores
        print("\nTop 5 chunks and their cosine similarity scores:\n")
        for rank, (idx, score) in enumerate(zip(I[0], D[0]), start=1):
            print(f"Chunk {rank}:")
            print(f"Cosine similarity: {score:.4f}")
            print(f"Content:\n{docstore[idx].page_content[:400]}...\n{'-'*60}")

        # Build system prompt
        system_prompt = (
            "You are a helpful assistant. "
            "Use ONLY the following knowledge base context to answer the user. "
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{retrieved_context}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        # Generate response
        response = llm.invoke(messages)
        print(f"\nAssistant: {response.content.strip()}\n")

if __name__ == "__main__":
    main()
