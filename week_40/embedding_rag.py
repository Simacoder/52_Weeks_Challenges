import os
from langchain_community.chat_models import ChatHuggingFace
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

#  choose a local open-source model
# smaller options: "microsoft/phi-2", "mistralai/Mistral-7B-Instruct-v0.2"
# larger options need more VRAM (≥16GB)
model_id = "mistralai/Mistral-7B-Instruct-v0.2"

#  initialize Hugging Face LLM
print("Loading model... (this may take a minute)")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=500,
    temperature=0.3,
)
llm = ChatHuggingFace(pipeline=pipe)

#  initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#  load text documents
text_folder = "RAG files"
documents = []
for filename in os.listdir(text_folder):
    if filename.lower().endswith(".txt"):
        file_path = os.path.join(text_folder, filename)
        loader = TextLoader(file_path)
        documents.extend(loader.load())

#  split text into smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = []
for doc in documents:
    chunks = splitter.split_text(doc.page_content)
    for chunk in chunks:
        split_docs.append(Document(page_content=chunk))
documents = split_docs

#  create vector database
vector_store = FAISS.from_documents(documents, embeddings)
retriever = vector_store.as_retriever()

def main():
    print("Welcome to the Local RAG Assistant (Hugging Face Edition). Type 'exit' to quit.\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Exiting…")
            break

        # retrieve context
        relevant_docs = retriever.invoke(user_input)
        retrieved_context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # build prompt
        system_prompt = (
            "You are a helpful assistant. "
            "Use ONLY the following knowledge base context to answer the user. "
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{retrieved_context}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        # generate response
        response = llm.invoke(messages)
        print(f"\nAssistant: {response.content.strip()}\n")

if __name__ == "__main__":
    main()
