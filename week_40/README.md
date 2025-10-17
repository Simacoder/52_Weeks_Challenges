# RAG and Word Vector Analogy Project

This project demonstrates two main concepts: **word vector visualizations** and **Retrieval-Augmented Generation (RAG)** using embeddings.  

It includes a visualization of word vectors and a basic implementation of RAG with detailed explanation about embeddings and retrieval metrics.

---

## Project Structure

- `cosine_sim.py` – Visualizes word vectors and the famous analogy:  
  `king - man + woman ≈ queen`  
  - Plots each word as a colored vector in 2D space.  
  - Illustrates the vector arithmetic visually.  

- `download.py` – Downloads large text files (e.g., *War and Peace*) in chunks to avoid network issues.  

---

## Word Vector Analogy

The analogy visualization uses the following word vectors:

| Word  | Vector       | Color   |
|-------|-------------|---------|
| king  | [0.25, 0.75] | Red     |
| queen | [0.23, 0.77] | Blue    |
| man   | [0.15, 0.80] | Green   |
| woman | [0.13, 0.82] | Purple  |

The script performs the vector operation:



king - man + woman = [0.23, 0.77] ≈ queen


This demonstrates how word embeddings capture semantic relationships.

---

## RAG Explained: Understanding Embeddings, Similarity, and Retrieval

Retrieval-Augmented Generation (RAG) enhances language models by retrieving relevant pieces of text from a knowledge base. Here's how it works:

### How Retrieval Works

1. **Chunking** – Text is split into smaller chunks.
2. **Embedding** – Each chunk is converted into a vector using an embedding model.  
   - Embeddings map text to a high-dimensional space, where semantically similar chunks are closer together.
   - Example: `"happy"` and `"joyful"` have vectors close to each other, while `"sad"` is far away.
3. **Similarity Search** – When a query is made:
   - By default, the script retrieves chunks using **L2 distance**.
   - It retrieves the **top k=4 most similar chunks** to the query.

### Using Cosine Similarity

To use cosine similarity instead of L2 distance:
- **Normalize the embeddings** for both the user query and the knowledge base.  
- Configure the vector store to use **dot product** as the similarity metric.  

This change ensures retrieval focuses on **directional similarity** rather than absolute distance in vector space.

---

## Getting Started

1. **Clone the repository**:

```bash
git clone https://github.com/Simacoder/52_Weeks_Challenges.git
cd 52_Weeks_Challenges
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Download a sample text (optional):
```bash
python download.py
```

Visualize word vectors:

```bash
python cosine_sim.py
```

This will generate a file word_vector_analogy.png with the vector visualization.

### Requirements

- Python 3.10+

- numpy

- matplotlib

- requests

> [!NOTE]

> Large text downloads are handled in **chunks** to prevent incomplete read errors.

> RAG retrieval metrics can be **switched between L2 and cosine similarity** depending on the application.

### References

[Project Gutenberg: War and Peace](https://www.gutenberg.org/ebooks/2600)

[Introduction to Word Embeddings](https://www.tensorflow.org/text/guide/word_embeddings)

[RAG: Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)