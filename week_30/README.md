# 🧠 Text Moderation System with Embeddings

A production-ready, notebook-based system for moderating chat or user-generated content using a lightweight embedding-classifier pipeline—built for speed, cost-efficiency, and deterministic performance.

---

## 🔍 Why Use Embeddings Over LLMs?

Unlike large language models, embedding-based classifiers provide:

- ⚡ **Ultra-low latency** — Inference in milliseconds on CPU.
- 💸 **Predictable cost** — Fixed-size vectors, no per-token charges.
- 🌍 **Multilingual support** — Works well across dozens of languages.
- 📊 **Deterministic output** — Always the same label for the same input.
- 🔁 **Easy retraining** — Fine-tune with new labels and a `.fit()` call.

LLMs are best suited as a second-pass reviewer, not for first-line filtering in high-volume systems.

---

## 📦 Installation

Install the required Python packages:

```bash
pip install -q sentence-transformers scikit-learn pandas matplotlib openai tqdm
```
# 📁 Project Structure
```bash
├── moderation_dataset.csv        # Dataset with 'text' and 'label' columns
├── text_moderation_pipeline.ipynb  # Jupyter Notebook with step-by-step code
├── .env                          # (optional) Stores your OpenAI API key
└── README.md
```
# ⚙️ Environment Setup

Set your OpenAI API key as an environment variable:

Option A: Export it manually

```bash
export OPENAI_API_KEY=your_key_here
Option B: Use a .env file
```
```bash
OPENAI_API_KEY=your_key_here
Then load it in the notebook using:
```

```bash
from dotenv import load_dotenv
load_dotenv()
```
# 🧪 Dataset

The dataset should include:

text: the message content

label: binary flag (1 = requires moderation, 0 = acceptable)

**Example**:
```bash
text,label
"You're an idiot",1
"Let's play soccer later",0
```
# 🛠️ Pipeline Overview

Text preprocessing (if needed)

Embedding generation using text-embedding-3-small (256D)

Classifier training using LogisticRegression

Evaluation on hold-out test data

# 📈 Sample Results

With a balanced dataset and logistic regression:

- Accuracy: 87%

- F1-score (toxic): 0.873

- F1-score (non-toxic): 0.867

The model is both accurate and fair—especially suitable for live moderation use cases.

# 🔁 Retraining

To update for new policy changes:

```bash
# Add new labeled data
# Create new embeddings
# Re-train:
```

```bash
clf.fit(X_train, y_train)
```
# 📊 Visualization
Include optional performance plots using:

```bash
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_estimator(clf, X_test, y_test)
```

# 🛡️ License

MIT — free to use, modify, and distribute.

#  Author
- Simanga Mchunu