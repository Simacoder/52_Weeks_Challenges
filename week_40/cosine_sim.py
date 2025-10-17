import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Define word vectors
vectors = {
    "king": np.array([0.25, 0.75]),
    "queen": np.array([0.23, 0.77]),
    "man": np.array([0.15, 0.80]),
    "woman": np.array([0.13, 0.82])
}

# Colors for each word
colors = {
    "king": "red",
    "queen": "blue",
    "man": "green",
    "woman": "purple"
}

# Create figure
plt.figure(figsize=(8, 8))

# Plot each word vector
for word, vec in vectors.items():
    plt.arrow(0, 0, vec[0], vec[1], head_width=0.005, head_length=0.005,
              fc=colors[word], ec=colors[word], length_includes_head=True)
    plt.text(vec[0]+0.005, vec[1]+0.005, word, fontsize=12, color=colors[word])

# Perform analogy: king - man + woman
analogy_vec = vectors["king"] - vectors["man"] + vectors["woman"]
plt.arrow(0, 0, analogy_vec[0], analogy_vec[1], head_width=0.005, head_length=0.005,
          fc="orange", ec="orange", linestyle='--', length_includes_head=True)
plt.text(analogy_vec[0]+0.005, analogy_vec[1]+0.005,
         "king - man + woman ≈ queen", fontsize=10, color="orange")

# Set limits and labels
plt.xlim(0, 0.3)
plt.ylim(0.7, 0.85)
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("Word Vector Analogy Visualization")
plt.grid(True)

# Save the figure instead of showing it
plt.savefig("word_vector_analogy.png", dpi=300)
print("Plot saved as word_vector_analogy.png")
