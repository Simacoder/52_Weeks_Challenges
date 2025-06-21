# 🧠 Introduction to Unsupervised Learning: K-Means Clustering

Welcome to the **Unsupervised Learning with K-Means Clustering** project! This repository serves as an educational guide and hands-on implementation of one of the most popular unsupervised machine learning algorithms — **K-Means**.

---

## 📚 Project Overview

Unsupervised learning is a type of machine learning where models uncover patterns in data **without labeled outputs**. K-Means clustering is one of the fundamental techniques used for **discovering structure** in unlabeled datasets by grouping similar data points into clusters.

In this project, you will:
- Understand the concept of unsupervised learning
- Explore the theory behind K-Means clustering
- Apply K-Means to synthetic datasets
- Visualize and evaluate clustering performance
- Use the Elbow Method to determine the optimal number of clusters

---

## 📁 Project Structure

```bash
    unsupervised-learning-kmeans
│
├── means.py # Main Python script for K-Means clustering
├── README.md # This documentation file
└── requirements.txt # Required Python libraries
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/simacoder/52_weeks_challenges.git
cd 52_Weeks_Challenges
cd week_25

```

### 2. Install dependencies

```bash
    pip install -r requirements.txt
```

# 🧪 How It Works

**K-Means Algorithm Steps**:
- Select the number of clusters K

- Initialize K centroids randomly

- Assign each data point to the nearest centroid

- Recalculate centroids as the mean of all assigned points

- Repeat steps 3–4 until convergence

**Elbow Method**:

Used to find the optimal number of clusters by plotting inertia (within-cluster sum of squares) vs. K.

# 📊 Visual Output

**The script generates**:

- 📈 **Elbow plot**: Shows optimal K value

- 🟢 **Cluster scatter plot**: Displays clustered data and centroids

# 💡 Example Use Cases
- Customer segmentation

- Image compression

- Document categorization

- Social network analysis

- Market basket analysis

# 📌 Run the Script
To execute the clustering and view the results:

```bash
    python means.py
```

# ✍️ Author
- Simanga Mchunu

- [LinkedIn](https://www.linkedin.com/in/simanga-mchunu-7570078a/)

- [GitHub](https://github.com/Simacoder)

# 📝 License

This project is open-source and available under the [MIT License](https://mit-license.org/).

# 📎 References
- [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/clustering.html#k-means)

- [Unsupervised Learning Overview - Wikipedia](https://en.wikipedia.org/wiki/Unsupervised_learning)