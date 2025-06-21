# Import necessary libraries
from sklearn.cluster import KMeans  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt

# Function to perform K-Means clustering
def KMeans_Algorithm(data, K):
    df = pd.DataFrame(data, columns=["X", "Y"])
    kmeans = KMeans(
        n_clusters=K,
        init='k-means++',
        max_iter=300,
        random_state=2021
    )
    kmeans.fit(df)
    df["labels"] = kmeans.labels_
    return df, kmeans.cluster_centers_

# Function to perform Elbow Method
def Elbow_Method(data):
    inertia = []
    K = range(1, 10)
    df = pd.DataFrame(data, columns=["X", "Y"])
    for k in K:
        model = KMeans(n_clusters=k, random_state=2022)
        model.fit(df)
        inertia.append(model.inertia_)
    return inertia

# Generate synthetic data
X1 = np.random.randint(0, 4, size=(300, 1))  
X2 = np.random.uniform(0, 10, size=(300, 1))  
data = np.append(X1, X2, axis=1)

# Apply Elbow Method
inertia = Elbow_Method(data)
K = range(1, 10)

# Plot Elbow Curve
plt.figure(figsize=(10, 5))
plt.plot(K, inertia, 'bx-')
plt.xlabel("K: Number of clusters")
plt.ylabel("Inertia")
plt.title("K-Means: Elbow Method")
plt.grid(True)
plt.show()

# Apply K-Means clustering
df, centroids = KMeans_Algorithm(data, K=4)

# Plot Clusters
fig, ax = plt.subplots(figsize=(7, 6))
colors = ['black', 'green', 'red', 'yellow']
for label in range(4):
    plt.scatter(
        df[df["labels"] == label]["X"],
        df[df["labels"] == label]["Y"],
        c=colors[label],
        label=f'Cluster {label + 1}',
        alpha=0.7
    )

# Plot centroids
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker='*',
    s=300,
    c='blue',
    label='Centroids'
)

# Add horizontal labels next to centroids
for idx, (x, y) in enumerate(centroids):
    plt.annotate(
        f'Cluster {idx + 1}',
        (x + 0.3, y),  # Offset label to right of point
        fontsize=10,
        color=colors[idx],
        ha='left',
        va='center'
    )

# Final plot settings
plt.legend()
plt.xlim([-1, 5.5])  # Adjust to ensure no overlap
plt.ylim([0, 11])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('K-Means Clustering Visualization')
ax.set_aspect('equal')
plt.grid(True)
plt.tight_layout()
plt.show()
