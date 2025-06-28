import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set the style
sns.set(style="whitegrid")

# Generate data
np.random.seed(0)
right_skewed = np.random.exponential(scale=2, size=1000)
left_skewed = -np.random.exponential(scale=2, size=1000) + 10  # flip and shift

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Right Skewed
sns.histplot(right_skewed, bins=30, kde=True, ax=axes[0], color='orange')
axes[0].set_title('Right-Skewed Distribution: Ziyakhala')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

# Left Skewed
sns.histplot(left_skewed, bins=30, kde=True, ax=axes[1], color='purple')
axes[1].set_title('Left-Skewed Distribution: Aykhale')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
