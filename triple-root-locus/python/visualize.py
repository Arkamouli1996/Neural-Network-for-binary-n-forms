import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import seaborn as sns

# Load data
X_train = np.loadtxt('data/X_train.csv', delimiter=',')
y_train = np.loadtxt('data/y_train.csv', delimiter=',')
positive = np.loadtxt('data/positive_samples.csv', delimiter=',')
negative = np.loadtxt('data/negative_samples.csv', delimiter=',')

print(f"Loaded {len(X_train)} samples")
print(f"Positive samples: {np.sum(y_train == 1)}")
print(f"Negative samples: {np.sum(y_train == 0)}")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Create figure with subplots
fig = plt.figure(figsize=(18, 12))

# 1. PCA projection to 3D (real parts)
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
pca_real = PCA(n_components=3)
positive_real_pca = pca_real.fit_transform(positive[:, :5])
negative_real_pca = pca_real.transform(negative[:, :5])
ax1.scatter(positive_real_pca[:, 0], positive_real_pca[:, 1], positive_real_pca[:, 2], 
           c='blue', alpha=0.3, s=10, label='On Surface')
ax1.scatter(negative_real_pca[:, 0], negative_real_pca[:, 1], negative_real_pca[:, 2], 
           c='red', alpha=0.3, s=10, label='Random')
ax1.set_title('PCA Projection (Real Parts)', fontsize=12, fontweight='bold')
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_zlabel('PC3')
ax1.legend()

# 2. PCA projection to 2D
ax2 = fig.add_subplot(2, 3, 2)
pca_2d = PCA(n_components=2)
positive_2d = pca_2d.fit_transform(positive)
negative_2d = pca_2d.transform(negative)
ax2.scatter(positive_2d[:, 0], positive_2d[:, 1], c='blue', alpha=0.3, s=10, label='On Surface')
ax2.scatter(negative_2d[:, 0], negative_2d[:, 1], c='red', alpha=0.3, s=10, label='Random')
ax2.set_title('PCA 2D Projection (All Features)', fontsize=12, fontweight='bold')
ax2.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} var)')
ax2.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} var)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Distribution of norms
ax3 = fig.add_subplot(2, 3, 3)
pos_norms = np.linalg.norm(positive[:, :5], axis=1)
neg_norms = np.linalg.norm(negative[:, :5], axis=1)
ax3.hist(pos_norms, bins=50, alpha=0.6, label='On Surface', color='blue', density=True)
ax3.hist(neg_norms, bins=50, alpha=0.6, label='Random', color='red', density=True)
ax3.set_title('Distribution of Norms (Real Parts)', fontsize=12, fontweight='bold')
ax3.set_xlabel('||a||')
ax3.set_ylabel('Density')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Coordinate distributions
ax4 = fig.add_subplot(2, 3, 4)
for i in range(5):
    ax4.hist(positive[:, i], bins=30, alpha=0.3, label=f'a{i} (surface)')
ax4.set_title('Real Coordinate Distributions (On Surface)', fontsize=12, fontweight='bold')
ax4.set_xlabel('Value')
ax4.set_ylabel('Frequency')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# 5. Pairwise scatter: a0 vs a1
ax5 = fig.add_subplot(2, 3, 5)
ax5.scatter(positive[:, 0], positive[:, 1], c='blue', alpha=0.3, s=10, label='On Surface')
ax5.scatter(negative[:, 0], negative[:, 1], c='red', alpha=0.3, s=10, label='Random')
ax5.set_title('a₀ vs a₁ (Real Parts)', fontsize=12, fontweight='bold')
ax5.set_xlabel('a₀')
ax5.set_ylabel('a₁')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Explained variance by PCA
ax6 = fig.add_subplot(2, 3, 6)
pca_full = PCA()
pca_full.fit(positive)
cumsum = np.cumsum(pca_full.explained_variance_ratio_)
ax6.plot(range(1, len(cumsum)+1), cumsum, 'bo-', linewidth=2, markersize=8)
ax6.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
ax6.set_title('Cumulative Explained Variance', fontsize=12, fontweight='bold')
ax6.set_xlabel('Number of Components')
ax6.set_ylabel('Cumulative Variance Explained')
ax6.grid(True, alpha=0.3)
ax6.legend()

plt.suptitle('Triple-Root Surface Dataset Visualization', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('plot/data_visualization.png', dpi=300, bbox_inches='tight')
print("Saved visualization to plot/data_visualization.png")
plt.close()

# Additional plot: t-SNE visualization
print("Computing t-SNE projection...")
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
all_data = np.vstack([positive[:500], negative[:500]])  # Subsample for speed
all_labels = np.hstack([np.ones(500), np.zeros(500)])
tsne_result = tsne.fit_transform(all_data)

fig, ax = plt.subplots(figsize=(10, 8))
scatter1 = ax.scatter(tsne_result[all_labels==1, 0], tsne_result[all_labels==1, 1], 
                     c='blue', alpha=0.6, s=20, label='On Surface', edgecolors='k', linewidth=0.5)
scatter2 = ax.scatter(tsne_result[all_labels==0, 0], tsne_result[all_labels==0, 1], 
                     c='red', alpha=0.6, s=20, label='Random', edgecolors='k', linewidth=0.5)
ax.set_title('t-SNE Visualization (1000 samples)', fontsize=14, fontweight='bold')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot/tsne_visualization.png', dpi=300, bbox_inches='tight')
print("Saved t-SNE visualization to plot/tsne_visualization.png")

print("\nVisualization complete!")
