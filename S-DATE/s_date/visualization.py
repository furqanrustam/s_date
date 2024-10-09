# s_data/visualization.py
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_data(original_data, synthetic_data):
    """Visualize means, standard deviations, PCA, and correlation matrices of original and synthetic data."""
    
    # Calculate correlations
    original_corr = original_data.corr()
    synthetic_corr = synthetic_data.corr()

    # Step 1: Compare Means and Standard Deviations
    original_mean = original_data.mean()
    synthetic_mean = synthetic_data.mean()
    original_std = original_data.std()
    synthetic_std = synthetic_data.std()

    # Plot means
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(original_mean, label='Original Mean', marker='o')
    plt.plot(synthetic_mean, label='Synthetic Mean', marker='o')
    plt.title('Means of Original and Synthetic Data')
    plt.xlabel('Features')
    plt.ylabel('Mean Value')
    plt.legend()

    # Plot stds
    plt.subplot(1, 2, 2)
    plt.plot(original_std, label='Original Std', marker='o')
    plt.plot(synthetic_std, label='Synthetic Std', marker='o')
    plt.title('Standard Deviations of Original and Synthetic Data')
    plt.xlabel('Features')
    plt.ylabel('Standard Deviation')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Step 2: PCA Visualization
    pca = PCA(n_components=2)
    original_pca = pca.fit_transform(original_data)
    synthetic_pca = pca.transform(synthetic_data)

    plt.figure(figsize=(10, 7))
    plt.scatter(original_pca[:, 0], original_pca[:, 1], label='Original Data', alpha=0.5)
    plt.scatter(synthetic_pca[:, 0], synthetic_pca[:, 1], label='Synthetic Data', alpha=0.5)
    plt.title('PCA of Original and Synthetic Data')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.show()

    # Step 3: Heatmaps of Correlation Matrices
    plt.figure(figsize=(20, 6))
    
    plt.subplot(1, 3, 1)
    sns.heatmap(original_corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix of Original Data')

    plt.subplot(1, 3, 2)
    sns.heatmap(synthetic_corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix of Synthetic Data')

    plt.subplot(1, 3, 3)
    difference_corr = original_corr - synthetic_corr
    sns.heatmap(difference_corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Difference in Correlation Matrices')

    plt.tight_layout()
    plt.show()
