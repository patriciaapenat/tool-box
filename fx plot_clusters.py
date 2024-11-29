def plot_clusters(data, labels, cluster_centers, title=None, xlabel=None, ylabel=None):
    # Dimensionality reduction with PCA to 2 components for 2D visualization and 3 components for 3D visualization
    pca_2d = PCA(n_components=2)
    data_pca_2d = pca_2d.fit_transform(data)
    
    pca_3d = PCA(n_components=3)
    data_pca_3d = pca_3d.fit_transform(data)
    
    # Project cluster centers onto PCA space
    pca_centers_2d = pca_2d.transform(cluster_centers)
    pca_centers_3d = pca_3d.transform(cluster_centers)

    # Convert data to DataFrame for easier manipulation
    df_2d = pd.DataFrame(data_pca_2d, columns=['PC1', 'PC2'])
    df_2d['Cluster'] = labels.astype(str)
    
    df_3d = pd.DataFrame(data_pca_3d, columns=['PC1', 'PC2', 'PC3'])
    df_3d['Cluster'] = labels.astype(str)
    
    # 2D plot with data points and centroids using Matplotlib
    plt.figure(figsize=(12, 6))  # Set figure size for the 2D plot
    ax1 = plt.subplot(121)  # Use 121 to place the plot on the left
    scatter = ax1.scatter(data_pca_2d[:, 0], data_pca_2d[:, 1], c=labels, cmap='viridis', s=10, alpha=0.5)
    ax1.scatter(pca_centers_2d[:, 0], pca_centers_2d[:, 1], marker='*', c='red', s=200, label='Centroids')
    if title:
        ax1.set_title(title + ' - 2D')
    if xlabel:
        ax1.set_xlabel(xlabel)
    if ylabel:
        ax1.set_ylabel(ylabel)
    ax1.legend()
    legend1 = ax1.legend(*scatter.legend_elements(), title="Clusters")
    ax1.add_artist(legend1)

    # 3D plot with data points and centroids using Matplotlib
    ax2 = plt.subplot(122, projection='3d')
    ax2.scatter(data_pca_3d[:, 0], data_pca_3d[:, 1], data_pca_3d[:, 2], c=labels, cmap='viridis', s=10, alpha=0.5)
    ax2.scatter(pca_centers_3d[:, 0], pca_centers_3d[:, 1], pca_centers_3d[:, 2], marker='*', c='red', s=200, label='Centroids')
    if title:
        ax2.set_title(title + ' - 3D')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_zlabel('PC3')
    ax2.legend()

    plt.tight_layout()
    plt.show()