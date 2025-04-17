import pandas as pd
import argparse
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt  # For visualization
from sklearn.manifold import TSNE
import umap
import os

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Reading CSV input file...")
        
        # Preprocess data
        numerical_features = df.select_dtypes(include=['number']).columns
        categorical_features = df.select_dtypes(include=['object']).columns
        
        numerical_imputer = SimpleImputer(strategy='mean')
        df[numerical_features] = numerical_imputer.fit_transform(df[numerical_features])
        
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_features] = categorical_imputer.fit_transform(df[categorical_features])
        
        df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
        
        # Scale the features
        scaler = StandardScaler()
        df[numerical_features] = scaler.fit_transform(df[numerical_features])

        return df, scaler
    except Exception as e:
        print(f"Error during data loading and preprocessing: {e}")
        return None, None

# Perform KMeans clustering
def perform_kmeans_clustering(df, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df)
    print(f"KMeans clustering performed with {n_clusters} clusters.")
    return df

# Perform DBSCAN clustering
def perform_dbscan_clustering(df, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['Cluster'] = dbscan.fit_predict(df)
    print(f"DBSCAN clustering performed with eps={eps}, min_samples={min_samples}.")
    return df

# Perform Agglomerative clustering
def perform_agglomerative_clustering(df, n_clusters=3):
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    df['Cluster'] = agglomerative.fit_predict(df)
    print(f"Agglomerative clustering performed with {n_clusters} clusters.")
    return df

# Perform PCA for dimensionality reduction
def perform_pca(df, n_components=2):
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(df)
    print(f"PCA performed, reducing dimensions to {n_components} components.")
    return pca_components

# Perform t-SNE dimensionality reduction
def perform_tsne(df, n_components=2):
    tsne = TSNE(n_components=n_components)
    tsne_components = tsne.fit_transform(df)
    print(f"t-SNE performed, reducing dimensions to {n_components} components.")
    return tsne_components

# Perform UMAP dimensionality reduction
def perform_umap(df, n_components=2):
    reducer = umap.UMAP(n_components=n_components)
    umap_components = reducer.fit_transform(df)
    print(f"UMAP performed, reducing dimensions to {n_components} components.")
    return umap_components

# Visualize the PCA, t-SNE, or UMAP components and the clusters
def visualize_clusters_and_dimensions(df, pca_components=None, tsne_components=None, umap_components=None, output_image_path=None):
    plt.figure(figsize=(8, 6))
    
    if pca_components is not None:
        plt.scatter(pca_components[:, 0], pca_components[:, 1], c=df['Cluster'], cmap='viridis', label='Data Points (PCA)')
        plt.colorbar(label='Cluster')
        plt.title('PCA of Data with Clustering')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
    elif tsne_components is not None:
        plt.scatter(tsne_components[:, 0], tsne_components[:, 1], c=df['Cluster'], cmap='viridis', label='Data Points (t-SNE)')
        plt.colorbar(label='Cluster')
        plt.title('t-SNE of Data with Clustering')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
    elif umap_components is not None:
        plt.scatter(umap_components[:, 0], umap_components[:, 1], c=df['Cluster'], cmap='viridis', label='Data Points (UMAP)')
        plt.colorbar(label='Cluster')
        plt.title('UMAP of Data with Clustering')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')

    # Save the plot image
    if output_image_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        
        # Save the plot as PNG
        plt.savefig(output_image_path)
        print(f"Plot saved to {output_image_path}")
    
    plt.show()


if __name__ == "__main__":
    # Define the argument parser
    parser = argparse.ArgumentParser(description="Perform unsupervised learning (Clustering or PCA).")
    parser.add_argument("file_path", type=str, help="Path to the input file (CSV or Excel) containing data.")
    parser.add_argument("--n_clusters", type=int, default=3, help="Number of clusters for KMeans and Agglomerative.")
    parser.add_argument("--n_components", type=int, default=2, help="Number of components for PCA, t-SNE, or UMAP.")
    parser.add_argument("--output_image", type=str, default="pca_clusters_plot.png", help="Path to save the plot image.")
    parser.add_argument("--clustering_method", type=str, choices=['kmeans', 'dbscan', 'agglomerative'], default='kmeans', help="Clustering method to use.")
    parser.add_argument("--eps", type=float, default=0.5, help="Epsilon for DBSCAN.")
    parser.add_argument("--min_samples", type=int, default=5, help="Minimum samples for DBSCAN.")
    
    args = parser.parse_args()

    # Load and preprocess data
    df, scaler = load_and_preprocess_data(args.file_path)
    
    if df is not None:
        # Perform clustering based on selected method
        if args.clustering_method == 'kmeans':
            df = perform_kmeans_clustering(df, n_clusters=args.n_clusters)
        elif args.clustering_method == 'dbscan':
            df = perform_dbscan_clustering(df, eps=args.eps, min_samples=args.min_samples)
        elif args.clustering_method == 'agglomerative':
            df = perform_agglomerative_clustering(df, n_clusters=args.n_clusters)

        # Perform dimensionality reduction (PCA, t-SNE, or UMAP)
        pca_components = perform_pca(df, n_components=args.n_components)
        tsne_components = perform_tsne(df, n_components=args.n_components)
        umap_components = perform_umap(df, n_components=args.n_components)
        
        # Visualize the clustering with the dimensionality reduction results
        visualize_clusters_and_dimensions(df, pca_components=pca_components, tsne_components=tsne_components, umap_components=umap_components, output_image_path=args.output_image)
        
        # Save the clustered data to CSV
        df.to_csv('../data_analysis/clustered_and_reduced_data_02.csv', index=False)
        print(f"Clustered and reduced data saved to 'clustered_and_reduced_data_02.csv'.")
