import pandas as pd
import argparse
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
import umap
import os
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
import seaborn as sns

# Function to extract and preprocess features
def extract_features(df):
    numerical_features = df.select_dtypes(include=['number']).columns
    categorical_features = df.select_dtypes(include=['object']).columns

    # Handle missing data
    numerical_imputer = SimpleImputer(strategy='mean')
    df[numerical_features] = numerical_imputer.fit_transform(df[numerical_features])

    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_features] = categorical_imputer.fit_transform(df[categorical_features])

    # One-hot encoding for categorical variables
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    return df


# Load and preprocess dataset
def load_and_preprocess_data(file_path):
    try:
        print(f"Reading input file from: {file_path}")
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file at {file_path} does not exist.")
        
        # Read dataset
        if file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)

        if df.empty:
            raise ValueError("The dataset is empty.")

        # Process the features
        df = extract_features(df)
        return df

    except Exception as e:
        print(f"Error loading and preprocessing data: {e}")
        return None


# Function to scale the data using different scalers
def scale_data(df, method='standard'):
    try:
        # Choose scaler
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()

        # Fit and transform the data
        df[df.columns] = scaler.fit_transform(df)
        return df, scaler

    except Exception as e:
        print(f"Error during scaling: {e}")
        return None, None


# Function to compare clustering algorithms
def compare_algorithms(df, n_clusters_range=[3, 4, 5, 6], eps=0.5, min_samples=5):
    results = {}

    # KMeans Clustering
    df_kmeans, _, _ = perform_kmeans_clustering(df, n_clusters_range)
    kmeans_silhouette = silhouette_score(df.drop('Cluster', axis=1), df_kmeans['Cluster'])
    kmeans_db_score = davies_bouldin_score(df.drop('Cluster', axis=1), df_kmeans['Cluster'])
    results['KMeans'] = {'Silhouette Score': kmeans_silhouette, 'Davies-Bouldin Score': kmeans_db_score}

    # DBSCAN Clustering
    df_dbscan = perform_dbscan_clustering(df, eps, min_samples)
    dbscan_silhouette = silhouette_score(df.drop('Cluster', axis=1), df_dbscan['Cluster'])
    dbscan_db_score = davies_bouldin_score(df.drop('Cluster', axis=1), df_dbscan['Cluster'])
    results['DBSCAN'] = {'Silhouette Score': dbscan_silhouette, 'Davies-Bouldin Score': dbscan_db_score}

    # Agglomerative Clustering
    df_agg = perform_agglomerative_clustering(df, n_clusters=3)
    agg_silhouette = silhouette_score(df.drop('Cluster', axis=1), df_agg['Cluster'])
    agg_db_score = davies_bouldin_score(df.drop('Cluster', axis=1), df_agg['Cluster'])
    results['Agglomerative'] = {'Silhouette Score': agg_silhouette, 'Davies-Bouldin Score': agg_db_score}

    return results


# KMeans clustering with hyperparameter tuning
def perform_kmeans_clustering(df, n_clusters_range=[3, 4, 5, 6]):
    best_score = -1
    best_n_clusters = n_clusters_range[0]
    silhouette_scores = []

    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=42)
        df['Cluster'] = kmeans.fit_predict(df)

        sil_score = silhouette_score(df.drop('Cluster', axis=1), df['Cluster'])
        silhouette_scores.append(sil_score)
        print(f"KMeans with {n_clusters} clusters - Silhouette Score: {sil_score:.4f}")

        if sil_score > best_score:
            best_score = sil_score
            best_n_clusters = n_clusters

    print(f"\nBest KMeans result: {best_n_clusters} clusters with Silhouette Score {best_score:.4f}")
    return df, best_n_clusters, silhouette_scores


# DBSCAN clustering
def perform_dbscan_clustering(df, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['Cluster'] = dbscan.fit_predict(df)
    print(f"DBSCAN clustering done with eps={eps}, min_samples={min_samples}")

    valid_points = df[df['Cluster'] != -1]
    if len(valid_points['Cluster'].unique()) > 1:
        sil_score = silhouette_score(valid_points.drop('Cluster', axis=1), valid_points['Cluster'])
        db_score = davies_bouldin_score(valid_points.drop('Cluster', axis=1), valid_points['Cluster'])
        print(f"Silhouette Score: {sil_score:.4f}")
        print(f"Davies-Bouldin Score: {db_score:.4f}")
    else:
        print("Silhouette and DB scores cannot be calculated due to too few valid clusters.")

    return df


# Agglomerative clustering
def perform_agglomerative_clustering(df, n_clusters=3):
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    df['Cluster'] = agg.fit_predict(df)
    print(f"Agglomerative clustering done with {n_clusters} clusters")

    sil_score = silhouette_score(df.drop('Cluster', axis=1), df['Cluster'])
    db_score = davies_bouldin_score(df.drop('Cluster', axis=1), df['Cluster'])
    print(f"Silhouette Score: {sil_score:.4f}")
    print(f"Davies-Bouldin Score: {db_score:.4f}")

    return df


# Dimensionality reductions
def perform_pca_reduction(df, n_components=2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(df.drop('Cluster', axis=1))


def perform_tsne_reduction(df, n_components=2):
    tsne = TSNE(n_components=n_components, random_state=42)
    return tsne.fit_transform(df.drop('Cluster', axis=1))


def perform_umap_reduction(df, n_components=2):
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    return reducer.fit_transform(df.drop('Cluster', axis=1))


# Visualization of Clusters
def visualize_clusters_and_dimensions(df, pca_components=None, tsne_components=None, umap_components=None, output_image_path=None):
    if output_image_path:
        output_dir = "../data_analysis"
        os.makedirs(output_dir, exist_ok=True)

    def save_or_show(components, title, file_suffix):
        plt.figure(figsize=(8, 6))
        plt.scatter(components[:, 0], components[:, 1], c=df['Cluster'], cmap='viridis')
        plt.title(title)
        plt.xlabel(f"{title} Component 1")
        plt.ylabel(f"{title} Component 2")
        plt.colorbar(label='Cluster')
        if output_image_path:
            path = f"{output_dir}/{output_image_path}_{file_suffix}.png"
            plt.savefig(path)
            print(f"Plot saved to {path}")
        plt.show()

    if pca_components is not None:
        save_or_show(pca_components, "PCA", "pca")

    if tsne_components is not None:
        save_or_show(tsne_components, "t-SNE", "tsne")

    if umap_components is not None:
        save_or_show(umap_components, "UMAP", "umap")


# Performance Comparison Visualization
def plot_comparison(results):
    # Convert results dictionary to DataFrame for better visualization
    df_results = pd.DataFrame(results).T
    df_results = df_results[['Silhouette Score', 'Davies-Bouldin Score']]
    
    # Plot the comparison of performance metrics
    df_results.plot(kind='bar', figsize=(10, 6), colormap='viridis', title='Clustering Algorithms Comparison')
    plt.ylabel('Score')
    plt.xlabel('Algorithm')
    plt.tight_layout()
    plt.show()


# Main execution
def main(file_path, clustering_method='kmeans', n_clusters=3, eps=0.5, min_samples=5, output_image="cluster_plot", scale_method='standard'):
    # Load and preprocess data
    df = load_and_preprocess_data(file_path)
    
    if df is None:
        return
    
    # Scale the data using selected method
    df, scaler = scale_data(df, method=scale_method)
    
    if df is not None:
        # Compare algorithms
        comparison_results = compare_algorithms(df, n_clusters_range=[n_clusters], eps=eps, min_samples=min_samples)
        
        # Visualize clustering results
        pca_components = perform_pca_reduction(df, 2)
        tsne_components = perform_tsne_reduction(df, 2)
        umap_components = perform_umap_reduction(df, 2)

        visualize_clusters_and_dimensions(df, pca_components, tsne_components, umap_components, output_image_path=output_image)
        
        # Plot comparison metrics
        plot_comparison(comparison_results)

        df.to_csv(f"../data_analysis/clustered_and_reduced_data.csv", index=False)
        print(f"Clustered data saved to '../data_analysis/clustered_and_reduced_data.csv'")


# Argument Parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unsupervised learning pipeline with clustering and dimensionality reduction")
    parser.add_argument("file_path", type=str, help="Path to CSV or Excel file")
    parser.add_argument("--n_clusters", type=int, default=3, help="Number of clusters")
    parser.add_argument("--n_components", type=int, default=2, help="Number of components for PCA/t-SNE/UMAP")
    parser.add_argument("--output_image", type=str, default="cluster_plot", help="Filename suffix for saving plots")
    parser.add_argument("--clustering_method", type=str, choices=["kmeans", "dbscan", "agglomerative"], default="kmeans", help="Clustering algorithm")
    parser.add_argument("--eps", type=float, default=0.5, help="Epsilon for DBSCAN")
    parser.add_argument("--min_samples", type=int, default=5, help="Minimum samples for DBSCAN")
    parser.add_argument("--scale_method", type=str, choices=['standard', 'minmax', 'robust'], default='standard', help="Scaling method")

    args = parser.parse_args()

    main(args.file_path, args.clustering_method, args.n_clusters, args.eps, args.min_samples, args.output_image, args.scale_method)
