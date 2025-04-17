import pandas as pd
import argparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt  # For visualization

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

# Perform PCA for dimensionality reduction
def perform_pca(df, n_components=2):
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(df)
    print(f"PCA performed, reducing dimensions to {n_components} components.")
    return pca_components

# Visualize the PCA components and the clusters
def visualize_pca_and_clusters(pca_components, df, output_image_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_components[:, 0], pca_components[:, 1], c=df['Cluster'], cmap='viridis', label='Data Points')
    plt.colorbar(label='Cluster')
    plt.title('PCA of Data with Clustering')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    # Save the plot image
    plt.savefig(output_image_path)
    print(f"Plot saved to {output_image_path}")
    plt.show()

if __name__ == "__main__":
    # Define the argument parser
    parser = argparse.ArgumentParser(description="Perform unsupervised learning (Clustering or PCA).")
    parser.add_argument("file_path", type=str, help="Path to the input file (CSV or Excel) containing data.")
    parser.add_argument("--n_clusters", type=int, default=3, help="Number of clusters for KMeans.")
    parser.add_argument("--n_components", type=int, default=2, help="Number of components for PCA.")
    parser.add_argument("--output_image", type=str, default="../data_analysis/pca_clusters_plot.png", help="Path to save the plot image.")
    
    args = parser.parse_args()

    df, scaler = load_and_preprocess_data(args.file_path)
    
    if df is not None:
        # Perform KMeans clustering
        df = perform_kmeans_clustering(df, n_clusters=args.n_clusters)
        
        # Perform PCA
        pca_components = perform_pca(df, n_components=args.n_components)
        
        # Visualize PCA and clusters
        visualize_pca_and_clusters(pca_components, df, args.output_image)
        
        # Save results
        df.to_csv('../data_analysis/clustered_and_reduced_data.csv', index=False)
        print(f"Clustered and reduced data saved to 'data_analysis/clustered_and_reduced_data.csv'.")
