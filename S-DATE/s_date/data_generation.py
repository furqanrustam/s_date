# s_data/data_generation.py
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from .visualization import visualize_data

def dimRed(features, method='PCA'):
    # Perform dimensionality reduction
    if method == 'PCA':
        reducer = PCA(n_components=2)
    elif method == 'SVD':
        reducer = TruncatedSVD(n_components=2)
    else:
        raise ValueError("Invalid method specified. Use 'PCA' or 'SVD'.")

    return reducer.fit_transform(features)

def generate_synthetic_data(features, num_samples=100, dim_reduction_method='PCA', distance_metric='euclidean', 
                            normalization=True, visualize=False, closest_samples=10):
    """Generate synthetic samples based on the original features."""
    
    # Normalize features if specified
    if normalization:
        scaler = MinMaxScaler()  # Can also use StandardScaler() for standardization
        features_normalized = scaler.fit_transform(features)
    else:
        features_normalized = features

    # Dimensionality reduction
    reduced_features = dimRed(features_normalized, method=dim_reduction_method)

    # Prepare for synthetic data generation
    generated_samples = []
    all_samples = features.copy()  # Keep a copy of original features for concatenation

    # Valid distance metrics
    valid_metrics = ['euclidean', 'manhattan', 'cosine']
    if distance_metric not in valid_metrics:
        raise ValueError(f"Invalid distance metric specified. Choose from {valid_metrics}.")

    while len(generated_samples) < num_samples:
        used_indices = set()
        current_total_samples = len(all_samples)

        for l in range(current_total_samples):
            if len(generated_samples) >= num_samples:
                break  # Stop if we have generated the required number of samples
            
            #print(f"Processing sample {l + 1}...")

            # Calculate pairwise distances for the current point using the specified distance metric
            distances = pairwise_distances(reduced_features[l:l+1, :], reduced_features, metric=distance_metric).flatten()
            
            # Identify indices of the closest samples (excluding itself)
            closest_indices = distances.argsort()[1:closest_samples + 1]  # Get closest_samples closest samples

            # Generate a synthetic sample as the mean of the closest samples
            synthetic_sample = np.mean(all_samples.iloc[closest_indices], axis=0)  # Use original features
            
            generated_samples.append(synthetic_sample)

            # Mark the current sample and its closest indices as used
            used_indices.add(l)
            used_indices.update(closest_indices)

        # Combine original and generated samples for the next iteration
        generated_df = pd.DataFrame(generated_samples, columns=features.columns)  # Retain original feature names
        all_samples = pd.concat([all_samples, generated_df], ignore_index=True)  # Update all_samples with new data

        # Normalize again if needed
        if normalization:
            features_normalized = scaler.fit_transform(all_samples)

        # Update reduced features for the new combined dataset
        reduced_features = dimRed(features_normalized, method=dim_reduction_method)

    # Create a DataFrame for the generated samples
    final_generated_df = pd.DataFrame(generated_samples, columns=features.columns)

    # Visualize if requested
    if visualize:
        visualize_data(features, final_generated_df)

    return final_generated_df
