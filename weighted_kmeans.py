# libraries
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.lines import Line2D
import contextily as ctx
import matplotlib.pyplot as plt
import pandas as pd
# plt.rcParams['figure.figsize'] = [15, 15]


class EnhancedKMeans:
    """
    Enhanced K-means clustering with automatic CV-based weighting, feature scaling, and cluster statistics
    
    Parameters:
    - n_clusters: Number of clusters (default=8)
    - weighted: Use reliability weights (True) or standard K-means (False)
    - scaler_type: None, 'standard' (z-score), or 'minmax' (default='standard')
    - random_state: Random seed (default=None)
    - max_iter: Maximum iterations (default=300)

    Additional features: 
    - WCSS tracking for elbow method and cluster statistics
    - Combined evaluation plots (Silhouette and WCSS)
    - Automatic scaling inversion for cluster statistics
    """
    
    def __init__(self, n_clusters=8, weighted=True, scaler_type='standard', 
                 random_state=None, max_iter=300):
        self.n_clusters = n_clusters
        self.weighted = weighted
        self.scaler_type = scaler_type
        self.random_state = random_state
        self.max_iter = max_iter
        self.kmeans = KMeans(n_clusters=n_clusters,
                            random_state=random_state,
                            max_iter=max_iter)
        self.weights_ = None
        self.labels_ = None
        self.cluster_centers_ = None
        self.scaler = None
        self.X_scaled_ = None
        self.wcss_ = None

    def calculate_weights(self, data, variables):
        """Calculate reliability weights using CV columns"""
        cv_columns = [f"CV {var}" for var in variables if f"CV {var}" in data.columns]
        
        if not cv_columns:
            return np.ones(len(data))
            
        mean_cv = data[cv_columns].mean(axis=1)
        reliability = 1 / (mean_cv + 1e-6)
        return MinMaxScaler().fit_transform(reliability.values.reshape(-1, 1)).flatten()

    def _scale_features(self, X):
        """Internal method for feature scaling"""
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaler_type is None:
            return X.copy()
        
        return self.scaler.fit_transform(X)

    def fit(self, data, variables):
        """Fit the model to data with optional scaling"""
        X = data[variables].values
        self.X_scaled_ = self._scale_features(X)
        
        if self.weighted:
            self.weights_ = self.calculate_weights(data, variables)
        else:
            self.weights_ = None
            
        self.kmeans.fit(self.X_scaled_, sample_weight=self.weights_)
        self.labels_ = self.kmeans.labels_
        self.cluster_centers_ = self.kmeans.cluster_centers_
        self.wcss_ = self.kmeans.inertia_  
        return self

    def fit_predict(self, data, variables, add_to_data=False):
        """Fit model and return cluster labels"""
        self.fit(data, variables)
        if add_to_data:
            col_name = f"Cluster{' (weighted)' if self.weighted else ''}"
            data[col_name] = self.labels_
        return self.labels_

    def elbow_analysis(self, data, variables, cluster_range=range(2, 11)):
        """Perform elbow method analysis using WCSS"""
        wcss_values = []
        
        for k in cluster_range:
            temp_model = EnhancedKMeans(
                n_clusters=k,
                weighted=self.weighted,
                scaler_type=self.scaler_type,
                random_state=self.random_state
            )
            temp_model.fit(data, variables)
            wcss_values.append(temp_model.wcss_)
            
        plt.figure(figsize=(10, 6))
        plt.plot(cluster_range, wcss_values, 'bo-', 
                markersize=8, linewidth=2, color='darkblue')
        plt.xlabel('Number of Clusters', fontsize=12)
        plt.ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12)
        plt.title('Elbow Method for Optimal Cluster Number', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(cluster_range)
        plt.show()
        
        return wcss_values

    def cluster_evaluation(self, data, variables, cluster_range=range(2, 11)):
        """Combined evaluation with WCSS and Silhouette Score"""
        wcss_values = []
        silhouette_scores = []
        
        for k in cluster_range:
            temp_model = EnhancedKMeans(
                n_clusters=k,
                weighted=self.weighted,
                scaler_type=self.scaler_type,
                random_state=self.random_state
            )
            temp_model.fit(data, variables)
            
            wcss_values.append(temp_model.wcss_)
            silhouette_scores.append(
                silhouette_score(temp_model.X_scaled_, temp_model.labels_)
            )
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        # Elbow plot
        ax1.plot(cluster_range, wcss_values, 'bo-', 
                markersize=8, linewidth=2, color='darkblue')
        ax1.set_title('Elbow Method (WCSS)', fontsize=14)
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Within-Cluster Sum of Squares')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_xticks(cluster_range)
        
        # Silhouette plot
        ax2.plot(cluster_range, silhouette_scores, 'bo-', 
                markersize=8, linewidth=2, color='darkred')
        ax2.set_title('Silhouette Analysis', fontsize=14)
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_xticks(cluster_range)
        
        plt.tight_layout()
        plt.show()
        
        return {'wcss': wcss_values, 'silhouette': silhouette_scores}

    def plot_clusters(self, data, cluster_column='Cluster', figsize=(60, 30), zoom=5, 
                    basemap_source=ctx.providers.OpenStreetMap.Mapnik):
        """Visualize clusters on a map"""
        if 'geometry' not in data.columns:
            raise ValueError("Data must be a GeoDataFrame with geometry column")
            
        data = data.to_crs(epsg=3857)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        unique_clusters = data[cluster_column].nunique()
        cmap = plt.get_cmap('tab20').resampled(unique_clusters)
        
        data.plot(
            ax=ax,
            column=cluster_column,
            cmap=cmap,
            edgecolor='lightgray',
            alpha=0.4,
            categorical=True,
            legend=False
        )
        
        try:
            ctx.add_basemap(ax, crs=data.crs, zoom=zoom, source=basemap_source)
        except Exception as e:
            print(f"Basemap loading failed: {e}")

        legend_elements = [
            Line2D([0], [0],
                   marker='o',
                   color='w',
                   markerfacecolor=cmap(i/unique_clusters),
                   markersize=30,
                   label=f'Cluster {i+1}')
            for i in range(unique_clusters)
        ]
        
        ax.legend(
            handles=legend_elements,
            title="Clusters",
            loc="lower right",
            fontsize=32,
            frameon=True,
            framealpha=0.9
        )
        
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()
        
    
    def get_cluster_stats(self, variables):
        """
        Returns DataFrame with cluster statistics:
        - Mean feature values (in original scale if scaled)
        - Number of observations
        - Observation percentage
        - WCSS (Within-Cluster Sum of Squares)
        - WCSS % (Contribution to total WCSS)
        
        Args:
            variables: List of variable names used in clustering
            
        Returns:
            DataFrame with cluster statistics
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        # Get centers in original feature space
        if self.scaler is not None:
            centers = self.scaler.inverse_transform(self.cluster_centers_)
        else:
            centers = self.cluster_centers_
            
        # Create DataFrame with cluster means
        stats_df = pd.DataFrame(centers, columns=variables)
        stats_df['Cluster'] = stats_df.index + 1  # Cluster numbering from 1
        
        # Add observation counts
        unique, counts = np.unique(self.labels_, return_counts=True)
        stats_df['Observations'] = stats_df.index.map(lambda x: counts[x])
        stats_df['Observation %'] = (stats_df['Observations'] / stats_df['Observations'].sum()) * 100
        
        # Calculate WCSS per cluster with weights
        cluster_wcss = []
        for i in range(self.n_clusters):
            cluster_mask = (self.labels_ == i)
            cluster_points = self.X_scaled_[cluster_mask]
            cluster_weights = self.weights_[cluster_mask] if self.weights_ is not None else None
            
            # Weighted squared distances
            squared_dist = np.sum((cluster_points - self.cluster_centers_[i])**2, axis=1)
            if cluster_weights is not None:
                squared_dist *= cluster_weights  # Apply weights
            
            cluster_wcss.append(np.sum(squared_dist))
        
        # Verify total WCSS matches self.wcss_
        total_calculated_wcss = np.sum(cluster_wcss)
        if not np.isclose(total_calculated_wcss, self.wcss_, rtol=1e-3):
            raise ValueError(f"WCSS mismatch: {total_calculated_wcss} vs {self.wcss_}")
        
        stats_df['WCSS'] = cluster_wcss
        stats_df['WCSS %'] = (stats_df['WCSS'] / self.wcss_) * 100  # Now correct
        
        # Reorder columns and sort
        stats_df = stats_df[['Cluster', 'Observations', 'Observation %', 'WCSS', 'WCSS %'] + variables]
        return stats_df.sort_values('Cluster').reset_index(drop=True)
        
