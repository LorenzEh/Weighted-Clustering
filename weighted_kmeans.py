
class EnhancedKMeans:
    """
    Enhanced K-means clustering with automatic CV-based weighting, feature scaling, and cluster statistics
    
    Parameters:
    - n_clusters: Number of clusters (default=8)
    - weighted: Use reliability weights (True) or standard K-means (False)
    - scaler_type: None, 'standard' (z-score), or 'minmax' (default='standard')
    - random_state: Random seed (default=None)
    - max_iter: Maximum iterations (default=300)
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
        return self

    def fit_predict(self, data, variables, add_to_data=False):
        """Fit model and return cluster labels"""
        self.fit(data, variables)
        if add_to_data:
            col_name = f"Cluster{' (weighted)' if self.weighted else ''}"
            data[col_name] = self.labels_
        return self.labels_

    def silhouette_analysis(self, data, variables, cluster_range=range(2, 15)):
        """Perform silhouette analysis with proper scaling"""
        scores = []
        
        for k in cluster_range:
            temp_model = EnhancedKMeans(
                n_clusters=k,
                weighted=self.weighted,
                scaler_type=self.scaler_type,
                random_state=self.random_state
            )
            temp_model.fit(data, variables)
            scores.append(silhouette_score(temp_model.X_scaled_, temp_model.labels_))
            
        plt.figure(figsize=(10, 6))
        plt.plot(cluster_range, scores, marker='o', linestyle='--', color='r')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        title_type = ['Unscaled', 'Z-score Scaled', 'MinMax Scaled'][
            ['none', 'standard', 'minmax'].index(self.scaler_type or 'none')]
        plt.title(f'Silhouette Analysis ({title_type} Features)')
        plt.grid(True)
        plt.show()
        
        return scores

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
        
        # Reorder columns and sort
        stats_df = stats_df[['Cluster', 'Observations', 'Observation %'] + variables]
        return stats_df.sort_values('Cluster').reset_index(drop=True)
