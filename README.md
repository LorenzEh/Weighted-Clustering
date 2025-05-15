# Weighted-Clustering with Data Quality Integration ![Python](https://img.shields.io/badge/Python-3.8-blue.svg)

This project implements a reliability-weighted K-means clustering algorithm that incorporates data quality metrics through Coefficient of Variation (CV) values. Designed for use with American Community Survey (ACS-5) data, this approach gives higher weight to observations with lower measurement error. It's supposed to be used with GeoDataFrames. The following features are included: 

## Features

- **Reliability-Weighted Clustering**  
  This algorithm allows you to incorporate data quality directly into the clustering process. Observations with lower Coefficient of Variation (CV) values — indicating more reliable measurements — receive higher weights. This ensures that clusters are driven more by high-quality data rather than noisy inputs.

- **Customizable Feature Scaling**  
  Choose from `standard` (z-score), `minmax`, or no scaling to preprocess your data before clustering. Feature scaling is critical when variables are on different scales, and this option provides flexibility depending on your dataset's characteristics.

- **Silhouette Analysis & WCSS for Optimal Cluster Selection**  
  The silhouette analysis method is integrated to help you identify the most appropriate number of clusters. This feature generates silhouette scores and WCSS for a user-defined range of cluster counts and visualizes them, aiding in model selection.

- **Map-Based Cluster Visualization**  
  Designed for use with spatial data, the class can generate map plots using `GeoDataFrames`. It overlays clusters on a basemap (via `contextily`), making it easier to interpret spatial patterns in the results.

- **Descriptive Cluster Statistics**  
  After fitting the model, you can retrieve a summary table with cluster-wise statistics. This includes the mean value of each feature (in original scale), the WCSS per Cluster (percentage of total and absolute value), the number of observations per cluster, and their percentage share — useful for interpreting and comparing clusters.
  

## Key Concepts

### Coefficient of Variation (CV)

A standardized measure of estimate reliability:

<div align="center">
<img src="https://latex.codecogs.com/svg.image?CV%20=%20\left(%20\frac{\text{MOE}/1.645}{\text{Estimate}}%20\right)%20\times%20100" title="CV = \left( \frac{MOE/1.645}{Estimate} \right) \times 100" />
</div>



- **MOE**: Margin of Error (at 90% confidence level)  
- **Threshold**: CV > 30% considered unreliable (per convention)

---

## ⚖️ Reliability Weighting Principle

### Weight Calculation Pipeline

We assign higher weights to more reliable estimates (lower CV):

<div align="center">
<img src="https://latex.codecogs.com/svg.image?\text{Reliability}_i%20=%20\frac{1}{\text{MeanCV}_i%20+%20\epsilon}%20\quad%20(\epsilon%20=%2010^{-6})" title="\text{Reliability}_i = \frac{1}{\text{MeanCV}_i + \epsilon} \quad (\epsilon = 10^{-6})" />
</div>


Where:
- `MeanCVᵢ`: Mean coefficient of variation for observation *i*

<div align="center">
<img src="https://latex.codecogs.com/svg.image?\text{MeanCV}_i%20=%20\frac{1}{m}%20\sum_{j=1}^{m}%20\text{CV}_{ij}" title="\text{MeanCV}_i = \frac{1}{m} \sum_{j=1}^{m} \text{CV}_{ij}" />
</div>

- `ε`: Small constant to avoid division by zero 

### Min-Max Normalization

Normalize weights to [0, 1]:

<div align="center">
<img src="https://latex.codecogs.com/svg.image?\text{Weight}_i%20=%20\frac{\text{Reliability}_i%20-%20\min(\text{Rel})}{\max(\text{Rel})%20-%20\min(\text{Rel})}" title="\text{Weight}_i = \frac{\text{Reliability}_i - \min(\text{Rel})}{\max(\text{Rel}) - \min(\text{Rel})}" />
</div>


- 1 = highest reliability (lowest CV)  
- 0 = lowest reliability (highest CV)

---

## 📊 Within-Cluster Sum of Squares (WCSS)

Measures how tightly grouped the points in a cluster are.

### Cluster-Level WCSS Calculation

For each cluster *k*, compute the weighted squared distance to its centroid:

<div align="center">
<img src="https://latex.codecogs.com/svg.image?\text{WCSS}_k%20=%20\sum_{i%20\in%20C_k}%20w_i%20\cdot%20\|x_i%20-%20\mu_k\|^2" title="\text{WCSS}_k = \sum_{i \in C_k} w_i \cdot \|x_i - \mu_k\|^2" />
</div>


Where:
- `Cₖ`: Observations in cluster *k*  
- `wᵢ`: Reliability weight for observation *i*  
- `xᵢ`: Scaled feature vector of observation *i*  
- `μₖ`: Centroid of cluster *k*

### Total WCSS

<div align="center">
<img src="https://latex.codecogs.com/svg.image?\text{Total%20WCSS}%20=%20\sum_{k=1}^{K}%20\text{WCSS}_k" title="\text{Total WCSS} = \sum_{k=1}^{K} \text{WCSS}_k" />
</div>


### Cluster WCSS Contribution (%)

<div align="center">
<img src="https://latex.codecogs.com/svg.image?\text{WCSS\%}_k%20=%20\left(%20\frac{\text{WCSS}_k}{\text{Total%20WCSS}}%20\right)%20\times%20100" title="\text{WCSS\%}_k = \left( \frac{\text{WCSS}_k}{\text{Total WCSS}} \right) \times 100" />
</div>

- High **WCSS%**: Cluster has higher internal variability  
- Low **WCSS%**: Cluster is more homogeneous

---

## ⚙️ Feature Preprocessing

### Z-score Normalization

<div align="center">
<img src="https://latex.codecogs.com/svg.image?z%20=%20\frac{x%20-%20\mu}{\sigma}" title="z = \frac{x - \mu}{\sigma}" />
</div>


- Standardizes to mean 0, standard deviation 1

### Min-Max Scaling

<div align="center">
<img src="https://latex.codecogs.com/svg.image?x'%20=%20\frac{x%20-%20x_{\min}}{x_{\max}%20-%20x_{\min}}" title="x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}" />
</div>


- Scales features to [0, 1]

---
