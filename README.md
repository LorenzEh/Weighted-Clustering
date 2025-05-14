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
  After fitting the model, you can retrieve a summary table with cluster-wise statistics. This includes the mean value of each feature (in original scale), the number of observations per cluster, and their percentage share — useful for interpreting and comparing clusters.



# Key Concepts

## Coefficient of Variation (CV)

Standardized measure of estimate reliability:

<div align="center">
<img src="https://latex.codecogs.com/svg.image?CV%20%3D%20%5Cleft(%20%5Cfrac%7B%5Ctext%7BMOE%7D%20%2F%201.645%7D%7B%5Ctext%7BEstimate%7D%7D%20%5Cright)%20%5Ctimes%20100" title="CV = \left( \frac{\text{MOE} / 1.645}{\text{Estimate}} \right) \times 100" />
</div>

- **MOE**: Margin of Error (90% confidence level)  
- **Threshold**: CV > 30% considered unreliable (Census Bureau guideline)

## Reliability Weighting Principle

**Weight:**

<div align="center">
<img src="https://latex.codecogs.com/svg.image?w_i%20%5Cpropto%20%5Cfrac%7B1%7D%7BCV_i%7D" title="w_i \propto \frac{1}{CV_i}" />
</div>

Lower CV → Higher weight → Greater cluster influence

# Methodology

## 1. Weight Calculation Pipeline

For each geographic unit:

### 1.1 Mean CV
<div align="center">
<img src="https://latex.codecogs.com/svg.image?\text{MeanCV}_i%20=%20\frac{1}{n_{\text{CV}}}%20\sum_{j=1}^{n_{\text{CV}}}%20\text{CV}_{ij}" title="\text{MeanCV}_i = \frac{1}{n_{\text{CV}}} \sum_{j=1}^{n_{\text{CV}}} \text{CV}_{ij}" />
</div>

### 1.2 Reliability Score
<div align="center">
<img src="https://latex.codecogs.com/svg.image?\text{Reliability}_i%20=%20\frac{1}{\text{MeanCV}_i%20+%20\epsilon}%20\quad%20(\epsilon%20=%2010^{-6})" title="\text{Reliability}_i = \frac{1}{\text{MeanCV}_i + \epsilon} \quad (\epsilon = 10^{-6})" />
</div>

Small epsilon was introduced to avoid the division by zero error. 

### 1.3 Min-Max Normalization
<div align="center">
<img src="https://latex.codecogs.com/svg.image?\text{Weight}_i%20=%20\frac{\text{Reliability}_i%20-%20\min(\text{Rel})}{\max(\text{Rel})%20-%20\min(\text{Rel})}" title="\text{Weight}_i = \frac{\text{Reliability}_i - \min(\text{Rel})}{\max(\text{Rel}) - \min(\text{Rel})}" />
</div>

## 2. Feature Preprocessing


### 2.1 Z-score Normalization
<div align="center">
<img src="https://latex.codecogs.com/svg.image?z%20=%20\frac{x%20-%20\mu}{\sigma}" alt="Z-score" />
</div>

### 2.2 Min-Max Scaling
<div align="center">
<img src="https://latex.codecogs.com/svg.image?x'%20=%20\frac{x%20-%20x_{\min}}{x_{\max}%20-%20x_{\min}}" alt="Min-Max" />
</div>

## 3 Weighting

<div align="center">
<img src="https://latex.codecogs.com/svg.image?\mu_j%20=%20\frac{\sum_{i=1}^N%20w_i%20x_i}{\sum_{i=1}^N%20w_i}" title="\mu_j = \frac{\sum_{i=1}^N w_i x_i}{\sum_{i=1}^N w_i}" />
</div>

Only relative weights matter for the weighting process, and only positive values are meaningful. Therefore, Min-Max scaling is a suitable transformation, as it scales the weights between 0 and 1 — with 1 representing the highest weight (i.e., the lowest coefficient of variation, CV) and 0 representing the lowest weight (i.e., the highest CV).
