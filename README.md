# ClusteringAnalysis
Datasets Used:
• Simulation Datasets: Square and Elliptical__
• Real-Word Datasets: Cho and Iyer (two gene sequences datasets)__


Tasks for Simulation Datasets:__
• Implement k-means and spectral clustering algorithms to find 2 clusters on Square and Elliptical datasets
and visualize your results__
• Present the performance analysis of the spectral clustering algorithm using different similarity measures like
cosine similarity and Gaussian kernel similarity__
• Present the performance analysis of the spectral clustering algorithm using different Laplacian matrices like
unnormalized Laplacian and normalized symmetric Laplacian__

Tasks for Real-World Datasets:
• Use k-means and spectral clustering algorithms to find clusters of genes on Cho and Iyer datasets which
exhibit similar expression profiles. Note that there are some noise samples in Iyer dataset where a label of
“-1” means outliers, you should consider preprocessing your data.
• Validate your clustering results using the following methods:
– External Index: Use the Accuracy measure to compare the clustering results between k-means and
spectral clustering algorithms on Cho and Iyer datasets (the ground truth clusters are provided in the
data sets).
– Internal Index: Use the Sum of Squared Error (SSE) measure to compare the clustering results between
k-means and spectral clustering algorithms Cho and Iyer datasets.
