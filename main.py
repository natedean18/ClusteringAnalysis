#Nate Dean

#import statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import spectral_embedding


# Simulation Datasets - Part 1
#read in datasets
square = pd.read_csv('Datasets/square.txt', delimiter=' ', header=None)
elliptical = pd.read_csv('Datasets/elliptical.txt', delimiter=' ', header=None)

#separate x and y columns
square_x_values = pd.to_numeric(square[0].tolist())
square_y_values = pd.to_numeric(square[1].tolist())
elliptical_x_values = pd.to_numeric(elliptical[0].tolist())
elliptical_y_values = pd.to_numeric(elliptical[1].tolist())

#make numpy arrays using x and y columns
square = np.array(list(zip(square_x_values, square_y_values)))
elliptical = np.array(list(zip(elliptical_x_values, elliptical_y_values)))

#K Means Clustering Algorithm
def k_means(data, string):
    km = KMeans(n_clusters=2)
    km.fit(data)
    lb = km.labels_
    cols = ['green', 'blue']
    for i in range(2):
        plt.scatter(data[lb == i, 0], data[lb == i, 1], c=cols[i])
    plt.title(string + ': K-MEANS Clustering Algorithm Visualization')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

#Spectral Clustering Algorithm
def spectral(data, string):
    spec = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='kmeans')
    lb = spec.fit_predict(data)
    cols = ['green', 'blue']
    for i in range(2):
        plt.scatter(data[lb == i, 0], data[lb == i, 1], c=cols[i])
    plt.title(string + ': Spectral Clustering Algorithm Visualization')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

#Run K-means and spectral clustering algorithms
k_means(square, "Square")
k_means(elliptical, "Elliptical")
spectral(square, "Square")
spectral(elliptical, "Elliptical")


#Part 1 - Performance Analysis
#Gaussian Kernel Similarity
#calculate gaussian kernel affinity matrices
af = rbf_kernel(square, gamma=len(square[0]))
af1 = rbf_kernel(elliptical, gamma=len(elliptical[0]))

#Spectral Clustering Algorithm using Gaussian Kernel
def spectral_gk(data, gaussian, string):
    spec = SpectralClustering(n_clusters=2, affinity='precomputed', gamma=1000)
    lb = spec.fit_predict(gaussian)
    cols = ['red', 'purple']
    for i in range(2):
        plt.scatter(data[lb == i, 0], data[lb == i, 1], c=cols[i])
    plt.title(string + ': Spectral Clustering Algorithm Visualization with Gaussian Kernel')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

spectral_gk(square, af, "Square")
spectral_gk(elliptical, af1, "Elliptical")

#Cosine Similarity
#perform spectral clustering using jaccard similarity matrix metric
def spectral_cosine(data, string):
    cosine = cosine_similarity(data) #calculate cosine similarity
    cosine_scaled = MinMaxScaler().fit_transform(cosine) #apply min max scale
    spec = SpectralClustering(n_clusters=2, affinity='precomputed')
    lb = spec.fit_predict(cosine_scaled)
    cols = ['red', 'purple']
    for i in range(2):
        plt.scatter(data[lb == i, 0], data[lb == i, 1], c=cols[i])
    plt.title(string + ': Spectral Clustering Algorithm Visualization with Cosine Similarity')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

spectral_cosine(square, "Square")
spectral_cosine(elliptical, "Elliptical")


##Spectral Clustering Algorithm using Normalized Laplacian **
def spectral_norm_laplacian(data, string):
    sm = rbf_kernel(data, gamma=len(data[0])) #get similarity matrix
    embed = spectral_embedding(sm, n_components=2, norm_laplacian=True) #embed with normal laplacian parameter true
    a = SpectralClustering(n_clusters=2).fit_predict(embed) #apply spectral and get labels
    plt.scatter(data[:,0], data[:,1], c=a)
    plt.title(string + ': Spectral Clustering Algorithm Visualization with Normalized Laplacian')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

spectral_norm_laplacian(square, "Square")
spectral_norm_laplacian(elliptical, "Elliptical")

##Spectral Clustering Algorithm using Unnormalized Laplacian **
def spectral_un_laplacian(data, string):
    sm = rbf_kernel(data, gamma=len(data[0])) #get similarity matrix
    embed = spectral_embedding(sm, n_components=2, norm_laplacian=False) #embed with normal laplacian parameter false
    a = SpectralClustering(n_clusters=2).fit_predict(embed) #apply spectral and get labels
    plt.scatter(data[:,0], data[:,1], c=a)
    plt.title(string + ': Spectral Clustering Algorithm Visualization with Unnormalized Laplacian')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

spectral_un_laplacian(square, "Square")
spectral_un_laplacian(elliptical, "Elliptical")





#Real World Datasets - Part 2
#read in real world datasets and assign to numpy array
with open('Datasets/cho.txt', 'r') as file:
    rows = file.readlines()
    cho = []
    for row in rows:
        entry = row.split()
        entry = [float(i) for i in entry]
        cho.append(entry)
cho = np.array(cho)

with open('Datasets/iyer.txt', 'r') as file:
    rows = file.readlines()
    iyer = []
    for row in rows:
        entry = row.split()
        entry = [float(i) for i in entry]
        iyer.append(entry)
#this is iyer RAW dataset (no preprocessing)
iyer = np.array(iyer)

#preprocess iyer data by removing all entries where -1 is the ground truth label
iyer_preprocessed = []
for x in iyer:
    if x[1] != -1:
        iyer_preprocessed.append(x)
iyer_preprocessed = np.array(iyer_preprocessed)

#k means on real world datasets
def k_means_multi_dimensional(data, string, num_clusters):
    km = KMeans(n_clusters=num_clusters)
    km.fit(data)
    lb = km.labels_
    reduction = PCA(n_components=2).fit_transform(data) #reduce to 2 elements using PCA for visualization purposes
    plt.scatter(reduction[:, 0], reduction[:, 1], c=lb)
    plt.title(string + ': K-Means Clustering Algorithm Visualization with PCA Dimensionality Reduction')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

k_means_multi_dimensional(cho, "Cho", 5)
k_means_multi_dimensional(iyer_preprocessed, "Iyer Preprocessed", 10)


#spectral clustering on real world datasets
def spectral_multi_dimensional(data, string, num_clusters):
    spec = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', assign_labels='kmeans')
    lb = spec.fit_predict(data)
    reduction = PCA(n_components=2).fit_transform(data) #reduce to 2 elements using PCA for visualization purposes
    plt.scatter(reduction[:, 0], reduction[:, 1], c=lb)
    plt.title(string + ': Spectral Clustering Algorithm Visualization with PCA Dimensionality Reduction')
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.show()

spectral_multi_dimensional(cho, "Cho", 5)
spectral_multi_dimensional(iyer_preprocessed, "Iyer Preprocessed", 10)


#Validate Results:
#External Index: Accuracy Score
def kmeans_validate_results(num_clusters, data, string):
    count = 0
    for i in range(100):    #run 100 times and take average of accuracy scores
        km = KMeans(n_clusters=num_clusters)
        km.fit(data)
        lb = km.labels_
        k_labels_matched = np.empty_like(lb)
        # best-matching function adapted from LEC10_VISUALIZATION_SIMILARITY_MATRIX to reassign truth label values
        for k in np.unique(lb):
             # ...find and assign the best-matching truth label
            match_nums = [np.sum((lb==k)*(data[:,1]==t)) for t in np.unique(data[:,1])]
            k_labels_matched[lb==k] = np.unique(data[:,1])[np.argmax(match_nums)]
        accuracy = accuracy_score(k_labels_matched, data[:,1])
        count = count + accuracy
    count = count / 100         #get ratio
    count = count * 100         #get percentage
    print("Kmeans Accuracy for " + string + ": " + str(count))


def spectral_validate_results(num_clusters, data, string):
    count = 0
    for i in range(100):     #run 100 times and take average of accuracy scores
        spec = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', assign_labels='kmeans')
        lb = spec.fit_predict(data)
        k_labels_matched = np.empty_like(lb)
        # best-matching function adapted from LEC10_VISUALIZATION_SIMILARITY_MATRIX to reassign truth label values
        for k in np.unique(lb):
             # ...find and assign the best-matching truth label
            match_nums = [np.sum((lb==k)*(data[:,1]==t)) for t in np.unique(data[:,1])]
            k_labels_matched[lb==k] = np.unique(data[:,1])[np.argmax(match_nums)]
        accuracy = accuracy_score(k_labels_matched, data[:,1])
        count = count + accuracy
    count = count / 100         #get ratio
    count = count * 100         #get percentage
    print("Spectral Accuracy for " + string + ": " + str(count))

kmeans_validate_results(5, cho, "Cho")
spectral_validate_results(5, cho, "Cho")

kmeans_validate_results(10, iyer_preprocessed, "Iyer Preprocessed")
spectral_validate_results(10, iyer_preprocessed, "Iyer Preprocessed")


#Internal Index: Sum Squared Error (SSE)
def kmeans_validate_results_sse(num_clusters, data, string):
    count = 0
    for i in range(100):
        km = KMeans(n_clusters=num_clusters)
        km.fit(data)
        sse = km.inertia_
        count = count + sse
    count = count / 100
    print("Kmeans SSE for " + string + ": " + str(count))

def spectral_validate_results_sse(num_clusters, data, string):
    count = 0
    for i in range(100):
        spec = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors').fit(data)
        sse = 0
        for x in range(len(data)):
            l = spec.labels_[x] #get a label
            c = np.mean(data[spec.labels_ == l], axis=0)    #get centroids
            d = np.linalg.norm(data[x] - c) #calculate distance from center
            sse = sse + d * d   #square the distance and add to sse count
        count = count + sse
    count = count / 100
    print("Spectral SSE for " + string + ": " + str(count))

kmeans_validate_results_sse(5, cho, "Cho")
spectral_validate_results_sse(5, cho, "Cho")

kmeans_validate_results_sse(10, iyer_preprocessed, "Iyer Preprocessed")
spectral_validate_results_sse(10, iyer_preprocessed, "Iyer Preprocessed")


#Impacts of Data Normalization of K-Means and Spectral Clustering (Cho and Iyer preprocessed datasets)
def spectral_validate_results_normalized(num_clusters, data, string):
    min_max_scaler = MinMaxScaler() #min max input scaling
    normalized = min_max_scaler.fit_transform(data)
    spec = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', assign_labels='kmeans').fit_predict(normalized)
    k_labels_matched = np.empty_like(spec)
    for k in np.unique(spec):
        # ...find and assign the best-matching truth label
        match_nums = [np.sum((spec==k)*(data[:,1]==t)) for t in np.unique(data[:,1])]
        k_labels_matched[spec==k] = np.unique(data[:,1])[np.argmax(match_nums)]
    ac = accuracy_score(k_labels_matched, data[:,1])
    ac = ac * 100
    print("Spectral Accuracy for Normalized " + string + ": " + str(ac))


def kmeans_validate_results_normalized(num_clusters, data, string):
    min_max_scaler = MinMaxScaler() #min max input scaling
    normalized = min_max_scaler.fit_transform(data)
    km = KMeans(n_clusters=num_clusters)
    km.fit(normalized)
    spec = km.labels_
    k_labels_matched = np.empty_like(spec)
    for k in np.unique(spec):
        # ...find and assign the best-matching truth label
        match_nums = [np.sum((spec==k)*(data[:,1]==t)) for t in np.unique(data[:,1])]
        k_labels_matched[spec==k] = np.unique(data[:,1])[np.argmax(match_nums)]
    ac = accuracy_score(k_labels_matched, data[:,1])
    ac = ac * 100
    print("K-Means Accuracy for Normalized " + string + ": " + str(ac))

spectral_validate_results_normalized(5, cho, "Cho")
spectral_validate_results_normalized(10, iyer_preprocessed, "Iyer")

kmeans_validate_results_normalized(5, cho, "Cho")
kmeans_validate_results_normalized(10, iyer_preprocessed, "Iyer")


#Impacts of Noise on Iyer dataset:
#calculate accuracy scores for iyer RAW dataset and compare to iyer preprocessed
kmeans_validate_results(10, iyer, "Iyer")
spectral_validate_results(10, iyer, "Iyer")
