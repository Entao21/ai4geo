# GEOL0069 Week 4- Echo Classification (Unsupervised Learning)

## **Goal of this week:** 
1) Classify SRAL echoes into lead and sea ice by using unsupervised machine learning methods.
2) Plot mean and standard deviation echo shapes for each class.
3) Evaluate our result against ESA official classification by using a confusion matrix.

## Link to full ipynb notebook
📓 Notebook: [notebooks/Week4_Echo_Classification.ipynb](notebooks/Week4_Echo_Classification.ipynb).
You can access the full version of this week's code by clicking above link.

## **Methods**
### 1) K-means Clustering 
### ▶ Definition
K-means clustering is a type of unsupervised learning algorithm used for partitioning a dataset into a set of k groups (or clusters), where k represents the number of groups pre-specified by the analyst. It classifies the data points based on the similarity of the features of the data {cite}macqueen1967some. The basic idea is to define k centroids, one for each cluster, and then assign each data point to the nearest centroid, while keeping the centroids as small as possible.

### ▶ Why K-means for Clustering?
- The structure of the data is not known beforehand: K-means doesn’t require any prior knowledge about the data distribution or structure, making it ideal for exploratory data analysis.
- Simplicity and scalability: The algorithm is straightforward to implement and can scale to large datasets relatively easily.

### ▶ Key Components of K-means
1) Choosing K: The number of clusters (k) is a parameter that needs to be specified before applying the algorithm.
2) Centroids Initialization: The initial placement of the centroids can affect the final results.
3) Assignment Step: Each data point is assigned to its nearest centroid, based on the squared Euclidean distance.
4) Update Step: The centroids are recomputed as the center of all the data points assigned to the respective cluster.

### ▶ The Iterative Process of K-means
The assignment and update steps are repeated iteratively until the centroids no longer move significantly, meaning the within-cluster variation is minimised. This iterative process ensures that the algorithm converges to a result, which might be a local optimum.

### ▶ Advantages of K-means
- Efficiency: K-means is computationally efficient.
- Ease of interpretation: The results of k-means clustering are easy to understand and interpret.

### 2) Gaussian Mixture Models (GMM)
### ▶ Definition
Gaussian Mixture Models (GMM) are a probabilistic model for representing normally distributed subpopulations within an overall population. The model assumes that the data is generated from a mixture of several Gaussian distributions, each with its own mean and variance {cite}reynolds2009gaussian, mclachlan2004finite. GMMs are widely used for clustering and density estimation, as they provide a method for representing complex distributions through the combination of simpler ones.

### ▶ Why Gaussian Mixture Models for Clustering?
- Soft clustering is needed: Unlike K-means, GMM provides the probability of each data point belonging to each cluster, offering a soft classification and understanding of the uncertainties in our data.
- Flexibility in cluster covariance: GMM allows for clusters to have different sizes and different shapes, making it more flexible to capture the true variance in the data.

### ▶ Key Components of GMM
1) Number of Components (Gaussians): Similar to K in K-means, the number of Gaussians (components) is a parameter that needs to be set.
2) Expectation-Maximization (EM) Algorithm: GMMs use the EM algorithm for fitting, iteratively improving the likelihood of the data given the model.
3) Covariance Type: The shape, size, and orientation of the clusters are determined by the covariance type of the Gaussians (e.g., spherical, diagonal, tied, or full covariance).

### ▶ The EM Algorithm in GMM
The Expectation-Maximization (EM) algorithm is a two-step process:
1) Expectation Step (E-step): Calculate the probability that each data point belongs to each cluster.
2) aximization Step (M-step): Update the parameters of the Gaussians (mean, covariance, and mixing coefficient) to maximize the likelihood of the data given these assignments.
This process is repeated until convergence, meaning the parameters do not significantly change from one iteration to the next.

### ▶ Advantages of GMM
- Soft Clustering: Provides a probabilistic framework for soft clustering, giving more information about the uncertainties in the data assignments.
- Cluster Shape Flexibility: Can adapt to ellipsoidal cluster shapes, thanks to the flexible covariance structure.

## **Workflow of this notebook**
1) Install the necessary package and setup environment.
2) Download data.
3) Implement unsupervised method (GMM) to these data.
4) Obtain result and save,

## **Getting Started**
### ▶ Where to run this notebook?
- Google Colab. Google colab will provide you access to strong AI training hardware and it would greatly help you to run your code.

### ▶ Environment set up.
- Please run ALL of below code to set up basic environment for this notebook. 
``` python
!pip install rasterio
!pip install netCDF4
!pip install basemap
!pip install cartopy
```
- Please allow access to your google drive while running.
```
from google.colab import drive
drive.mount('/content/drive')
```

### ▶ Want to try GMM and K-Means model on Sentinel-2 Image?
Please download the notebook and nevigate to 'K-means Implementation' and 'GMM Implementation' section for detailed code and results. This instruction will be mainly focused on to use GMM to do altimetry classification on sea ice and leads.

## **Result**
### ▶ Mean and standard deviation of all the echoes:
```
mean_ice = np.mean(waves_cleaned[clusters_gmm==0],axis=0)
std_ice = np.std(waves_cleaned[clusters_gmm==0], axis=0)

plt.plot(mean_ice, label='ice')
plt.fill_between(range(len(mean_ice)), mean_ice - std_ice, mean_ice + std_ice, alpha=0.3)


mean_lead = np.mean(waves_cleaned[clusters_gmm==1],axis=0)
std_lead = np.std(waves_cleaned[clusters_gmm==1], axis=0)

plt.plot(mean_lead, label='lead')
plt.fill_between(range(len(mean_lead)), mean_lead - std_lead, mean_lead + std_lead, alpha=0.3)

plt.title('Plot of mean and standard deviation for each class')
plt.legend()
```
! [Mean and standard deviation for all echoes]()

### ▶ All echoes:
```
x = np.stack([np.arange(1,waves_cleaned.shape[1]+1)]*waves_cleaned.shape[0])
plt.plot(x,waves_cleaned)  # plot of all the echos
plt.show()
```
! [All Echoes]()

### ▶ Sea ice and lead echoes seperated- diagram for each:
1) Sea ice echoes profile
```
x = np.stack([np.arange(1,waves_cleaned[clusters_gmm==0].shape[1]+1)]*waves_cleaned[clusters_gmm==0].shape[0])
plt.plot(x,waves_cleaned[clusters_gmm==0])  # plot of all the echos
plt.show()
```
! [Sea ice echoes profile]()

2) Leads echoes profile
```
x = np.stack([np.arange(1,waves_cleaned[clusters_gmm==1].shape[1]+1)]*waves_cleaned[clusters_gmm==1].shape[0])
plt.plot(x,waves_cleaned[clusters_gmm==1])  # plot of all the echos
plt.show()
```
! [Lead echoes profile]()

### ▶ Compare with ESA data (Confusion Matrix)
In the ESA dataset, sea ice = 1 and lead = 2. Therefore, we need to subtract 1 from it so our predicted labels are comparable with the official product labels.
! [Confusion Matrix]()
