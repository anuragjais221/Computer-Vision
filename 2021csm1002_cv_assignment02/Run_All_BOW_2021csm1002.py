import tensorflow as tf
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

# from scipy.cluster.vq import kmeans
import matplotlib.pyplot as plt
import random
import joblib
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
from numpy.random import uniform
import warnings
from scipy.cluster.vq import vq



(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()


with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)



"""KMeans Code
"""



def euclidean(point, data):
    return np.sqrt(np.sum((point - data)**2, axis=1))

class CustomKMeans:

    def __init__(self, n_clusters=8, init = "k-means++", n_init = 10, max_iter=300, tol=1e-4):
        # init = {"k-means++", "random"}
        # stop when either # iteration is greater than max_iter or the delta of self.inertia_ is smaller than tol.
        # repeat n_init times and keep the best run (cluster_centers_, inertia_) with the lowest inertia_.
        print('init')
        self.n_clusters = int(n_clusters)
        self.init = init
        self.n_init = n_init
        self.max_iter = int(max_iter)
        self.tol = tol

        self.classes_ = range(n_clusters)
        # Centroids
        self.cluster_centers_ = None
        # Sum of squared distances of samples to their closest cluster center.
        self.inertia_ = None

    def dist(self, point, data):
        return np.sqrt(np.sum((point - data)**2))

    def random_initialization(self, X):
        if self.init == 'k-means++':
          self.cluster_centers_ = [random.choice(X)]

        for _ in range(self.n_clusters-1):
            dists = np.sum([euclidean(centroid, X) for centroid in self.cluster_centers_], axis=0)
          
            dists /= np.sum(dists)
            
            new_centroid_idx = np.random.choice(range(len(X)), size=1, p=dists)[0]  
            self.cluster_centers_ += [X[new_centroid_idx]]
        else:
          min_, max_ = np.min(X, axis=0), np.max(X, axis=0)
          self.cluster_centers_ = [uniform(min_, max_) for _ in range(self.n_clusters)]

    def fit(self, X_):
        # X: pd.DataFrame, independent variables, float        
        # repeat self.n_init times and keep the best run 
            # (self.cluster_centers_, self.inertia_) with the lowest self.inertia_.
        X = X_
        self.random_initialization(X)
        iteration = 0
        prev_centroids=[]
        for cen in self.cluster_centers_:
          prev_centroids.append(np.zeros(cen.shape))
        # prev_centroids = [np.zeros(c.shape) for c in self.cluster_centers_]
        while np.not_equal(self.cluster_centers_, prev_centroids).any() and iteration < self.max_iter:
          with warnings.catch_warnings():
              warnings.simplefilter("ignore", category=RuntimeWarning)  
              sorted_points = [[] for _ in range(self.n_clusters)]
              for x in X:
                  dists = euclidean(x, self.cluster_centers_)
                  centroid_idx = np.argmin(dists)
                  sorted_points[centroid_idx].append(x)

              
              prev_centroids = self.cluster_centers_
              
              self.cluster_centers_ = [np.mean(cluster, axis=0) for cluster in sorted_points]
              for i, centroid in enumerate(self.cluster_centers_):
                  if np.isnan(centroid).any():  
                      self.cluster_centers_[i] = prev_centroids[i]
              iteration += 1
        self.inertia_ = np.sum([ np.min(dist)**2 for dist in self.transform(X_)])
        
        return

    def transform(self, X):
        # Transform to cluster-distance space
        # X: pd.DataFrame, independent variables, float
        # return dists = list of [dist to centroid 1, dist to centroid 2, ...]
        dists = [[self.dist(x,centroid) for centroid in self.cluster_centers_] for x in X]
        return dists

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        predictions = [np.argmin(dist) for dist in self.transform(X)]
        return predictions


    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def elbow_method(train_descriptor):

  asampled_list = random.sample(train_descriptor, 2000)
  sum_of_squares_distance = []
  for k in tqdm(range(1, 500)):
    km = KMeans(n_clusters=k, max_iter=200, n_init=10)
    km = km.fit(asampled_list)
    sum_of_squares_distance.append(km.inertia_)

  plt.figure(figsize=(100, 70))
  plt.plot(range(1,500),sum_of_squares_distance,color='red', marker='o', markerfacecolor='black', markersize=7)
  plt.xlabel('K')
  plt.ylabel('Sum of squared distances')
  plt.title('Elbow Method For Optimal k')
  plt.show()

def get_sift_feature_descriptor(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


def sift(data):
  descriptors = []
  desc=[]
  features = {}
  for index, img in tqdm(enumerate(data)):
    k, d = get_sift_feature_descriptor(img) 
    if d is not None:
      descriptors.extend(d)
      desc.append(d)
      features[index] = d
  
  return features,descriptors,desc


def get_cluster_centers(train_desc):
  print('get_cluster_center')
  optimal_k = 200
  kmeans = CustomKMeans(n_clusters=optimal_k)
  # kmeans.fit(train_desc)
  dists = kmeans.fit_transform(train_desc)
  bovw = kmeans.cluster_centers_
  # print(type(bow))
  print(np.array(bovw).shape) 
  return optimal_k,kmeans,bovw,dists


def CreateVisualDictionary(train_feature,train_descriptors,train_desc):
  print("create visual dictionary")
  optimal_k,kmeans,bovw,dists = get_cluster_centers(train_descriptors)
  joblib.dump((optimal_k, bovw), "bovw-codebook.pkl", compress=3)
  np.savetxt('foo.txt', dists, fmt='%d')
  # visual_words = []
  # for img_descr in train_desc:
  #     # print(img_descr.shape)
  #     # print(bovw.shape)
  #     # for each image, map each descriptor to the nearest codebook entry
  #     img_visual_words, distance = vq(img_descr, bovw)
  #     visual_words.append(img_visual_words)
  # print(type(visual_words))
  return optimal_k,kmeans,bovw,dists


def ComputeHistogram(features, bovw,kmeans):
  histograms = {}
  for img in tqdm(features):
    # all descriptors of the img
    all_descriptor = np.array(features[img],np.double)
    histogram = np.zeros(len(bovw),dtype=np.double)
    predictions = kmeans.predict(all_descriptor)
    # for each des, find the cluster and create histogram
    for pred in predictions:
      histogram[pred] += 1
    # update global histograms
    histograms[img] = histogram
  return histograms
  

def get_train_and_test(train_hist, y_train,test_hist,y_test):
  trainX = []
  trainY = []
  testX = []
  testY = []
  for x in train_hist:
    trainX.append(train_hist[x])
    trainY.append(y_train[x])

  for x in test_hist:
    testX.append(test_hist[x])
    testY.append(y_test[x])
  return trainX,trainY,testX,testY


def train_fit(Xtrain,Ytrain):
  model = LinearSVC(max_iter=100000)  #Default of 100 is not converging
  model.fit(Xtrain, Ytrain)
  return model


def test_fit(model,Xtest,Ytest):
  predictions = model.predict(Xtest)
  return predictions

# def MatchHistogram(hist1,hist2):
#   return np.sqrt(np.sum((hist1 - hist2)**2, axis=1))

def MatchHistogram(Xtrain,Xtest,Ytrain,Ytest,train_desc,bovw):
  model = LinearSVC(max_iter=100000)  #Default of 100 is not converging
  model.fit(Xtrain, Ytrain)
  predictions = model.predict(Xtest)
  print('Accuracy =', accuracy_score(Ytest, predictions))
  print('Confusion Matrix:\n', confusion_matrix(Ytest, predictions))
  print(classification_report(Ytest, predictions))
  visual_words = []
  distances = []
  for img_descr in train_desc:
      # print(img_descr.shape)
      # print(bovw.shape)
      # for each image, map each descriptor to the nearest codebook entry
      img_visual_words, distance = vq(img_descr, bovw)
      visual_words.append(img_visual_words)
      distances.append(distance)
  # print(type(visual_words))
  # return np.sqrt(np.sum((np.array(Xtrain) - np.array(Xtest))**2, axis=1))
  return distances,visual_words 


# x_train = x_train[:1000]
# y_train = y_train[:1000]

# # x_test = x_test[:1000]
# # y_test = y_test[:1000]
# train_feature , train_descriptor, train_desc = sift(x_train)

# test_feature, test_descriptor , test_desc = sift(x_test)

# optimal_k,kmeans,bovw,dists = CreateVisualDictionary(train_feature,train_descriptor,train_desc)

# train_histogram = ComputeHistogram(train_feature,bovw,kmeans)

# test_histogram = ComputeHistogram(test_feature, bovw,kmeans)



# Xtrain_hist,Ytrain,Xtest_hist,Ytest = get_train_and_test(train_histogram, y_train,test_histogram,y_test)



# distances = MatchHistogram(Xtrain_hist,Xtest_hist,Ytrain,Ytest,train_desc)

# print('dist',distances)
# np.savetxt('foo1.txt', distances)


