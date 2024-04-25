
import numpy as np
import matplotlib.pyplot as plt

class KMeansClustering:
    def __init__(self, k):
        self.k = k
        self.center = None

    def euclid_distance(self,dataPoint, centers):
        return np.sqrt(np.sum((centers-dataPoint)**2, axis=1))

    def create_center(self,X):
        minValues = np.amin(X, axis=0) #get min value of each axis
        maxValues = np.amax(X, axis=0) #get max value of each axis

        #generate random centers within range of data X,  X.shape is the shape of each point
        center = np.random.uniform(minValues, maxValues, size=(self.k, X.shape[1]) )
        return center
        
    # label every point
    def create_label(self,X):
        y = []
        for data_point in X:
            distances = self.euclid_distance(data_point, self.center)
            cluster_num = np.argmin(distances) #get the index of the min distance
            y.append(cluster_num)

        y=np.array(y)
        return y

    def create_cluster(self,X, label):
        cluster_indices = []
        for i in range(self.k):
            cluster_indices.append(np.argwhere(label ==i))

        return cluster_indices
        
    def update_centers(self, X, cluster):
        cluster_centers = []
        for i, indices in enumerate(cluster):
            if(len(indices)==0):
                cluster_centers.append(self.center[i])
            else:
                cluster_centers.append( np.mean(X[indices], axis=0)[0] ) #append the avarage point as new center

        if np.max(self.center - np.array(cluster_centers)) < 0.001:
            return False
        else:
            self.center = np.array(cluster_centers)
        return True

    def clustering(self, X, max_iterations = 10): #X is the list of points
        
        self.center = self.create_center(X)

        for i in range(max_iterations):

            label = self.create_label(X)

            cluster = self.create_cluster(X, label)

            if self.update_centers(X,cluster) == False:
                break
        return label


#testing
random_point = np.random.randint(0,100, (100,2))  ## create random_array = [[a,b], [c,d],...], length = 100
kmeans = KMeansClustering(k=3)
labels = kmeans.clustering(random_point, max_iterations= 100)

plt.scatter(random_point[:,0], random_point[:,1], c=labels)
plt.scatter(kmeans.center[:,0], kmeans.center[:,1], c=range(len(kmeans.center)), marker="*", s=200)
plt.show()