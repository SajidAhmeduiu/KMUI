# Test No Minority No PCA

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.model_selection import ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.cluster import AffinityPropagation
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import Adaboost
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import LocallyLinearEmbedding,TSNE
from prettytable import PrettyTable

def ensemble(X_train, X_test, y_train, y_test, depth, cluster):

    classifier = Adaboost.ExpAdaBoost(M=50,depth=depth)
    classifier.fit(X_train, y_train, X, y, train_index, test_index, cluster)
    predictions = classifier.predict_proba(X_test)

    return y_test,predictions


table_avg = PrettyTable()
table_avg.field_names = ["Dataset", "AUC AVG", "AUPR AVG"]

# dataset_list = ["glass0.dat","glass2.dat","glass5.dat","glass6.dat","glass-0-1-2-3_vs_4-5-6.dat","glass-0-1-4-6_vs_2.dat","glass-0-1-5_vs_2.dat","ecoli-0-3-4_vs_5.dat","led7digit-0-2-4-5-6-7-8-9_vs_1.dat","page-blocks-1-3_vs_4.dat","yeast-0-2-5-7-9_vs_3-6-8.dat","yeast-1_vs_7.dat","yeast-2_vs_4.dat","yeast4.dat","yeast5.dat","yeast6.dat","pima.csv","abalone9-18.csv","abalone19.csv"]
# dataset_list = ["glass0.dat"]
dataset_list = ["nr_ABCD.csv"]

for dataset in dataset_list:
    
    print("\n\t ## ",dataset," ##\n")
    df = pd.read_csv(dataset, header=None)
    df['label'] = df[df.shape[1] - 1]

    df.drop([df.shape[1] - 2], axis=1, inplace=True)
    labelencoder = LabelEncoder()
    df['label'] = labelencoder.fit_transform(df['label'])
    array = df['label']
    array[array == 0] = -1
    df['label'] = array

    # Dividing the dataset into X and Y
    X = np.array(df.drop(['label'], axis=1))
    y = np.array(df['label'])

    # Normalizing X
    normalization_object = Normalizer()
    X = normalization_object.fit_transform(X)

    # Hyperparameters for Adaboost classifier
    depth = 0
    auc_list_for_avg = []
    aupr_list_for_avg = []

    number_of_clusters = 23
    
    for i in range(0,5):
        auc_list_for_max = []
        aupr_list_for_max = []
        for depth in range(5,50,10):

            # Using shuffle split for X and Y
            sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2)
            sss.get_n_splits(X, y)
            train_index, test_index = next(sss.split(X, y))

            X_train,X_test = X[train_index], X[test_index]
            y_train,y_test = y[train_index], y[test_index]

            value, counts = np.unique(y_train, return_counts=True)
            minority_class = value[np.argmin(counts)]
            majority_class = value[np.argmax(counts)]

            idx_min = np.where(y_train == minority_class)[0]
            idx_maj = np.where(y_train == majority_class)[0]
            
            full_X = np.concatenate((X_train[idx_maj], X_test))
            full_y = np.concatenate((y_train[idx_maj], y_test))
  
            # Training the kmean model
            kmeans = KMeans(n_clusters=number_of_clusters)
            kmeans.fit(full_X)

            points_under_each_cluster = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}

            # From each cluster removing the test instances 
            for i in points_under_each_cluster.keys():
                temp = []
                for j in range(len(points_under_each_cluster[i])):
                    if points_under_each_cluster[i][j] not in test_index:
                        temp.append(points_under_each_cluster[i][j])   
                points_under_each_cluster[i] = np.array(temp)
            
            y_test, predictions = ensemble(X_train, X_test, y_train, y_test, depth, points_under_each_cluster)

            auc = roc_auc_score(y_test, predictions)
            auc_list_for_max.append(auc)
            aupr = average_precision_score(y_test, predictions)
            aupr_list_for_max.append(aupr)

        auc_our     = round(max(auc_list_for_max),5)
        aupr_our    = round(max(aupr_list_for_max),5)
        
        auc_list_for_avg.append(max(auc_list_for_max))
        aupr_list_for_avg.append(max(aupr_list_for_max))
        
    auc_avg_our     = round(sum(auc_list_for_avg)/len(auc_list_for_avg),5)
    aupr_avg_our    = round(sum(aupr_list_for_avg)/len(aupr_list_for_avg),5)
    
    table_avg.add_row([dataset, auc_avg_our, aupr_avg_our])    

print(table_avg)
table_avg.clear_rows()