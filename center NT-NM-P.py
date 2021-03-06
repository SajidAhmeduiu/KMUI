print("NO TEST NO MINORITY PCA ")
import numpy as np
import pandas as pd
import math
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import euclidean_distances
from prettytable import PrettyTable

table_avg = PrettyTable()

table_avg.field_names = ["Dataset", "AUC AVG(Our)", "AUPR AVG(Our)"]

# dataset_list = ["glass0.dat","glass2.dat","glass5.dat","glass6.dat","glass-0-1-2-3_vs_4-5-6.dat","glass-0-1-4-6_vs_2.dat","glass-0-1-5_vs_2.dat","ecoli-0-3-4_vs_5.dat","led7digit-0-2-4-5-6-7-8-9_vs_1.dat","page-blocks-1-3_vs_4.dat","yeast-0-2-5-7-9_vs_3-6-8.dat","yeast-1_vs_7.dat","yeast-2_vs_4.dat","yeast4.dat","yeast5.dat","yeast6.dat","pima.csv","abalone9-18.csv","abalone19.csv"]
dataset_list = ["nr_ABCD.csv"]

for dataset in dataset_list:
    
    # print("\n\t ## ",dataset," ##\n")
    df = pd.read_csv(dataset, header=None)
    df['label'] = df[df.shape[1] - 1]

    df.drop([df.shape[1] - 2], axis=1, inplace=True)
    labelencoder = LabelEncoder()
    df['label'] = labelencoder.fit_transform(df['label'])

    # Dividing the dataset into X and Y
    X = np.array(df.drop(['label'], axis=1))
    X_original = X
    y = np.array(df['label'])
    y_original = y

    # Normalizing X
    normalization_object = Normalizer()
    X = normalization_object.fit_transform(X)

    auc_list_for_avg = []
    auc_list1_for_avg = []
    aupr_list_for_avg = []
    aupr_list1_for_avg = []
            
    for i in range(0,5):
        auc_list_for_max = []
        auc_list1_for_max = []
        aupr_list_for_max = []
        aupr_list1_for_max = []
        for depth in range(5,50,10):

            # Getting train and test instances
            sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2)
            sss.get_n_splits(X, y)
            train_index, test_index = next(sss.split(X, y))

            # Sampling Indexes
            X_train,X_test = X[train_index], X[test_index]
            y_train,y_test = y[train_index], y[test_index]

            value, counts = np.unique(y_train, return_counts=True)
            minority_class = value[np.argmin(counts)]
            majority_class = value[np.argmax(counts)]

            idx_min = np.where(y_train == minority_class)[0]
            idx_maj = np.where(y_train == majority_class)[0]
            
            number_of_clusters = len(idx_min)
                
            # Adding PCA Method
            transformer = KernelPCA(n_components=math.ceil(X_train.shape[1]/3), kernel='poly')
            X_transformed = transformer.fit_transform(X_train)    

            # Training the kmean model
            kmeans = KMeans(n_clusters=number_of_clusters)
            kmeans.fit(X_transformed[idx_maj])

            points_under_each_cluster = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}
            centers = kmeans.cluster_centers_

            min_distance_list = []
            for key,value in points_under_each_cluster.items():
                min_distance = {}
                for index in value:
                    data = X_transformed[index]
                    distance_from_data = euclidean_distances([data], [centers[key]])
                    min_distance[index] = distance_from_data
                min_distance_list.append(min(min_distance))

            majority_data_X = X_train[min_distance_list]
            minority_data_X = X_train[idx_min]
            majority_data_y = y_train[min_distance_list]
            minority_data_y = y_train[idx_min]

            X_sampled = np.concatenate((majority_data_X, minority_data_X))
            y_sampled = np.concatenate((majority_data_y, minority_data_y))

            classifier = AdaBoostClassifier(
                DecisionTreeClassifier(max_depth=depth),
                n_estimators=50,
                learning_rate=1, algorithm='SAMME')

            classifier.fit(X_sampled, y_sampled)

            predictions = classifier.predict_proba(X_test)

            auc = roc_auc_score(y_test, predictions[:,1])
            auc_list_for_max.append(auc)
            aupr = average_precision_score(y_test, predictions[:,1])
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