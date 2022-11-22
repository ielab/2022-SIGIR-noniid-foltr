"""
Test script for generating non-iid data
"""

from data.LetorDataset import LetorDataset
import numpy as np
from sklearn.cluster import KMeans

# dataset parameters
dataset = "Istella-s"  # ["MQ2007", "MQ2008", "MSLR10K", "Yahoo", "Istella-s"]
n_features = 220#220#700#136#46
dataset_path = f"./datasets/{dataset}"
n_folds = 1 #1#5
data_norm = True#False#True

for fold_id in range(n_folds):

    trainset = LetorDataset("{}/Fold{}/train.txt".format(dataset_path, fold_id+1),
                            n_features, query_level_norm=data_norm,
                            cache_root="./datasets/cache", abs_path=False)
    testset = LetorDataset("{}/Fold{}/test.txt".format(dataset_path, fold_id+1),
                           n_features, query_level_norm=data_norm,
                           cache_root="./datasets/cache", abs_path=False)
    validset = LetorDataset("{}/Fold{}/vali.txt".format(dataset_path, fold_id+1),
                           n_features, query_level_norm=data_norm,
                           cache_root="./datasets/cache", abs_path=False)

    query_rep_avg = dict()
    query_features = validset.get_query_get_all_features() #dict
    # print(sum(query_features['10']))
    # print(sum(query_features['10'])/len(query_features['10']))
    # print(np.mean(query_features['10'], 0))
    # print(sum(query_features['10'])/len(query_features['10']) - np.mean(query_features['10'], 0))

    keys = list(query_features.keys())

    # if fold_id == 0:
    #     for x in ['22447', '21169', '22624', '12367', '16942', '19666', '17272', '20827']:
    #         keys.remove(x)
    # elif fold_id == 1:
    #     for x in ['22447', '21169', '22624', '12367', '20560', '17272', '20827', '9265', '28285', '16942']:
    #         keys.remove(x)
    # elif fold_id == 2:
    #     for x in ['22447', '12367', '20560', '22528', '17272', '20827', '9265', '28285', '16942']:
    #         keys.remove(x)
    # elif fold_id == 3:
    #     for x in ['20560', '9265', '19666', '28285', '22528']:
    #         keys.remove(x)
    # elif fold_id == 4:
    #     for x in ['22528', '21169', '22624', '29299', '19666', '28924']:
    #         keys.remove(x)

    # if fold_id == 0:
    #     for x in ['22447', '21169', '22624', '12367', '16942', '19666', '17272', '20827']:
    #         print(len(query_features[x]))
    # elif fold_id == 1:
    #     for x in ['22447', '21169', '22624', '12367', '20560', '17272', '20827', '9265', '28285', '16942']:
    #         print(len(query_features[x]))
    # elif fold_id == 2:
    #     for x in ['22447', '12367', '20560', '22528', '17272', '20827', '9265', '28285', '16942']:
    #         print(len(query_features[x]))
    # elif fold_id == 3:
    #     for x in ['20560', '9265', '19666', '28285', '22528']:
    #         print(len(query_features[x]))
    # elif fold_id == 4:
    #     for x in ['22528', '21169', '22624', '29299', '19666', '28924']:
    #         print(len(query_features[x]))

    for key in keys:
        if len(query_features[key]) > 1:
            query_rep_avg[key] = sum(query_features[key]) / len(query_features[key])
    all_query = list(query_rep_avg.keys())
    print("Fold {fold_id+1}: number of queries is", len(all_query))

    features_for_kmeans = []
    for query in all_query:
        features_for_kmeans.append(query_rep_avg[query])

    kmeans = KMeans(n_clusters=5, random_state=5).fit(features_for_kmeans)
    # kmeans = KMeans(n_clusters=10, random_state=10).fit(features_for_kmeans)
    cluster = kmeans.labels_ #numpy.ndarray

    cluster_set = set(kmeans.labels_)
    for ele in cluster_set:
        print("Fold {}: number of cluster {} is: {}".format(fold_id+1, ele,np.sum(cluster == ele)))
