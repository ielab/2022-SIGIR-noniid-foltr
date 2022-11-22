import os
from tqdm import tqdm
import random
import json
import numpy as np
from data.LetorDataset import LetorDataset
#
#
# dataset = "Yahoo"
# dataset_path = "datasets/Yahoo/"
# fold_id = 1
# n_features = 700
# trainset = LetorDataset(f"datasets/Yahoo/Fold{fold_id}/train.txt",
#                         n_features, query_level_norm=True,  # False(MQ),True(MS)
#                         cache_root="datasets/cache",
#                         abs_path=False)  # "/media/bigdata/uqbliu3/ltr_datasets/cache" "./datasets/cache"
# testset = LetorDataset(f"datasets/MSLR10K/Fold{fold_id}/test.txt",
#                        n_features, query_level_norm=True,  # False(MQ),True(MS)
#                        cache_root="datasets/cache",
#                        abs_path=False)  # "/media/bigdata/uqbliu3/ltr_datasets/cache" "./datasets/cache"
#

###################################################
#################### LDS2 - V1 ####################
###################################################
for fold in tqdm(range(1)):
    fold_id = fold + 1
    path = f"../datasets/Istella-s/Fold{fold_id}/train.txt"
    lines_0 = ""
    lines_1 = ""
    lines_2 = ""
    lines_3 = ""
    lines_4 = ""
    lines_5 = ""
    lines_6 = ""
    lines_7 = ""
    lines_8 = ""
    lines_9 = ""
    with open(path, "r") as f:
        for line in f:
            cols = line.strip().split()
            query = cols[1].split(":")[1]
            relevance = float(cols[0]) # Sometimes the relevance label can be a float
            if relevance.is_integer():
                relevance = int(relevance) # But if it is indeed an int, cast it into one.

            if relevance == 0 or relevance == 1:
                lines_0 += line
            if relevance == 0 or relevance == 2:
                lines_1 += line
            if relevance == 0 or relevance == 3:
                lines_2 += line
            if relevance == 0 or relevance == 4:
                lines_3 += line
            if relevance == 1 or relevance == 2:
                lines_4 += line
            if relevance == 1 or relevance == 3:
                lines_5 += line
            if relevance == 1 or relevance == 4:
                lines_6 += line
            if relevance == 2 or relevance == 3:
                lines_7 += line
            if relevance == 2 or relevance == 4:
                lines_8 += line
            if relevance == 3 or relevance == 4:
                lines_9 += line

    new_data = [lines_0, lines_1, lines_2, lines_3, lines_4, lines_5, lines_6, lines_7, lines_8, lines_9]
    # print("length of new_data:", len(lines_0))
    # print("type of new_data:", type(new_data[0]))
    # print(new_data[0][0])


    for label in range(10):
        save_path = f"../datasets/Istella-s/LDS2_V1/Fold{fold_id}/label_{label}/train.txt"

        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        with open(save_path, 'w') as f:
            f.write(new_data[label])



def split_LDS1_dataset(dataset, n_features, fold_id, label, norm=True):
    lines_1 = ""
    lines_2 = ""
    lines_3 = ""
    lines_4 = ""
    lines_len = 0
    count = 0
    path = f"../datasets/{dataset}/LDS/Fold{fold_id}/label_{label}/train.txt"
    data = LetorDataset(path, n_features, query_level_norm=norm, cache_root="../datasets/cache", abs_path=False)

    query_set = data.get_all_querys()
    for query in query_set:
        label_list = data.get_all_relevance_label_by_query(query)
        lines_len += len(label_list)

    print("total lens:", lines_len)
    print((lines_len/4)*3, (lines_len/4)*2, (lines_len/4))

    with open(path, "r") as f:
        for line in f:
            if count > (lines_len/4)*3:
                lines_4 += line
            elif count > (lines_len/4)*2:
                lines_3 += line
            elif count > (lines_len/4):
                lines_2 += line
            else:
                lines_1 += line
            count += 1
    return lines_1, lines_2, lines_3, lines_4





###################################################
#################### LDS2 - V2 ####################
###################################################
# for fold in tqdm(range(5)):
#     fold_id = fold + 1
#     path = f"../datasets/MSLR10K/Fold{fold_id}/train.txt"
#     # lines_0 = ""
#     # lines_1 = ""
#     # lines_2 = ""
#     # lines_3 = ""
#     # lines_4 = ""
#     # lines_5 = ""
#     # lines_6 = ""
#     # lines_7 = ""
#     # lines_8 = ""
#     # lines_9 = ""
#
#     train_0 = LetorDataset(path,feature_size=136, query_level_norm=True, cache_root="../datasets/cache", abs_path=False)
#
#     query_set = train_0.get_all_querys()
#     random.Random(1).shuffle(query_set)
#     query_set_split = np.split(query_set, 10)
#
#     label_set = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
#
#     for label in range(10):
#         queries = query_set_split[label]
#         labels = label_set[label]
#         lines = ""
#         with open(path, "r") as f:
#             for line in f:
#                 cols = line.strip().split()
#                 query = cols[1].split(":")[1]
#                 relevance = float(cols[0]) # Sometimes the relevance label can be a float
#                 if relevance.is_integer():
#                     relevance = int(relevance) # But if it is indeed an int, cast it into one.
#
#                 if (query in queries) and (relevance in labels):
#                     lines += line
#
#         save_path = f"../datasets/MSLR10K/LDS2_V3/Fold{fold_id}/label_{label}/train.txt"
#
#         if not os.path.exists(os.path.dirname(save_path)):
#             os.makedirs(os.path.dirname(save_path))
#
#         with open(save_path, 'w') as f:
#             f.write(lines)



###################################################
############ LDS1 - V2 (data sharing) #############
###################################################
# for fold in tqdm(range(5)):
#     fold_id = fold + 1
#     path = f"../datasets/MSLR10K/Fold{fold_id}/train.txt"
#
#     train_0 = LetorDataset(path,feature_size=136, query_level_norm=True, cache_root="../datasets/cache", abs_path=False)
#
#     query_set = train_0.get_all_querys()
#     random_state = np.random.RandomState(1)
#     n_query = 600 # select 10% of data
#     index = random_state.randint(len(query_set), size=n_query)
#     query_selected = []
#     for i in range(n_query):
#         id = index[i]
#         qid = query_set[id]
#         query_selected.append(qid)
#
#     lines_selected = ""
#     with open(path, "r") as f:
#         for line in f:
#             cols = line.strip().split()
#             query = cols[1].split(":")[1]
#
#             if query in query_selected:
#                 lines_selected += line
#
#     for label in range(5):
#         noniid_path = f"../datasets/MSLR10K/LDS/Fold{fold_id}/label_{label}/train.txt"
#         lines_origin = ""
#         with open(noniid_path, "r") as f:
#             for line in f:
#                 cols = line.strip().split()
#                 query = cols[1].split(":")[1]
#
#                 if query not in query_selected:
#                     lines_origin += line
#
#         lines_new = lines_origin + lines_selected
#         save_path = f"../datasets/MSLR10K/LDS_V3/Fold{fold_id}/label_{label}/train.txt"
#
#         if not os.path.exists(os.path.dirname(save_path)):
#             os.makedirs(os.path.dirname(save_path))
#
#         with open(save_path, 'w') as f:
#             f.write(lines_new)



# for fold_id in range(5):
#     fold_id += 1
#     n_features = 136
#     for label in range(10):
#         train_0 = LetorDataset(f"../datasets/MSLR10K/LDS2_V2/Fold{fold_id}/label_{label}/train.txt",
#                                 n_features, query_level_norm=True,
#                                 cache_root="../datasets/cache",
#                                 abs_path=False)
#
#         query_set = train_0.get_all_querys()
#         print("length of query_set:", np.shape(query_set))
#
#         for rel in range(5):
#             a = 0
#             for query in query_set:
#                 a += train_0.get_all_relevance_label_by_query(query).count(rel)
#             print("label:", rel, a)




# rel_len = [377957, 232569, 95082, 12658, 5146]
# query_set_len = [5999, 5800, 5396, 3494, 1662]
# rel_len = [373029, 230368, 95117, 12814, 5355]
# query_set_len = [6000, 5803, 5396, 3538, 1709]
# rel_len = [371725, 232302, 96663, 12903, 5518]
# query_set_len = [5998, 5784, 5415, 3582, 1712]
# rel_len = [372756, 231727, 96244, 12712, 5329]
# query_set_len = [5997, 5779, 5401, 3559, 1679]
# rel_len = [377322, 231874, 95247, 12864, 5295]
# query_set_len = [5997, 5787, 5386, 3512, 1653]
