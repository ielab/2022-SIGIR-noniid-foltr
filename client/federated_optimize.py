from typing import Dict, Any, NamedTuple, List
import numpy as np
from tqdm import tqdm

from client.client import RankingClient_iid
from client.client import RankingClient_iid_LRS
from client.client import RankingClient_iid_LRS_V2
from client.client import RankingClient_iid_intent_change
from client.client_noniid import RankingClient_noniid
from client.client_noniid import RankingClient_noniid_LRS # 'click preference skew' non-iid clients
from ranker.PDGDLinearRanker import PDGDLinearRanker
from data.LetorDataset import LetorDataset
from utils import intent_change

TrainResult = NamedTuple("TrainResult", [
    ("ranker", PDGDLinearRanker),
    ("ndcg_server", list),
    ("mrr_server", list),
    ("ndcg_client", list),
    ("mrr_client", list)
])


def train_uniform(params: Dict[str, Any], traindata: LetorDataset, testdata: LetorDataset, message, num_update=None, is_iid=True, non_iid_type=None, save_path=None) -> TrainResult:
    """

    :param params:
    :param traindata: dataset used for training server ranker
    :param testdata: dataset used for testing true performance of server ranker - using true relevance label
    :param message:
    :return:
    """
    seed = params["seed"]

    n_clients = params["n_clients"]
    interactions_per_feedback = params["interactions_per_feedback"]
    click_model = params["click_model"]
    ranker = params["ranker_generator"]
    multi_update = params["multi_update"]
    sensitivity = params["sensitivity"]
    epsilon = params["epsilon"]
    enable_noise = params["enable_noise"]
    intent_path = params["intent_path"]
    is_personal_layer = params["is_personal_layer"]
    is_mixed = params["is_mixed"] # set True if you want to pair this non-iid type with "data_quan_skew"
    fed_alg = params["fed_alg"]
    mu = params["mu"]

    if is_iid == True and non_iid_type == None: # the original iid one
        clients = [RankingClient_iid(traindata, ranker, seed * n_clients + client_id, click_model, sensitivity, epsilon, enable_noise, n_clients, client_id) for client_id in range(n_clients)]
    elif is_iid == True and non_iid_type == "label_dist_skew": # the (query_click pair) iid to compare with 'label_dist_skew' non-iid
        clients = [RankingClient_iid(traindata, ranker, seed * n_clients + client_id, click_model, sensitivity, epsilon, enable_noise, n_clients, client_id) for client_id in range(n_clients)]
    elif is_iid == True and non_iid_type == "click_pref_skew":
        clients = [RankingClient_iid_LRS_V2(traindata, ranker, seed * n_clients + client_id, click_model, sensitivity, epsilon, enable_noise, n_clients) for client_id in range(n_clients)]
    ## using the intent-change data: non-iid only in the label data
    elif is_iid == True and non_iid_type == "intent_change":
        dataset, testdata = intent_change.get_intent_dataset(traindata, testdata, "{}/0.txt".format(intent_path))
        clients = [RankingClient_iid(dataset, ranker, seed * n_clients + client_id, click_model, sensitivity, epsilon, enable_noise, n_clients, client_id) for client_id in range(n_clients)]

    elif is_iid == False and non_iid_type == "label_dist_skew": # the (query_click pair) 'label_dist_skew' non-iid
        clients = [RankingClient_iid(traindata[client_id], ranker, seed * n_clients + client_id, click_model, sensitivity, epsilon, enable_noise, n_clients, client_id) for client_id in range(n_clients)]
    elif is_iid == False and non_iid_type == "click_pref_skew":
        clients = [RankingClient_noniid_LRS(traindata, ranker, seed * n_clients + client_id, click_model[client_id], sensitivity, epsilon, enable_noise, n_clients, traindata.get_all_querys()) for client_id in range(n_clients)]
    ## using the intent-change data: non-iid only in the label data
    elif is_iid == False and non_iid_type == "intent_change":
        intent_paths = ["{}/1.txt".format(intent_path),
                        "{}/2.txt".format(intent_path),
                        "{}/3.txt".format(intent_path),
                        "{}/4.txt".format(intent_path)]
        datasets = intent_change.get_intent_groups_dataset(traindata, intent_paths, seed=seed)
        clients = [RankingClient_iid(datasets[client_id], ranker, seed * n_clients + client_id, click_model, sensitivity, epsilon, enable_noise, n_clients, client_id) for client_id in range(n_clients)]
    elif is_iid == False and non_iid_type == "data_quan_skew":
        clients = [RankingClient_iid(traindata, ranker, seed * n_clients + client_id, click_model, sensitivity, epsilon, enable_noise, n_clients, client_id) for client_id in range(n_clients)]

    else:
        pass

    n_iterations = params["interactions_budget"] # total iteration times (training times) for federated training

    ndcg_server = [] # off-line metric (on testset)
    mrr_server = [] # off-line metric (on testset)
    ndcg_clients = []
    mrr_clients = []

    if is_iid == False and non_iid_type == "intent_change":
        for i in range(len(datasets)):
            ndcg_server.append([])
            mrr_server.append([])

    # initialize gradient
    if is_iid == False and non_iid_type == "label_dist_skew":
        gradients = np.zeros(traindata[0]._feature_size)
    else:
        gradients = np.zeros(traindata._feature_size)

    for i in tqdm(range(n_iterations), desc=message):
        i += 1
        feedback = []
        online_ndcg = []
        online_mrr = []
        for client in clients:

            if is_iid == False and (non_iid_type == "data_quan_skew" or is_mixed == True):
                batch_size = [1, 3, 5, 7, 9]
                # batch_size = [9, 7, 5, 3, 1]
                interactions_per_feedback = batch_size[client.client_id]
                client_message, client_metric = client.client_ranker_update(interactions_per_feedback, multi_update)
            else:
                client_message, client_metric = client.client_ranker_update(interactions_per_feedback, multi_update, fed_alg=fed_alg, mu=mu)

            feedback.append(client_message)
            # online evaluation
            online_ndcg.append(client_metric.mean_ndcg)
            online_mrr.append(client_metric.mean_mrr)


        # online-line metrics
        ndcg_clients.append(np.mean(online_ndcg))
        mrr_clients.append(np.mean(online_mrr))

        # off-line metrics
        if is_iid == False and non_iid_type == "intent_change":
            for i in range(len(datasets)):
                all_result = ranker.get_all_query_result_list(datasets[i])
                ndcg = average_ndcg_at_k(datasets[i], all_result, 10)
                mrr = average_mrr_at_k(datasets[i], all_result, 10)
                ndcg_server[i].append(ndcg)
                mrr_server[i].append(mrr)
        else:
            all_result = ranker.get_all_query_result_list(testdata)
            ndcg = average_ndcg_at_k(testdata, all_result, 10)
            mrr = average_mrr_at_k(testdata, all_result, 10)
            ndcg_server.append(ndcg)
            mrr_server.append(mrr)

        # train the server ranker (clients send feedback to the server)
        ranker.federated_averaging_weights(feedback)

        # the server send the newly trained model to every client
        for client in clients:
            if is_personal_layer:
                client.update_model_personal_layer(ranker)
            else:
                client.update_model(ranker)

        if i % 499 == 0 and i > 0:
            tmp_result = TrainResult(ranker=ranker, ndcg_server=ndcg_server, mrr_server=mrr_server, ndcg_client=ndcg_clients,
                        mrr_client=mrr_clients)
            np.save(save_path, tmp_result)

    return TrainResult(ranker=ranker, ndcg_server = ndcg_server, mrr_server=mrr_server, ndcg_client=ndcg_clients, mrr_client=mrr_clients)



def get_iid_query_click_pair(query_group: list, click_model_list: list, seed: int):
    
    query_group_shuffle = []
    for i in range(len(query_group)):
        random_state = np.random.RandomState(seed)
        query_set_shuffle = random_state.permutation(query_group[i])
        query_group_shuffle.append(query_set_shuffle)
        # group_len = int(len(query_set_shuffle) / len(click_model_list))

    query_iid_group = []
    for i in range(len(click_model_list)):
        query_iid_group_i = np.array([])
        for j in range(len(query_group_shuffle)):
            query_group_shuffle_j = query_group_shuffle[j]
            group_len = int(len(query_group_shuffle_j) / len(click_model_list))
            if i == 0:
                query_iid_group_i = np.concatenate((query_iid_group_i, query_group_shuffle_j[: group_len * (i + 1)]))  # type : <class 'numpy.ndarray'>
            elif i == (len(click_model_list) - 1):
                query_iid_group_i = np.concatenate((query_iid_group_i, query_group_shuffle_j[group_len * i:]))  # type : <class 'numpy.ndarray'>
            else:
                query_iid_group_i = np.concatenate((query_iid_group_i, query_group_shuffle_j[group_len * i: group_len * (i + 1)]))  # type : <class 'numpy.ndarray'>

        query_iid_group.append(query_iid_group_i)

    return query_iid_group  # type : list



#ndcg@k & MRR
def average_ndcg_at_k(dataset, query_result_list, k):
    ndcg = 0.0
    num_query = 0
    for query in dataset.get_all_querys():
        if len(dataset.get_relevance_docids_by_query(query)) == 0:  # for this query, ranking list is None
            # num_query += 1
            continue
        else:
            pos_docid_set = set(dataset.get_relevance_docids_by_query(query))
        dcg = 0.0
        for i in range(0, min(k, len(query_result_list[query]))):
            docid = query_result_list[query][i]
            relevance = dataset.get_relevance_label_by_query_and_docid(query, docid)
            dcg += ((2 ** relevance - 1) / np.log2(i + 2))

        rel_set = []
        for docid in pos_docid_set:
            rel_set.append(dataset.get_relevance_label_by_query_and_docid(query, docid))
        rel_set = sorted(rel_set, reverse=True)
        n = len(pos_docid_set) if len(pos_docid_set) < k else k

        idcg = 0
        for i in range(n):
            idcg += ((2 ** rel_set[i] - 1) / np.log2(i + 2))

        if idcg != 0:
            ndcg += (dcg / idcg)

        num_query += 1
    return ndcg / float(num_query)

def average_mrr_at_k(dataset: LetorDataset, query_result_list, k):
    rr = 0
    num_query = 0
    for query in dataset.get_all_querys():
        if len(dataset.get_relevance_docids_by_query(query)) == 0: # for this query, ranking list is None
            continue
        got_rr = False
        for i in range(0, min(k, len(query_result_list[query]))):
            docid = query_result_list[query][i]
            relevance = dataset.get_relevance_label_by_query_and_docid(query, docid)
            if relevance in {1,2,3,4} and got_rr == False:
                rr += 1/(i+1)
                got_rr = True

        num_query += 1
    return rr / float(num_query)
