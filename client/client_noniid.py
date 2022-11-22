from typing import NamedTuple
import numpy as np
import copy
from clicks.click_simulate import CcmClickModel
from clicks.click_simulate import PbmClickModel
from data.LetorDataset import LetorDataset
# from data.datasets import DataSplit
from utils.dp import gamma_noise
from utils import evl_tool
from ranker.PDGDNeuralRanker import PDGDNeuralRanker
from sklearn.cluster import KMeans

# The message that each client send to the server:
# 1.updated parameters from client
# 2.volume of data that client use for each update
ClientMessage = NamedTuple("ClientMessage",
                           [("gradient", np.ndarray), ("parameters", np.ndarray), ("n_interactions", int)])

# Metric values (ndcg@k, mrr@k) of each client averaged on whole batch (computed by relevance label)
ClientMetric = NamedTuple("ClientMetric",
                          [("mean_ndcg", float), ("mean_mrr", float), ("ndcg_list", list), ("mrr_list", list)])


class RankingClient_noniid:
    """
    emulate general non-iid clients
    """

    def __init__(self, dataset: LetorDataset, init_model, seed: int, click_model: CcmClickModel, sensitivity, epsilon,
                 enable_noise, n_clients, client_id):
        """
        :param dateset: representing a (query -> {document relevances, document features}) mapping
                for the queries the client can submit
        :param init_model: A ranking model
        :param click_model: A click model that emulate the user's behaviour; together with a click metric it is used
                to reflect the ranking quality
        :param metric: A metric instance, applied on the simulated click return the measured quality
        """
        self.dataset = dataset
        self.model = copy.deepcopy(init_model)
        self.random_state = np.random.RandomState(seed)
        self.click_model = click_model
        self.query_set = self.get_non_iid_query_group(dataset, n_clients, client_id)#self.get_non_iid_query_group(dataset, 6, client_id) #self.get_non_iid_query_group(dataset, 3, min(client_id,2))
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.enable_noise = enable_noise
        self.n_clients = n_clients

    def update_model(self, model) -> None:
        """
        Update the client-side model
        :param model: The new model
        :return: None
        """
        self.model = copy.deepcopy(model)

    # evaluation metric: ndcg@k
    def eval_ranking_ndcg(self, ranking: np.ndarray, k=10) -> float:
        dcg = 0.0
        idcg = 0.0
        rel_set = []
        rel_set = sorted(ranking.copy().tolist(), reverse=True)
        for i in range(0, min(k, ranking.shape[0])):
            r = ranking[i]  # document (true) relevance label
            dcg += ((2 ** r - 1) / np.log2(i + 2))
            idcg += ((2 ** rel_set[i] - 1) / np.log2(i + 2))
        # deal with invalid value
        if idcg == 0.0:
            ndcg = 0.0
        else:
            ndcg = dcg / idcg

        return ndcg

    # evaluation metric: mrr@k
    def eval_ranking_mrr(self, ranking: np.ndarray, k=10) -> float:
        rr = 0.0
        got_rr = False
        for i in range(0, min(k, ranking.shape[0])):
            r = ranking[i]  # document (true) relevance label
            if r > 0 and got_rr == False:  # TODO: decide the threshold value for relevance label
                rr = 1 / (i + 1)
                got_rr = True

        return rr

    def client_ranker_update(self, n_interactions: int, multi_update=True):
        """
        Run submits queries to a ranking model and gets its performance (eval metrics) and updates gradient / models
        :param n_interactions:
        :param ranker:
        :return:
        """
        per_interaction_client_ndcg = []
        per_interaction_client_mrr = []
        index = self.random_state.randint(self.query_set.shape[0],
                                          size=n_interactions)  # randomly choose queries for simulation on each client (number of queries based o the set n_interactions)
        gradients = np.zeros(self.dataset._feature_size)  # initialize gradient
        for i in range(n_interactions):  # run in batches
            id = index[i]
            qid = self.query_set[id]

            ranking_result, scores = self.model.get_query_result_list(self.dataset, qid)
            # print("Type of result_list: ", type(ranking_result))   #Todo: should be np.ndarray, otherwise change function - [eval_ranking_mrr] and [eval_ranking_ndcg]
            ranking_relevance = np.zeros(ranking_result.shape[0])
            for i in range(0, ranking_result.shape[0]):
                docid = ranking_result[i]
                relevance = self.dataset.get_relevance_label_by_query_and_docid(qid, docid)
                ranking_relevance[i] = relevance
            # # compute online performance
            per_interaction_client_mrr.append(
                self.eval_ranking_mrr(ranking_relevance))  # using relevance label for evaluation
            # per_interaction_client_ndcg.append(self.eval_ranking_ndcg(ranking_relevance))# using relevance label for evaluation
            # another way to compute online ndcg
            online_ndcg = evl_tool.query_ndcg_at_k(self.dataset, ranking_result, qid, 10)
            per_interaction_client_ndcg.append(online_ndcg)

            click_label = self.click_model(ranking_relevance, self.random_state)

            g = self.model.update_to_clicks(click_label, ranking_result, scores,
                                            self.dataset.get_all_features_by_query(qid), return_gradients=True)
            if multi_update:  # update in each interaction
                self.model.update_to_gradients(g)
            else:  # accumulate gradients in batch (sum)
                gradients += g

        # testset_result = ranker.get_all_query_reuslt_list(test_set)
        # ndcg = evl_tool.average_ndcg_at_k(test_set, testset_result, 10)  # off-line ndcg evaluation on test_set of each batch
        # ndcg_scores.append(ndcg)
        if not multi_update:
            self.model.update_to_gradients(gradients)

        updated_weights = self.model.get_current_weights()

        ## add noise
        if self.model.enable_noise:
            noise = gamma_noise(np.shape(updated_weights), self.sensitivity, self.epsilon, self.n_clients)

            updated_weights += noise

        mean_client_ndcg = np.mean(per_interaction_client_ndcg)
        mean_client_mrr = np.mean(per_interaction_client_mrr)

        return ClientMessage(gradient=gradients, parameters=updated_weights,
                             n_interactions=n_interactions), ClientMetric(mean_ndcg=mean_client_ndcg,
                                                                          mean_mrr=mean_client_mrr,
                                                                          ndcg_list=per_interaction_client_ndcg,
                                                                          mrr_list=per_interaction_client_mrr)

    def get_client_feedback(self, n_interactions: int) -> ClientMessage:
        """

        :param n_interactions:
        :return:
        """

        pass

    def get_non_iid_query_group(self, dataset: LetorDataset, n_clusters, client_id, seed=5):
        query_rep_avg = dict()
        query_features = dataset.get_query_get_all_features()  # dict

        ## for other normalised datasets
        # for key in query_features.keys():
        #     query_rep_avg[key] = sum(query_features[key]) / len(query_features[key]) # average of doc feature vectors
        # all_query = list(query_rep_avg.keys())

        ## for MSLR10K dataset
        for key in query_features.keys():
            if len(query_features[key]) > 1:
                query_rep_avg[key] = sum(query_features[key]) / len(query_features[key]) # average of doc feature vectors
        all_query = list(query_rep_avg.keys())

        features_for_kmeans = []
        for query in all_query:
            features_for_kmeans.append(query_rep_avg[query])

        kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(features_for_kmeans)
        cluster = kmeans.labels_
        position = list(np.where(cluster == client_id))[0]
        query_set = []
        for p in position:
            query_set.append(all_query[p])

        return np.array(query_set)


class RankingClient_noniid_LRS:
    """
    emulate 'label preference skew' non-iid clients using query_click pair
    """

    def __init__(self, dataset: LetorDataset, init_model, seed: int, click_model, sensitivity, epsilon,
                 enable_noise, n_clients, query_group: np.ndarray):
        """
        :param dateset: representing a (query -> {document relevances, document features}) mapping
                for the queries the client can submit
        :param init_model: A ranking model
        :param click_model: A click model that emulate the user's behaviour; together with a click metric it is used
                to reflect the ranking quality
        :param metric: A metric instance, applied on the simulated click return the measured quality
        """
        self.dataset = dataset
        self.model = copy.deepcopy(init_model)
        self.random_state = np.random.RandomState(seed)
        # self.click_model = click_model_list[client_id]
        self.click_model = click_model
        self.query_set = query_group
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.enable_noise = enable_noise
        self.n_clients = n_clients

    def update_model(self, model) -> None:
        """
        Update the client-side model
        :param model: The new model
        :return: None
        """
        self.model = copy.deepcopy(model)

    # evaluation metric: ndcg@k
    def eval_ranking_ndcg(self, ranking: np.ndarray, k=10) -> float:
        dcg = 0.0
        idcg = 0.0
        rel_set = []
        rel_set = sorted(ranking.copy().tolist(), reverse=True)
        for i in range(0, min(k, ranking.shape[0])):
            r = ranking[i]  # document (true) relevance label
            dcg += ((2 ** r - 1) / np.log2(i + 2))
            idcg += ((2 ** rel_set[i] - 1) / np.log2(i + 2))
        # deal with invalid value
        if idcg == 0.0:
            ndcg = 0.0
        else:
            ndcg = dcg / idcg

        return ndcg

    # evaluation metric: mrr@k
    def eval_ranking_mrr(self, ranking: np.ndarray, k=10) -> float:
        rr = 0.0
        got_rr = False
        for i in range(0, min(k, ranking.shape[0])):
            r = ranking[i]  # document (true) relevance label
            if r > 0 and got_rr == False:  # TODO: decide the threshold value for relevance label
                rr = 1 / (i + 1)
                got_rr = True

        return rr

    def client_ranker_update(self, n_interactions: int, multi_update=True):
        """
        Run submits queries to a ranking model and gets its performance (eval metrics) and updates gradient / models
        :param n_interactions:
        :param ranker:
        :return:
        """
        per_interaction_client_ndcg = []
        per_interaction_client_mrr = []
        index = self.random_state.randint(self.query_set.shape[0],
                                          size=n_interactions)  # randomly choose queries for simulation on each client (number of queries based o the set n_interactions)
        gradients = np.zeros(self.dataset._feature_size)  # initialize gradient
        for i in range(n_interactions):  # run in batches
            id = index[i]
            qid = self.query_set[id]

            ranking_result, scores = self.model.get_query_result_list(self.dataset, qid)
            # print("Type of result_list: ", type(ranking_result))   #Todo: should be np.ndarray, otherwise change function - [eval_ranking_mrr] and [eval_ranking_ndcg]
            ranking_relevance = np.zeros(ranking_result.shape[0])
            for i in range(0, ranking_result.shape[0]):
                docid = ranking_result[i]
                relevance = self.dataset.get_relevance_label_by_query_and_docid(qid, docid)
                ranking_relevance[i] = relevance
            # # compute online performance
            per_interaction_client_mrr.append(
                self.eval_ranking_mrr(ranking_relevance))  # using relevance label for evaluation
            # per_interaction_client_ndcg.append(self.eval_ranking_ndcg(ranking_relevance))# using relevance label for evaluation
            # another way to compute online ndcg
            online_ndcg = evl_tool.query_ndcg_at_k(self.dataset, ranking_result, qid, 10)
            per_interaction_client_ndcg.append(online_ndcg)

            click_label = self.click_model(ranking_relevance, self.random_state)

            ## linear ranker
            # g = self.model.update_to_clicks(click_label, ranking_result, scores, self.dataset.get_all_features_by_query(qid), return_gradients=True)
            ## neural ranker
            # g = self.model.update_to_clicks(click_label, ranking_result, scores, return_gradients=True)
            if isinstance(self.model, PDGDNeuralRanker):
                g = self.model.update_to_clicks(click_label, ranking_result, scores, return_gradients=True)
            else:
                g = self.model.update_to_clicks(click_label, ranking_result, scores, self.dataset.get_all_features_by_query(qid), return_gradients=True)

            if multi_update:  # update in each interaction
                self.model.update_to_gradients(g)
            else:  # accumulate gradients in batch (sum)
                gradients += g

        # testset_result = ranker.get_all_query_reuslt_list(test_set)
        # ndcg = evl_tool.average_ndcg_at_k(test_set, testset_result, 10)  # off-line ndcg evaluation on test_set of each batch
        # ndcg_scores.append(ndcg)
        if not multi_update:
            self.model.update_to_gradients(gradients)

        updated_weights = self.model.get_current_weights()

        ## add noise
        if self.model.enable_noise:
            noise = gamma_noise(np.shape(updated_weights), self.sensitivity, self.epsilon, self.n_clients)

            updated_weights += noise

        mean_client_ndcg = np.mean(per_interaction_client_ndcg)
        mean_client_mrr = np.mean(per_interaction_client_mrr)

        return ClientMessage(gradient=gradients, parameters=updated_weights,
                             n_interactions=n_interactions), ClientMetric(mean_ndcg=mean_client_ndcg,
                                                                          mean_mrr=mean_client_mrr,
                                                                          ndcg_list=per_interaction_client_ndcg,
                                                                          mrr_list=per_interaction_client_mrr)

    def get_client_feedback(self, n_interactions: int) -> ClientMessage:
        """

        :param n_interactions:
        :return:
        """

        pass


class RankingClient_noniid_LRS_pair:
    """
    emulate 'label preference skew' non-iid clients using query_click pair
    """

    def __init__(self, dataset: LetorDataset, init_model, seed: int, click_model, sensitivity, epsilon,
                 enable_noise, n_clients, query_group: np.ndarray):
        """
        :param dateset: representing a (query -> {document relevances, document features}) mapping
                for the queries the client can submit
        :param init_model: A ranking model
        :param click_model: A click model that emulate the user's behaviour; together with a click metric it is used
                to reflect the ranking quality
        :param metric: A metric instance, applied on the simulated click return the measured quality
        """
        self.dataset = dataset
        self.model = copy.deepcopy(init_model)
        self.random_state = np.random.RandomState(seed)
        # self.click_model = click_model_list[client_id]
        self.click_model = click_model
        self.query_set = query_group
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.enable_noise = enable_noise
        self.n_clients = n_clients

    def update_model(self, model) -> None:
        """
        Update the client-side model
        :param model: The new model
        :return: None
        """
        self.model = copy.deepcopy(model)

    # evaluation metric: ndcg@k
    def eval_ranking_ndcg(self, ranking: np.ndarray, k=10) -> float:
        dcg = 0.0
        idcg = 0.0
        rel_set = []
        rel_set = sorted(ranking.copy().tolist(), reverse=True)
        for i in range(0, min(k, ranking.shape[0])):
            r = ranking[i]  # document (true) relevance label
            dcg += ((2 ** r - 1) / np.log2(i + 2))
            idcg += ((2 ** rel_set[i] - 1) / np.log2(i + 2))
        # deal with invalid value
        if idcg == 0.0:
            ndcg = 0.0
        else:
            ndcg = dcg / idcg

        return ndcg

    # evaluation metric: mrr@k
    def eval_ranking_mrr(self, ranking: np.ndarray, k=10) -> float:
        rr = 0.0
        got_rr = False
        for i in range(0, min(k, ranking.shape[0])):
            r = ranking[i]  # document (true) relevance label
            if r > 0 and got_rr == False:  # TODO: decide the threshold value for relevance label
                rr = 1 / (i + 1)
                got_rr = True

        return rr

    def client_ranker_update(self, n_interactions: int, global_step: int, multi_update=True):
        """
        Run submits queries to a ranking model and gets its performance (eval metrics) and updates gradient / models
        :param n_interactions:
        :param ranker:
        :return:
        """
        per_interaction_client_ndcg = []
        per_interaction_client_mrr = []
        index = self.random_state.randint(self.query_set.shape[0],
                                          size=n_interactions)  # randomly choose queries for simulation on each client (number of queries based o the set n_interactions)
        gradients = np.zeros(self.dataset._feature_size)  # initialize gradient
        for i in range(n_interactions):  # run in batches
            id = index[i]
            qid = self.query_set[id]

            ranking_result, scores = self.model.get_query_result_list(self.dataset, qid)
            # print("Type of result_list: ", type(ranking_result))   #Todo: should be np.ndarray, otherwise change function - [eval_ranking_mrr] and [eval_ranking_ndcg]
            ranking_relevance = np.zeros(ranking_result.shape[0])
            for i in range(0, ranking_result.shape[0]):
                docid = ranking_result[i]
                relevance = self.dataset.get_relevance_label_by_query_and_docid(qid, docid)
                ranking_relevance[i] = relevance
            # # compute online performance
            per_interaction_client_mrr.append(
                self.eval_ranking_mrr(ranking_relevance))  # using relevance label for evaluation
            # per_interaction_client_ndcg.append(self.eval_ranking_ndcg(ranking_relevance))# using relevance label for evaluation
            # another way to compute online ndcg
            online_ndcg = evl_tool.query_ndcg_at_k(self.dataset, ranking_result, qid, 10)
            per_interaction_client_ndcg.append(online_ndcg)

            click_label = self.click_model(ranking_relevance, self.random_state)

            g = self.model.update_to_clicks(click_label, ranking_result, scores,
                                            self.dataset.get_all_features_by_query(qid), return_gradients=True)
            if multi_update:  # update in each interaction
                self.model.update_to_gradients(g)
            else:  # accumulate gradients in batch (sum)
                gradients += g

        # testset_result = ranker.get_all_query_reuslt_list(test_set)
        # ndcg = evl_tool.average_ndcg_at_k(test_set, testset_result, 10)  # off-line ndcg evaluation on test_set of each batch
        # ndcg_scores.append(ndcg)
        if not multi_update:
            self.model.update_to_gradients(gradients)

        updated_weights = self.model.get_current_weights()

        ## add noise
        if self.model.enable_noise:
            noise = gamma_noise(np.shape(updated_weights), self.sensitivity, self.epsilon, self.n_clients)

            updated_weights += noise

        mean_client_ndcg = np.mean(per_interaction_client_ndcg)
        mean_client_mrr = np.mean(per_interaction_client_mrr)

        return ClientMessage(gradient=gradients, parameters=updated_weights,
                             n_interactions=n_interactions), ClientMetric(mean_ndcg=mean_client_ndcg,
                                                                          mean_mrr=mean_client_mrr,
                                                                          ndcg_list=per_interaction_client_ndcg,
                                                                          mrr_list=per_interaction_client_mrr)

    def get_client_feedback(self, n_interactions: int) -> ClientMessage:
        """

        :param n_interactions:
        :return:
        """

        pass
