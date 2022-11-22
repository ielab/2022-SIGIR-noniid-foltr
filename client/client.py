from typing import NamedTuple
import numpy as np
import copy
from utils import evl_tool
from clicks.click_simulate import CcmClickModel
from data.LetorDataset import LetorDataset
from utils.dp import gamma_noise
from ranker.PDGDNeuralRanker import PDGDNeuralRanker

# The message that each client send to the server:
# 1.updated parameters from client
# 2.volume of data that client use for each update
ClientMessage = NamedTuple("ClientMessage",[("gradient", np.ndarray), ("parameters", np.ndarray), ("n_interactions", int)])

# Metric values (ndcg@k, mrr@k) of each client averaged on whole batch (computed by relevance label)
ClientMetric = NamedTuple("ClientMetric", [("mean_ndcg", float), ("mean_mrr", float), ("ndcg_list", list), ("mrr_list", list)])

class RankingClient_iid:
    """
    emulate clients
    """
    def __init__(self, dataset: LetorDataset, init_model, seed: int, click_model: CcmClickModel, sensitivity, epsilon, enable_noise, n_clients, client_id):
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
        self.query_set = dataset.get_all_querys()
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.enable_noise = enable_noise
        self.n_clients = n_clients
        self.client_id = client_id
        self.seed = seed

    def update_model(self, model) -> None:
        """
        Update the client-side model
        :param model: The new model
        :return: None
        """
        self.model = copy.deepcopy(model)

    def update_model_personal_layer(self, model) -> None:
        """
        Update the client-side model
        :param model: The new model
        :return: None
        """

        # only for neural model
        local_hidden_layers = self.model.hidden_layers
        local_biases = self.model.biases
        for idx in range(len(local_hidden_layers)):
            if idx != (len(local_hidden_layers)-1):
                local_hidden_layers[idx] = model.hidden_layers[idx]
                local_biases[idx] = model.biases[idx]
        self.model = copy.deepcopy(model)
        self.model.hidden_layers = local_hidden_layers
        self.model.biases = local_biases

    # evaluation metric: ndcg@k
    def eval_ranking_ndcg(self, ranking: np.ndarray, k = 10) -> float:
        dcg = 0.0
        idcg = 0.0
        rel_set = []
        rel_set = sorted(ranking.copy().tolist(), reverse=True)
        for i in range(0, min(k, ranking.shape[0])):
            r = ranking[i] # document (true) relevance label
            dcg += ((2 ** r - 1) / np.log2(i + 2))
            idcg += ((2 ** rel_set[i] - 1) / np.log2(i + 2))
        # deal with invalid value
        if idcg == 0.0:
            ndcg = 0.0
        else:
            ndcg = dcg/idcg

        return ndcg

    # evaluation metric: mrr@k
    def eval_ranking_mrr(self, ranking: np.ndarray, k = 10) -> float:
        rr = 0.0
        got_rr = False
        for i in range(0, min(k, ranking.shape[0])):
            r = ranking[i] # document (true) relevance label
            if r > 0 and got_rr == False: # TODO: decide the threshold value for relevance label
                rr = 1/(i+1)
                got_rr = True

        return rr

    def client_ranker_update(self, n_interactions: int, multi_update=True, fed_alg="fedavg", mu=0):
        """
        Run submits queries to a ranking model and gets its performance (eval metrics) and updates gradient / models
        :param n_interactions:
        :param ranker:
        :return:
        """
        per_interaction_client_ndcg = []
        per_interaction_client_mrr = []
        index = self.random_state.randint(self.query_set.shape[0], size=n_interactions+10) # randomly choose queries for simulation on each client (number of queries based o the set n_interactions)
        gradients = np.zeros(self.dataset._feature_size) # initialize gradient
        global_weights = self.model.get_current_weights() # used for fedprox

        n_success = 0
        for i in range(n_interactions+10): # run in batches

            if n_success == n_interactions:
                break

            id = index[i]
            qid = self.query_set[id]

            try:
                ranking_result, scores = self.model.get_query_result_list(self.dataset, qid)
                n_success += 1
            except Exception as e:
                print("exception in get_quey_result_list", str(e))
                continue

            # print("Type of result_list: ", type(ranking_result))   #Todo: should be np.ndarray, otherwise change function - [eval_ranking_mrr] and [eval_ranking_ndcg]
            ranking_relevance = np.zeros(ranking_result.shape[0])
            for i in range(0, ranking_result.shape[0]):
                docid = ranking_result[i]
                relevance = self.dataset.get_relevance_label_by_query_and_docid(qid, docid)
                ranking_relevance[i] = relevance
            # # compute online performance
            per_interaction_client_mrr.append(self.eval_ranking_mrr(ranking_relevance)) # using relevance label for evaluation
            # per_interaction_client_ndcg.append(self.eval_ranking_ndcg(ranking_relevance))# using relevance label for evaluation
            # another way to compute online ndcg
            online_ndcg = evl_tool.query_ndcg_at_k(self.dataset,ranking_result,qid,10)
            per_interaction_client_ndcg.append(online_ndcg)

            click_label = self.click_model(ranking_relevance, self.random_state)

            # g = self.model.update_to_clicks(click_label, ranking_result, scores, return_gradients=True)
            # g = self.model.update_to_clicks(click_label, ranking_result, scores,
            #                                 self.dataset.get_all_features_by_query(qid), return_gradients=True)

            current_weights = self.model.get_current_weights()  # used for fedprox
            if fed_alg == "fedprox": # update for fedprox
                if isinstance(self.model, PDGDNeuralRanker):
                    g = self.model.update_to_clicks(click_label, ranking_result, scores, return_gradients=True)
                    # regularization = mu * (current_weights - global_weights)
                    # g -= regularization

                    new_g = []
                    layer_num = len(g)
                    for idx in range(len(g)):
                        g_item_w = g[idx][0]
                        regu_w_item = mu * (current_weights[0][layer_num-idx-1] - global_weights[0][layer_num-idx-1])
                        g_item_w = g_item_w - regu_w_item
                        if idx != 0:
                            g_item_b = g[idx][1]
                            regu_b_item = mu * ( current_weights[1][layer_num - idx - 1] - global_weights[1][layer_num - idx - 1])
                            g_item_b = g_item_b - regu_b_item
                        else:
                            g_item_b = None
                        new_g.append((g_item_w, g_item_b))
                    g = new_g
                else:
                    g = self.model.update_to_clicks(click_label, ranking_result, scores, self.dataset.get_all_features_by_query(qid), return_gradients=True)
                    regularization = mu * (current_weights - global_weights)
                    g -= regularization
            else:
                if isinstance(self.model, PDGDNeuralRanker):
                    g = self.model.update_to_clicks(click_label, ranking_result, scores, return_gradients=True)
                else:
                    g = self.model.update_to_clicks(click_label, ranking_result, scores, self.dataset.get_all_features_by_query(qid), return_gradients=True)

            if multi_update:  # update in each interaction
                self.model.update_to_gradients(g)
            else: # accumulate gradients in batch (sum)
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

        return ClientMessage(gradient=gradients, parameters=updated_weights, n_interactions=n_interactions), ClientMetric(mean_ndcg=mean_client_ndcg, mean_mrr=mean_client_mrr, ndcg_list=per_interaction_client_ndcg, mrr_list=per_interaction_client_mrr)


    def get_client_feedback(self, n_interactions: int) -> ClientMessage:
        """

        :param n_interactions:
        :return:
        """

        pass



class RankingClient_iid_LRS:
    """
    emulate 'label preference skew' iid clients using query_click pair
    """
    def __init__(self, dataset: LetorDataset, init_model, seed: int, query_iid_group: list, query_click_pair: dict, sensitivity, epsilon, enable_noise, n_clients):
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
        self.query_click_pair = query_click_pair
        self.query_set = query_iid_group
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
    def eval_ranking_ndcg(self, ranking: np.ndarray, k = 10) -> float:
        dcg = 0.0
        idcg = 0.0
        rel_set = []
        rel_set = sorted(ranking.copy().tolist(), reverse=True)
        for i in range(0, min(k, ranking.shape[0])):
            r = ranking[i] # document (true) relevance label
            dcg += ((2 ** r - 1) / np.log2(i + 2))
            idcg += ((2 ** rel_set[i] - 1) / np.log2(i + 2))
        # deal with invalid value
        if idcg == 0.0:
            ndcg = 0.0
        else:
            ndcg = dcg/idcg

        return ndcg

    # evaluation metric: mrr@k
    def eval_ranking_mrr(self, ranking: np.ndarray, k = 10) -> float:
        rr = 0.0
        got_rr = False
        for i in range(0, min(k, ranking.shape[0])):
            r = ranking[i] # document (true) relevance label
            if r > 0 and got_rr == False: # TODO: decide the threshold value for relevance label
                rr = 1/(i+1)
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
        index = self.random_state.randint(len(self.query_set), size=n_interactions+10) # randomly choose queries for simulation on each client (number of queries based o the set n_interactions)
        gradients = np.zeros(self.dataset._feature_size) # initialize gradient

        n_success = 0
        for i in range(n_interactions+10): # run in batches

            if n_success == n_interactions:
                break

            id = index[i]
            qid = self.query_set[id]

            try:
                ranking_result, scores = self.model.get_query_result_list(self.dataset, qid)
                n_success += 1
            except Exception as e:
                print("exception in get_quey_result_list", str(e))
                continue

            # print("Type of result_list: ", type(ranking_result))   #Todo: should be np.ndarray, otherwise change function - [eval_ranking_mrr] and [eval_ranking_ndcg]
            ranking_relevance = np.zeros(ranking_result.shape[0])
            for i in range(0, ranking_result.shape[0]):
                docid = ranking_result[i]
                relevance = self.dataset.get_relevance_label_by_query_and_docid(qid, docid)
                ranking_relevance[i] = relevance
            # # compute online performance
            per_interaction_client_mrr.append(self.eval_ranking_mrr(ranking_relevance)) # using relevance label for evaluation
            # per_interaction_client_ndcg.append(self.eval_ranking_ndcg(ranking_relevance))# using relevance label for evaluation
            # another way to compute online ndcg
            online_ndcg = evl_tool.query_ndcg_at_k(self.dataset,ranking_result,qid,10)
            per_interaction_client_ndcg.append(online_ndcg)

            click_model = self.query_click_pair[qid]
            click_label = click_model(ranking_relevance, self.random_state)

            g = self.model.update_to_clicks(click_label, ranking_result, scores, self.dataset.get_all_features_by_query(qid), return_gradients=True)
            if multi_update:  # update in each interaction
                self.model.update_to_gradients(g)
            else: # accumulate gradients in batch (sum)
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

        return ClientMessage(gradient=gradients, parameters=updated_weights, n_interactions=n_interactions), ClientMetric(mean_ndcg=mean_client_ndcg, mean_mrr=mean_client_mrr, ndcg_list=per_interaction_client_ndcg, mrr_list=per_interaction_client_mrr)


    def get_client_feedback(self, n_interactions: int) -> ClientMessage:
        """

        :param n_interactions:
        :return:
        """

        pass



class RankingClient_iid_LRS_V2:
    """
    emulate 'label preference skew' iid clients using query_click pair
    """
    def __init__(self, dataset: LetorDataset, init_model, seed: int, click_model: list, sensitivity, epsilon, enable_noise, n_clients):
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
        self.query_set = dataset.get_all_querys()
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
    def eval_ranking_ndcg(self, ranking: np.ndarray, k = 10) -> float:
        dcg = 0.0
        idcg = 0.0
        rel_set = []
        rel_set = sorted(ranking.copy().tolist(), reverse=True)
        for i in range(0, min(k, ranking.shape[0])):
            r = ranking[i] # document (true) relevance label
            dcg += ((2 ** r - 1) / np.log2(i + 2))
            idcg += ((2 ** rel_set[i] - 1) / np.log2(i + 2))
        # deal with invalid value
        if idcg == 0.0:
            ndcg = 0.0
        else:
            ndcg = dcg/idcg

        return ndcg

    # evaluation metric: mrr@k
    def eval_ranking_mrr(self, ranking: np.ndarray, k = 10) -> float:
        rr = 0.0
        got_rr = False
        for i in range(0, min(k, ranking.shape[0])):
            r = ranking[i] # document (true) relevance label
            if r > 0 and got_rr == False: # TODO: decide the threshold value for relevance label
                rr = 1/(i+1)
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
        index = self.random_state.randint(len(self.query_set), size=n_interactions) # randomly choose queries for simulation on each client (number of queries based o the set n_interactions)
        gradients = np.zeros(self.dataset._feature_size) # initialize gradient
        for i in range(n_interactions): # run in batches
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
            per_interaction_client_mrr.append(self.eval_ranking_mrr(ranking_relevance)) # using relevance label for evaluation
            # per_interaction_client_ndcg.append(self.eval_ranking_ndcg(ranking_relevance))# using relevance label for evaluation
            # another way to compute online ndcg
            online_ndcg = evl_tool.query_ndcg_at_k(self.dataset,ranking_result,qid,10)
            per_interaction_client_ndcg.append(online_ndcg)

            click_id = self.random_state.randint(len(self.click_model), size=1)[0]
            click_model = self.click_model[click_id]
            click_label = click_model(ranking_relevance, self.random_state)

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
            else: # accumulate gradients in batch (sum)
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

        return ClientMessage(gradient=gradients, parameters=updated_weights, n_interactions=n_interactions), ClientMetric(mean_ndcg=mean_client_ndcg, mean_mrr=mean_client_mrr, ndcg_list=per_interaction_client_ndcg, mrr_list=per_interaction_client_mrr)


    def get_client_feedback(self, n_interactions: int) -> ClientMessage:
        """

        :param n_interactions:
        :return:
        """

        pass



class RankingClient_iid_intent_change:
    """
    emulate clients for IID data of intent-change data: merging 4 intents as a whole, each time randomly pick one intent
    """
    def __init__(self, dataset: list, init_model, seed: int, click_model: CcmClickModel, sensitivity, epsilon, enable_noise, n_clients, client_id):
        """
        :param dateset: representing a (query -> {document relevances, document features}) mapping
                for the queries the client can submit
        :param init_model: A ranking model
        :param click_model: A click model that emulate the user's behaviour; together with a click metric it is used
                to reflect the ranking quality
        :param metric: A metric instance, applied on the simulated click return the measured quality
        """
        self.dataset = dataset[0]
        self.dataset_list = dataset
        self.model = copy.deepcopy(init_model)
        self.random_state = np.random.RandomState(seed)
        self.click_model = click_model
        self.query_set = dataset[0].get_all_querys()
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.enable_noise = enable_noise
        self.n_clients = n_clients
        self.client_id = client_id
        self.seed = seed

    def update_model(self, model) -> None:
        """
        Update the client-side model
        :param model: The new model
        :return: None
        """
        self.model = copy.deepcopy(model)

    def update_model_personal_layer(self, model) -> None:
        """
        Update the client-side model
        :param model: The new model
        :return: None
        """

        # only for neural model
        local_hidden_layers = self.model.hidden_layers
        local_biases = self.model.biases
        for idx in range(len(local_hidden_layers)):
            if idx != (len(local_hidden_layers)-1):
                local_hidden_layers[idx] = model.hidden_layers[idx]
                local_biases[idx] = model.biases[idx]
        self.model = copy.deepcopy(model)
        self.model.hidden_layers = local_hidden_layers
        self.model.biases = local_biases

    # evaluation metric: ndcg@k
    def eval_ranking_ndcg(self, ranking: np.ndarray, k = 10) -> float:
        dcg = 0.0
        idcg = 0.0
        rel_set = []
        rel_set = sorted(ranking.copy().tolist(), reverse=True)
        for i in range(0, min(k, ranking.shape[0])):
            r = ranking[i] # document (true) relevance label
            dcg += ((2 ** r - 1) / np.log2(i + 2))
            idcg += ((2 ** rel_set[i] - 1) / np.log2(i + 2))
        # deal with invalid value
        if idcg == 0.0:
            ndcg = 0.0
        else:
            ndcg = dcg/idcg

        return ndcg

    # evaluation metric: mrr@k
    def eval_ranking_mrr(self, ranking: np.ndarray, k = 10) -> float:
        rr = 0.0
        got_rr = False
        for i in range(0, min(k, ranking.shape[0])):
            r = ranking[i] # document (true) relevance label
            if r > 0 and got_rr == False: # TODO: decide the threshold value for relevance label
                rr = 1/(i+1)
                got_rr = True

        return rr

    def client_ranker_update(self, n_interactions: int, multi_update=True, fed_alg="fedavg", mu=0):
        """
        Run submits queries to a ranking model and gets its performance (eval metrics) and updates gradient / models
        :param n_interactions:
        :param ranker:
        :return:
        """
        per_interaction_client_ndcg = []
        per_interaction_client_mrr = []
        index = self.random_state.randint(self.query_set.shape[0], size=n_interactions) # randomly choose queries for simulation on each client (number of queries based o the set n_interactions)
        index_intent = self.random_state.randint(len(self.dataset_list), size=n_interactions) # randomly choose intent for each query chosen
        gradients = np.zeros(self.dataset._feature_size) # initialize gradient
        global_weights = self.model.get_current_weights() # used for fedprox

        for i in range(n_interactions): # run in batches
            id = index[i]
            qid = self.query_set[id]
            # adding for randomly select data under different intent for IID distribution
            intent_type = index_intent[i]
            self.dataset = self.dataset_list[intent_type]

            ranking_result, scores = self.model.get_query_result_list(self.dataset, qid)
            # print("Type of result_list: ", type(ranking_result))   #Todo: should be np.ndarray, otherwise change function - [eval_ranking_mrr] and [eval_ranking_ndcg]
            ranking_relevance = np.zeros(ranking_result.shape[0])
            for i in range(0, ranking_result.shape[0]):
                docid = ranking_result[i]
                relevance = self.dataset.get_relevance_label_by_query_and_docid(qid, docid)
                ranking_relevance[i] = relevance
            # # compute online performance
            per_interaction_client_mrr.append(self.eval_ranking_mrr(ranking_relevance)) # using relevance label for evaluation
            # per_interaction_client_ndcg.append(self.eval_ranking_ndcg(ranking_relevance))# using relevance label for evaluation
            # another way to compute online ndcg
            online_ndcg = evl_tool.query_ndcg_at_k(self.dataset,ranking_result,qid,10)
            per_interaction_client_ndcg.append(online_ndcg)

            click_label = self.click_model(ranking_relevance, self.random_state)

            # g = self.model.update_to_clicks(click_label, ranking_result, scores, return_gradients=True)
            # g = self.model.update_to_clicks(click_label, ranking_result, scores,
            #                                 self.dataset.get_all_features_by_query(qid), return_gradients=True)

            current_weights = self.model.get_current_weights()  # used for fedprox
            if fed_alg == "fedprox": # update for fedprox
                if isinstance(self.model, PDGDNeuralRanker):
                    g = self.model.update_to_clicks(click_label, ranking_result, scores, return_gradients=True)
                    # regularization = mu * (current_weights - global_weights)
                    # g -= regularization

                    new_g = []
                    layer_num = len(g)
                    for idx in range(len(g)):
                        g_item_w = g[idx][0]
                        regu_w_item = mu * (current_weights[0][layer_num-idx-1] - global_weights[0][layer_num-idx-1])
                        g_item_w = g_item_w - regu_w_item
                        if idx != 0:
                            g_item_b = g[idx][1]
                            regu_b_item = mu * ( current_weights[1][layer_num - idx - 1] - global_weights[1][layer_num - idx - 1])
                            g_item_b = g_item_b - regu_b_item
                        else:
                            g_item_b = None
                        new_g.append((g_item_w, g_item_b))
                    g = new_g
                else:
                    g = self.model.update_to_clicks(click_label, ranking_result, scores, self.dataset.get_all_features_by_query(qid), return_gradients=True)
                    regularization = mu * (current_weights - global_weights)
                    g -= regularization
            else:
                if isinstance(self.model, PDGDNeuralRanker):
                    g = self.model.update_to_clicks(click_label, ranking_result, scores, return_gradients=True)
                else:
                    g = self.model.update_to_clicks(click_label, ranking_result, scores, self.dataset.get_all_features_by_query(qid), return_gradients=True)

            if multi_update:  # update in each interaction
                self.model.update_to_gradients(g)
            else: # accumulate gradients in batch (sum)
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

        return ClientMessage(gradient=gradients, parameters=updated_weights, n_interactions=n_interactions), ClientMetric(mean_ndcg=mean_client_ndcg, mean_mrr=mean_client_mrr, ndcg_list=per_interaction_client_ndcg, mrr_list=per_interaction_client_mrr)


    def get_client_feedback(self, n_interactions: int) -> ClientMessage:
        """

        :param n_interactions:
        :return:
        """

        pass
