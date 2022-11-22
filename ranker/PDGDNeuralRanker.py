import numpy as np
import copy


class PDGDNeuralRanker:
    def __init__(self, num_features, learning_rate, hidden_layers: list, tau=1, learning_rate_decay=1, random_initial=True):
        def normal(init, shape):
            return np.random.normal(0., init, shape)

        self.learning_rate = learning_rate
        self.hidden_layer_nodes = hidden_layers
        self.hidden_layers = []
        self.biases = []
        prev_units = num_features
        for n_units in hidden_layers:
            init = 1. / prev_units
            self.hidden_layers.append(normal(init, (prev_units, n_units)))
            self.biases.append(normal(init, n_units)[None, :])
            prev_units = n_units
        self.hidden_layers.append(normal(1. / prev_units, (prev_units, 1)))
        self.learning_rate_decay = learning_rate_decay

        self.tau = tau

        self.sensitivity = 0
        self.enable_noise = False

    def get_scores(self, features):
        prev_layer = features
        self.input = features
        self.activations = [prev_layer]
        for hidden_layer, bias in zip(self.hidden_layers[:-1], self.biases):
            prev_layer = np.dot(prev_layer, hidden_layer)
            prev_layer += bias
            prev_layer = 1. / (1. + np.exp(-prev_layer))
            self.activations.append(prev_layer)
        result = np.dot(prev_layer, self.hidden_layers[-1])
        self.activations.append(result)
        return result[:, 0]

    def backpropagate(self, doc_ind, doc_weights):
        activations = [a[doc_ind, :] for a in self.activations]
        doc_weights = np.expand_dims(doc_weights, axis=1)
        cur_der = (np.dot(activations[-2].T, doc_weights), None)
        derivatives = [cur_der]
        prev_der = doc_weights
        for i in range(len(self.hidden_layers) - 1):
            prev_der = np.dot(prev_der, self.hidden_layers[-i - 1].T)
            prev_der *= activations[-i - 2] * (1. - activations[-i - 2])

            w_der = np.dot(activations[-i - 3].T, prev_der)
            b_der = np.sum(prev_der, axis=0, keepdims=True)

            derivatives.append((w_der, b_der))

        return derivatives

    def get_query_result_list(self, dataset, query, random=False):
        feature_matrix = dataset.get_all_features_by_query(query)
        docid_list = np.array(dataset.get_candidate_docids_by_query(query))
        n_docs = docid_list.shape[0]

        k = np.minimum(10, n_docs)

        doc_scores = self.get_scores(feature_matrix)

        doc_scores += 18 - np.amax(doc_scores)

        ranking = self._recursive_choice(np.copy(doc_scores),
                                         np.array([], dtype=np.int32),
                                         k,
                                         random)
        return ranking, doc_scores

    def get_all_query_result_list(self, dataset):
        query_result_list = {}

        for query in dataset.get_all_querys():
            docid_list = np.array(dataset.get_candidate_docids_by_query(query))
            docid_list = docid_list.reshape((len(docid_list), 1))
            feature_matrix = dataset.get_all_features_by_query(query)
            score_list = self.get_scores(feature_matrix)

            docid_score_list = np.column_stack((docid_list, score_list))
            docid_score_list = np.flip(docid_score_list[docid_score_list[:, 1].argsort()], 0)

            query_result_list[query] = docid_score_list[:, 0]

        return query_result_list

    def _recursive_choice(self, scores, incomplete_ranking, k_left, random):
        n_docs = scores.shape[0]

        scores[incomplete_ranking] = np.amin(scores)

        scores += 18 - np.amax(scores)
        exp_scores = np.exp(scores/self.tau)

        exp_scores[incomplete_ranking] = 0
        probs = exp_scores / np.sum(exp_scores)

        safe_n = np.sum(probs > 10 ** (-4) / n_docs)

        safe_k = np.minimum(safe_n, k_left)
        if random:
            next_ranking = np.random.choice(np.arange(n_docs),
                                            replace=False,
                                            size=safe_k)
        else:
            next_ranking = np.random.choice(np.arange(n_docs),
                                            replace=False,
                                            p=probs,
                                            size=safe_k)

        ranking = np.concatenate((incomplete_ranking, next_ranking))
        k_left = k_left - safe_k

        if k_left > 0:
            return self._recursive_choice(scores, ranking, k_left, random)
        else:
            return ranking

    def get_current_weights(self):

        return copy.deepcopy(self.hidden_layers), copy.deepcopy(self.biases)

    def update_to_clicks(self, click_label, ranking, doc_scores, last_exam=None, return_gradients=False):

        if last_exam is None:

            clicks = np.array(click_label == 1)

            n_docs = ranking.shape[0]
            n_results = 10
            cur_k = np.minimum(n_docs, n_results)

            included = np.ones(cur_k, dtype=np.int32)

            if not clicks[-1]:
                included[1:] = np.cumsum(clicks[::-1])[:0:-1]

            neg_ind = np.where(np.logical_xor(clicks, included))[0]
            pos_ind = np.where(clicks)[0]

        else:
            if last_exam == 10:
                neg_ind = np.where(click_label[:last_exam] == 0)[0]
                pos_ind = np.where(click_label[:last_exam] == 1)[0]
            else:
                neg_ind = np.where(click_label[:last_exam + 1] == 0)[0]
                pos_ind = np.where(click_label[:last_exam] == 1)[0]


        n_pos = pos_ind.shape[0]
        n_neg = neg_ind.shape[0]
        n_pairs = n_pos * n_neg

        if n_pairs == 0:
            derivation = [(np.zeros_like(self.hidden_layers[-1]), None)]
            for idx in range(len(self.hidden_layers)-1):
                derivation.append((np.zeros_like(self.hidden_layers[-idx-2]), np.zeros_like(self.biases[-idx-1])))
            return derivation

        pos_r_ind = ranking[pos_ind]
        neg_r_ind = ranking[neg_ind]

        pos_scores = doc_scores[pos_r_ind]
        neg_scores = doc_scores[neg_r_ind]

        log_pair_pos = np.tile(pos_scores, n_neg)
        log_pair_neg = np.repeat(neg_scores, n_pos)

        pair_trans = 18 - np.maximum(log_pair_pos, log_pair_neg)
        exp_pair_pos = np.exp(log_pair_pos + pair_trans)
        exp_pair_neg = np.exp(log_pair_neg + pair_trans)

        pair_denom = (exp_pair_pos + exp_pair_neg)
        pair_w = np.maximum(exp_pair_pos, exp_pair_neg)
        pair_w /= pair_denom
        pair_w /= pair_denom
        pair_w *= np.minimum(exp_pair_pos, exp_pair_neg)

        pair_w *= self._calculate_unbias_weights(pos_ind, neg_ind, doc_scores, ranking)

        reshaped = np.reshape(pair_w, (n_neg, n_pos))
        pos_w = np.sum(reshaped, axis=0)
        neg_w = -np.sum(reshaped, axis=1)

        all_w = np.concatenate([pos_w, neg_w])
        all_ind = np.concatenate([pos_r_ind, neg_r_ind])

        if return_gradients:
            return self._get_update_gradients(all_ind, all_w)
        else:
            self._update_to_documents(all_ind, all_w)

    def federated_averaging_weights(self, feedbacks):
        assert len(feedbacks) > 0
        feedbacks = [(m.gradient, m.parameters, m.n_interactions) for m in feedbacks]
        total_interactions = 0

        tmp_parameters = feedbacks[0][1]
        hidden_layers = [np.zeros_like(p) for p in tmp_parameters[0]]
        biases = [np.zeros_like(p) for p in tmp_parameters[1]]
        for feedback in feedbacks:
            client_interactions = feedback[2]
            client_weights = feedback[1]
            for i in range(len(hidden_layers)):
                hidden_layers[i] = hidden_layers[i] + client_weights[0][i] * client_interactions
            for i in range(len(biases)):
                biases[i] = biases[i] + client_weights[1][i] * client_interactions
            total_interactions += client_interactions

        for i in range(len(hidden_layers)):
            hidden_layers[i] = hidden_layers[i] / total_interactions
        for i in range(len(biases)):
            biases[i] = biases[i] / total_interactions

        self.hidden_layers = hidden_layers
        self.biases = biases


    def _update_to_documents(self, doc_ind, doc_weights):
        derivatives = self.backpropagate(doc_ind, doc_weights)

        first_wd = derivatives[0][0]
        self.hidden_layers[-1] += first_wd * self.learning_rate
        for i, (wd, bd) in enumerate(derivatives[1:], 2):
            self.hidden_layers[-i] += wd * self.learning_rate
            self.biases[-i + 1] += bd * self.learning_rate
        self.learning_rate *= self.learning_rate_decay

        # clipping
        if self.enable_noise:
            square_sum = 0
            w_num = 0
            for i in range(len(self.hidden_layers)):
                w = self.hidden_layers[i]
                b = self.biases[i]
                square_sum += np.sum(np.power(w, 2)) + np.power(b, 2)
                w_num += (np.sum(np.ones_like(w)) + 1)
            norm = np.sqrt(square_sum / w_num)
            scale = np.minimum(1, self.sensitivity / norm)
            new_hidden_layers = []
            new_biases = []
            for i in range(len(self.hidden_layers)):
                new_hidden_layers.append(self.hidden_layers[i]*scale)
                new_biases.append(self.biases[i] * scale)
            self.hidden_layers = new_hidden_layers
            self.biases = new_biases

    def _get_update_gradients(self, doc_ind, doc_weights):
        derivatives = self.backpropagate(doc_ind, doc_weights)
        return derivatives

    def update_to_gradients(self, gradients):
        derivatives = gradients

        g_weights = [None] * len(derivatives)
        g_biases = [None] * len(derivatives)

        first_wd = derivatives[0][0]
        self.hidden_layers[-1] += first_wd * self.learning_rate
        g_weights[-1] = first_wd * self.learning_rate
        for i, (wd, bd) in enumerate(derivatives[1:], 2):
            self.hidden_layers[-i] += wd * self.learning_rate
            self.biases[-i + 1] += bd * self.learning_rate
            g_weights[-i] = wd * self.learning_rate
            g_biases[-i + 1] = bd * self.learning_rate
        self.learning_rate *= self.learning_rate_decay


    def _calculate_unbias_weights(self, pos_ind, neg_ind, doc_scores, ranking):
        ranking_prob = self._calculate_observed_prob(pos_ind, neg_ind,
                                                     doc_scores, ranking)
        flipped_prob = self._calculate_flipped_prob(pos_ind, neg_ind,
                                                    doc_scores, ranking)
        return flipped_prob / (ranking_prob + flipped_prob)

    def _calculate_flipped_prob(self, pos_ind, neg_ind, doc_scores, ranking):
        n_pos = pos_ind.shape[0]
        n_neg = neg_ind.shape[0]
        n_pairs = n_pos * n_neg
        n_results = ranking.shape[0]
        n_docs = doc_scores.shape[0]

        results_i = np.arange(n_results)
        pair_i = np.arange(n_pairs)
        doc_i = np.arange(n_docs)

        pos_pair_i = np.tile(pos_ind, n_neg)
        neg_pair_i = np.repeat(neg_ind, n_pos)

        flipped_rankings = np.tile(ranking[None, :],
                                   [n_pairs, 1])
        flipped_rankings[pair_i, pos_pair_i] = ranking[neg_pair_i]
        flipped_rankings[pair_i, neg_pair_i] = ranking[pos_pair_i]

        min_pair_i = np.minimum(pos_pair_i, neg_pair_i)
        max_pair_i = np.maximum(pos_pair_i, neg_pair_i)
        range_mask = np.logical_and(min_pair_i[:, None] <= results_i,
                                    max_pair_i[:, None] >= results_i)

        flipped_log = doc_scores[flipped_rankings]

        safe_log = np.tile(doc_scores[None, None, :],
                           [n_pairs, n_results, 1])

        results_ij = np.tile(results_i[None, 1:], [n_pairs, 1])
        pair_ij = np.tile(pair_i[:, None], [1, n_results - 1])
        mask = np.zeros((n_pairs, n_results, n_docs))
        mask[pair_ij, results_ij, flipped_rankings[:, :-1]] = True
        mask = np.cumsum(mask, axis=1).astype(bool)

        safe_log[mask] = np.amin(safe_log)
        safe_max = np.amax(safe_log, axis=2)
        safe_log -= safe_max[:, :, None] - 18
        flipped_log -= safe_max - 18
        flipped_exp = np.exp(flipped_log)

        safe_exp = np.exp(safe_log)
        safe_exp[mask] = 0
        safe_denom = np.sum(safe_exp, axis=2)
        safe_prob = np.ones((n_pairs, n_results))
        safe_prob[range_mask] = (flipped_exp / safe_denom)[range_mask]

        safe_pair_prob = np.prod(safe_prob, axis=1)

        return safe_pair_prob

    def _calculate_observed_prob(self, pos_ind, neg_ind, doc_scores, ranking):
        n_pos = pos_ind.shape[0]
        n_neg = neg_ind.shape[0]
        n_pairs = n_pos * n_neg
        n_results = ranking.shape[0]
        n_docs = doc_scores.shape[0]

        results_i = np.arange(n_results)
        # pair_i = np.arange(n_pairs)
        # doc_i = np.arange(n_docs)

        pos_pair_i = np.tile(pos_ind, n_neg)
        neg_pair_i = np.repeat(neg_ind, n_pos)

        min_pair_i = np.minimum(pos_pair_i, neg_pair_i)
        max_pair_i = np.maximum(pos_pair_i, neg_pair_i)
        range_mask = np.logical_and(min_pair_i[:, None] <= results_i,
                                    max_pair_i[:, None] >= results_i)

        safe_log = np.tile(doc_scores[None, :],
                           [n_results, 1])

        mask = np.zeros((n_results, n_docs))
        mask[results_i[1:], ranking[:-1]] = True
        mask = np.cumsum(mask, axis=0).astype(bool)

        safe_log[mask] = np.amin(safe_log)
        safe_max = np.amax(safe_log, axis=1)
        safe_log -= safe_max[:, None] - 18
        safe_exp = np.exp(safe_log)
        safe_exp[mask] = 0

        ranking_log = doc_scores[ranking] - safe_max + 18
        ranking_exp = np.exp(ranking_log)

        safe_denom = np.sum(safe_exp, axis=1)
        ranking_prob = ranking_exp / safe_denom

        tiled_prob = np.tile(ranking_prob[None, :], [n_pairs, 1])

        safe_prob = np.ones((n_pairs, n_results))
        safe_prob[range_mask] = tiled_prob[range_mask]

        safe_pair_prob = np.prod(safe_prob, axis=1)

        return safe_pair_prob

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_tau(self, tau):
        self.tau = tau