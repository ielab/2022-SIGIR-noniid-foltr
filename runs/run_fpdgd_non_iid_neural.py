# code for exploring the influence of non-iid data (without differential privacy noise here)
# for neural ranker

import time
import itertools
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

from data.LetorDataset import LetorDataset
from clicks.click_simulate import CcmClickModel
from clicks.click_simulate import PbmClickModel
from client.federated_optimize import train_uniform
from ranker.PDGDNeuralRanker import PDGDNeuralRanker
import os


def do_task(task):
    """
    Single task
    :param task:
    :return: result for single task
    """
    fold_id, click_model, e_s, ranker_id = task

    epsilon = e_s[0]
    sensitivity = e_s[1]
    params = common_params.copy()
    neural_ranker = PDGDNeuralRanker(n_features, Learning_rate, hidden_layers=[neural_layer])
    if enable_noise:
        neural_ranker.enable_noise_and_set_sensitivity(enable_noise, sensitivity)
    params.update(
        dict(click_model=click_model,
             sensitivity=sensitivity,
             epsilon=epsilon,
             ranker_generator=neural_ranker,
             enable_noise=enable_noise
             ))

    if non_iid_type == "intent_change":
        trainset = LetorDataset("{}/clueweb09_intent_change.txt".format(params['dataset_path']),
                                params['n_features'], query_level_norm=data_norm,
                                cache_root="../datasets/cache", abs_path=False)
        testset = LetorDataset("{}/clueweb09_intent_change.txt".format(params['dataset_path']),
                               params['n_features'], query_level_norm=data_norm,
                               cache_root="../datasets/cache", abs_path=False)
    elif non_iid_type == "label_dist_skew" and is_iid == False:
        trainset = []
        for label in range(5):
            train = LetorDataset("{}/LDS/Fold{}/label_{}/train.txt".format(params['dataset_path'], fold_id + 1, label),
                                    params['n_features'], query_level_norm=data_norm,
                                    cache_root="../datasets/cache",
                                    abs_path=False)
            trainset.append(train)
        testset = LetorDataset("{}/Fold{}/test.txt".format(params['dataset_path'], fold_id + 1),
                               params['n_features'], query_level_norm=data_norm,
                               cache_root="../datasets/cache",
                               abs_path=False)
    else:
        trainset = LetorDataset("{}/Fold{}/train.txt".format(params['dataset_path'], fold_id + 1),
                                params['n_features'], query_level_norm=data_norm,
                                cache_root="../datasets/cache", abs_path=False)
        testset = LetorDataset("{}/Fold{}/test.txt".format(params['dataset_path'], fold_id + 1),
                               params['n_features'], query_level_norm=data_norm,
                               cache_root="../datasets/cache", abs_path=False)

    task_info = "click_model:iid {} folder:{}".format(click_model.name, fold_id + 1) if is_iid else "click_model:non_iid {} folder:{}".format(click_model.name, fold_id + 1)

    click_type = click_model.name
    save_path = output_path + "fold_{}/{}/sensitivity_{}/epsilon_{}/result.npy".format(fold_id, click_type, sensitivity, epsilon)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    train_result = train_uniform(params=params, traindata=trainset, testdata=testset, message=task_info, num_update=num_update, is_iid=is_iid, non_iid_type=non_iid_type, save_path=save_path)
    return train_result


def run(path, tasks):
    tasks = list(tasks)
    print("num tasks:", len(tasks))
    # multi-processing
    n_cpu = min(5, len(tasks))
    print("num cpus:", n_cpu)
    with Pool(n_cpu) as p:
        results = p.map(do_task, tasks)

    for task, result in tqdm(zip(tasks, results)):
        fold_id, click_model, e_s, ranker_id = task
        epsilon = e_s[0]
        sensitivity = e_s[1]
        click_type = click_model.name
        save_path = path + "fold_{}/{}/sensitivity_{}/epsilon_{}/result.npy".format(fold_id, click_type, sensitivity, epsilon)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        np.save(save_path, result)


if __name__ == "__main__":
    # not recommend to change here
    update = True # client_multi_update

    ## non-iid experiment
    is_iid = False # set True if you want run iid baseline, set False to run non-iid baseline
    non_iid_type = "label_dist_skew" # ["intent_change", "label_dist_skew"]
    is_mixed = False # set True if you want to pair this non-iid type with "data_quan_skew"
    # solutions
    is_personal_layer = False # False # True for personalization layer
    fed_alg = "fedprox" # ["fedavg", "fedprox"]
    # # mu = 0.001 # [0 (equal to fedavg), 0.001, 0.01, 0.1, 1]
    # mu_list = [0, 0.00001, 0.00001, 0.001, 0.01, 0.1, 1, 10]
    mu_list = [0.1]

    ## iid experiment
    # is_iid = True # set 'True' if you want run iid baseline, set 'False' to run non-iid baseline
    # non_iid_type = "label_dist_skew" # ["intent_change", None, "label_dist_skew"]
    # is_mixed = False # set True if you want to pair this non-iid type with "data_quan_skew"
    # solutions
    # is_personal_layer = False
    # fed_alg = "fedavg" # ["fedavg", "fedprox"]
    # mu = 0 # [0 (equal to fedavg), 0.001, 0.01, 0.1, 1]
    # mu_list = [0, 0.001, 0.01, 0.1, 1]
    # mu_list = [0]

    ## without dp noise
    enable_noise = False  # set True if you want to add DP noise, otherwise set False
    e_s_list = [(0, 0)]

    # experiment parameters
    datasets = ["MSLR10K"]  # ["MQ2007", "MSLR10K", "Yahoo", "Istella-s", "intent-change"]
    experiment = "FPDGD-non iid"
    n_clients = 5
    batch_size = 5
    num_update = 10000
    # interactions_budget = n_clients * batch_size * num_update
    interactions_budget = num_update
    Learning_rate = 0.1
    click_model_type = "CCM"# "CCM" "PBM"
    neural_layer = 64

    # dataset and result path
    dataset_root_dir = "../datasets"
    output_root_dir = "../results"
    intent_path = "../datasets/intent-change/intents"

    for seed in tqdm(range(1, 2)):  # range(1,21)
        for mu in mu_list:
            for dataset in datasets:
                if dataset == "MQ2007":
                    n_folds = 5
                    n_features = 46
                    data_norm = False

                elif dataset == "MQ2008":
                    n_folds = 5
                    n_features = 46
                    data_norm = False

                elif dataset == "MSLR10K":
                    n_folds = 5
                    n_features = 136
                    data_norm = True

                elif dataset == "Yahoo":
                    n_folds = 1
                    n_features = 700
                    data_norm = True

                elif dataset == "Istella-s":
                    n_folds = 1
                    n_features = 220
                    data_norm = True

                elif dataset == "intent-change":
                    n_folds = 1
                    n_features = 105
                    data_norm = True

                dataset_path = f"{dataset_root_dir}/{dataset}"

                if is_iid == True and fed_alg == "fedavg":
                    output_path = f"{output_root_dir}/{experiment}/{dataset}/{dataset}_FPDGD_clients{n_clients}_batch{batch_size}_updates{num_update}_iid_{non_iid_type}_{click_model_type}_neural{neural_layer}/run_{seed}/"
                elif is_iid == True and fed_alg == "fedprox":
                    output_path = f"{output_root_dir}/{experiment}/{dataset}/{dataset}_FPDGD_clients{n_clients}_batch{batch_size}_updates{num_update}_iid_{non_iid_type}_{click_model_type}_neural{neural_layer}_fedprox_mu{mu}/run_{seed}/"
                elif is_iid == False and is_personal_layer == True:
                    output_path = f"{output_root_dir}/{experiment}/{dataset}/{dataset}_FPDGD_clients{n_clients}_batch{batch_size}_updates{num_update}_non_iid_{non_iid_type}_{click_model_type}_neural{neural_layer}_personal/run_{seed}/"
                elif is_iid == False and fed_alg == "fedprox":
                    output_path = f"{output_root_dir}/{experiment}/{dataset}/{dataset}_FPDGD_clients{n_clients}_batch{batch_size}_updates{num_update}_non_iid_{non_iid_type}_{click_model_type}_neural{neural_layer}_fedprox_mu{mu}_short/run_{seed}/"
                elif is_iid == False and fed_alg == "fedavg":
                    if is_mixed == True:
                        output_path = f"{output_root_dir}/{experiment}/{dataset}/{dataset}_FPDGD_clients{n_clients}_batch{batch_size}_updates{num_update}_non_iid_{non_iid_type}_{click_model_type}_neural{neural_layer}_mixed/run_{seed}/"
                    else:
                        output_path = f"{output_root_dir}/{experiment}/{dataset}/{dataset}_FPDGD_clients{n_clients}_batch{batch_size}_updates{num_update}_non_iid_{non_iid_type}_{click_model_type}_neural{neural_layer}_short/run_{seed}/"
                        # output_path = f"{output_root_dir}/{experiment}/{dataset}/{dataset}_FPDGD_clients{n_clients}_batch{batch_size}_updates{num_update}_non_iid_{non_iid_type}2_V1_{click_model_type}_neural{neural_layer}/run_{seed}/"

                # click models
                if dataset == "MQ2007" or dataset == "MQ2008":
                    PERFECT_MODEL = CcmClickModel(click_relevance={0: 0.0, 1: 0.5, 2: 1.0},
                                                  stop_relevance={0: 0.0, 1: 0.0, 2: 0.0},
                                                  name="Perfect", depth=10)
                    NAVIGATIONAL_MODEL = CcmClickModel(click_relevance={0: 0.05, 1: 0.5, 2: 0.95},
                                                       stop_relevance={0: 0.2, 1: 0.5, 2: 0.9},
                                                       name="Navigational", depth=10)
                    INFORMATIONAL_MODEL = CcmClickModel(click_relevance={0: 0.4, 1: 0.7, 2: 0.9},
                                                        stop_relevance={0: 0.1, 1: 0.3, 2: 0.5},
                                                        name="Informational", depth=10)
                    # click_models = [PERFECT_MODEL, NAVIGATIONAL_MODEL, INFORMATIONAL_MODEL]

                elif dataset == "intent-change":
                    PERFECT_MODEL = CcmClickModel(click_relevance={0: 0.0, 1: 1.0},
                                                  stop_relevance={0: 0.0, 1: 0.0},
                                                  name="Perfect", depth=10)
                    NAVIGATIONAL_MODEL = CcmClickModel(click_relevance={0: 0.05, 1: 0.95},
                                                       stop_relevance={0: 0.2, 1: 0.9},
                                                       name="Navigational", depth=10)
                    INFORMATIONAL_MODEL = CcmClickModel(click_relevance={0: 0.3, 1: 0.7},
                                                        stop_relevance={0: 0.1, 1: 0.5},
                                                        name="Informational", depth=10)

                elif dataset == "MSLR10K" or dataset == "Yahoo" or dataset == "Istella-s":
                    ## CCM
                    PERFECT_MODEL = CcmClickModel(click_relevance={0: 0.0, 1: 0.2, 2: 0.4, 3: 0.8, 4: 1.0},
                                                  stop_relevance={0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
                                                  name="Perfect",
                                                  depth=10)
                    NAVIGATIONAL_MODEL = CcmClickModel(click_relevance={0: 0.05, 1: 0.3, 2: 0.5, 3: 0.7, 4: 0.95},
                                                       stop_relevance={0: 0.2, 1: 0.3, 2: 0.5, 3: 0.7, 4: 0.9},
                                                       name="Navigational",
                                                       depth=10)
                    INFORMATIONAL_MODEL = CcmClickModel(click_relevance={0: 0.4, 1: 0.6, 2: 0.7, 3: 0.8, 4: 0.9},
                                                        stop_relevance={0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4, 4: 0.5},
                                                        name="Informational",
                                                        depth=10)

                    click_models = [PERFECT_MODEL, NAVIGATIONAL_MODEL, INFORMATIONAL_MODEL]

                    ## PBM
                    # click_models = []
                    # eta_list = [0, 0.5, 1, 1.5, 2]
                    # for eta in eta_list:
                    #     click_models.append(PbmClickModel(click_relevance={0: 0.0, 1: 0.2, 2: 0.4, 3: 0.8, 4: 1.0},
                    #                                       name="PBM-eta{}".format(eta),
                    #                                       depth=10,
                    #                                       eta=eta))

                common_params = dict(
                    n_clients=n_clients,
                    interactions_budget=interactions_budget,
                    seed=seed,
                    interactions_per_feedback=batch_size,
                    multi_update=update,
                    n_features=n_features,
                    dataset_path=dataset_path,
                    intent_path=intent_path,
                    is_personal_layer=is_personal_layer,
                    is_mixed=is_mixed,
                    fed_alg=fed_alg,
                    mu=mu
                )

                if non_iid_type == "click_pref_skew":
                    tasks = itertools.product(range(n_folds),
                                              [click_models],
                                              e_s_list,
                                              range(1))
                else:
                    tasks = itertools.product(range(n_folds),
                                              [INFORMATIONAL_MODEL,
                                               NAVIGATIONAL_MODEL,
                                               PERFECT_MODEL
                                               ],
                                              e_s_list,
                                              range(1))


                run(output_path, tasks)