import copy
import random

## codebase related to intent change datasets
def read_intent_qrel(path: str):
    # q-d pair dictionary
    qrel_dic = {}

    with open(path, 'r') as f:
        for line in f:
            qid, _, docid, rel = line.strip().split()
            if qid in qrel_dic.keys():
                qrel_dic[qid][docid] = int(rel)
            else:
                qrel_dic[qid] = {docid: int(rel)}
    return qrel_dic


def get_intent_dataset(train_set, test_set, intent_path):
    new_train_set = copy.deepcopy(train_set)
    new_test_set = copy.deepcopy(test_set)
    qrel_dic = read_intent_qrel(intent_path)
    new_train_set.update_relevance_label(qrel_dic)
    new_test_set.update_relevance_label(qrel_dic)
    return new_train_set, new_test_set


def get_intent_groups_dataset(train_set, intent_paths, seed, num_groups=4):
    """
    Get shuffled dataset from all intent folder to have balanced 0-1 data.
    (In the original intent folder, different intent has different 0-1 ratio)
    :param train_set:
    :param intent_paths:
    :param seed:
    :param num_groups:
    :return:
    """
    qrel_dics = []

    print("Reading intents......")
    for path in intent_paths:
        qrel_dics.append(read_intent_qrel(path))

    print("Randomly assign groups......")
    n_qid = len(qrel_dics[0].keys())
    n = 0

    for qid in qrel_dics[0].keys(): # qid is the same in different intent folder, so here only uses qid in the first folder
        n += 1
        qid_rel_lists = []
        for qrel_dic in qrel_dics:
            doc_rels = {}
            for docid in qrel_dic[qid].keys():
                doc_rels[docid] = qrel_dic[qid][docid]
            qid_rel_lists.append(doc_rels)

        random.Random(seed * n_qid + n).shuffle(qid_rel_lists)
        for i in range(len(qrel_dics)):
            for docid in qrel_dics[i][qid].keys():
                qrel_dics[i][qid][docid] = qid_rel_lists[i][docid]

    datasets = []
    print("Generating new datasets......")
    for qrel_dic in qrel_dics:
        new_train_set = copy.deepcopy(train_set)
        new_train_set.update_relevance_label(qrel_dic)
        datasets.append(new_train_set)
    return datasets[:num_groups]


