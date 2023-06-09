from datasketch import MinHash, MinHashLSH
import numpy as np

LSH = "lsh"
MINHASH_LIST = "minhash_list"
HF_TEXT_LABEL = "text"
HF_DATASET = "hf_dataset"


# create minhash signature
def generate_minhash_signature(text, num_perm=128, n_gram=3):
    """
    convert text string to minhash
    """
    assert n_gram >= 3, "n_gram should be more than or equal to 3"
    minhash = MinHash(num_perm=num_perm)
    shingles = [text[i : i + n_gram] for i in range(len(text) - n_gram - 1)]
    shingles = set(shingles)
    for shingle in shingles:
        minhash.update(shingle.encode("utf-8"))
    return minhash


# create minhasLSH from Huggingface dataset
def generate_minhashlsh(hf_dataset, threshold=0.9, num_perm=128):
    """
    create MinhashLSH from [hf_dataset] --> lsh_dict
    - hf_dataset is hugging dataset format
    !!! Warning:  if hf_dataset.shape contains split name, MUST alias
    split, for example: hf_dataset = hf_dataset["train"]
    - threshold is jaccard number. Default is 0.9 which can
    detect NearDup which is slightly changed in a few word.
    - For detecting ExactDup, recommend threshold=0.97
    - num_perm is number of permutation. Default is 128.
    Higher number, better accuracy,
    but trade-off with slower speed and higher memory
    """
    minhash_list = [generate_minhash_signature(t[HF_TEXT_LABEL]) for t in hf_dataset]
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i in range(len(minhash_list)):
        lsh.insert(i, minhash_list[i])

    # save hf_dataset in lsh_dict as pointer (global id)
    lsh_dict = {LSH: lsh, MINHASH_LIST: minhash_list, HF_DATASET: hf_dataset}
    return lsh_dict


def list_duplicated(lsh_dict, showtext=False):
    """
    list groups of duplication as python list
    - Must create lsh_dict (def generate_minhashLSH) first
    - showtext=False will return groups of duplication (list in list)
    - showtext=True will return groups of duplication and texts
    """
    hf_dataset = lsh_dict[HF_DATASET]
    lsh = lsh_dict[LSH]
    minhash_list = lsh_dict[MINHASH_LIST]
    dup_lists = []
    for m in minhash_list:
        dup_list = lsh.query(m)
        # sort index before adding to dup_lists
        # this helps when finding duplicated_list.
        dup_list.sort()
        dup_lists.append(dup_list)

    # remove len(dup_lists) < 1
    dup_lists = list(filter(lambda li: len(li) > 1, dup_lists))

    # convert to np array to get unique
    np_dup_lists = np.array(dup_lists, dtype=object)
    dup_lists = np.unique(np_dup_lists).tolist()

    if showtext is True:
        dup_lists_text = [
            [li, [hf_dataset[t][HF_TEXT_LABEL] for t in li]] for li in dup_lists
        ]
        return dup_lists_text
    return dup_lists


def remove_duplicated(lsh_dict, viewindex=False):
    """
    Remove duplication and return as deduplicated dataset or list
    - Must create lsh_dict (def generate_minhashLSH) first
    - viewindex=False will return dedup huggingface dataset format
    - viewindex=True will return list of removed indices
    """
    hf_dataset = lsh_dict[HF_DATASET]
    dup_lists = list_duplicated(lsh_dict)
    # get 1st index of dup_lists for keeping a representative of each dup_list
    first_index = [li[0] for li in dup_lists]
    first_index = list(set(first_index))
    first_index.sort()
    # flatten list for unique duplicated indexes
    new_dup_lists = [item for sublist in dup_lists for item in sublist]
    new_dup_lists = list(set(new_dup_lists))
    new_dup_lists.sort()

    # create removing indexes
    remove_indexes = list(filter(lambda li: li not in first_index, new_dup_lists))

    if viewindex is True:
        # return the remove_indexes
        return remove_indexes

    # remove duplicated text
    dedup_dataset = hf_dataset.filter(lambda dat: dat["id"] not in remove_indexes)

    return dedup_dataset


# Example of usage
"""
from datasets import load_dataset, load_from_disk

dataset = load_from_disk("/project/lt200056-opgpth/oscar2301_th/datasets")

# firstly create minhashlsh which return as lsh_dict
lsh_dict = generate_minhashlsh(dataset, 0.9)
# remove_duplicated from lsh_dict
# this will return duduplicated huggingface dataset
dedup_dat = remove_duplicated(lsh_dict)
# save dataset to proj dir
dedup_dat.save_to_disk("/project/lt200056-opgpth/oscar2301/deduplicated_dataset")

"""
