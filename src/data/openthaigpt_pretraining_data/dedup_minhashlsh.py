from datasketch import MinHash, MinHashLSH
import numpy as np

LSH = "lsh"
TEXT_LIST = "text_list"
MINHASH_LIST = "minhash_list"


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


# create minhasLSH from text_lists
def generate_minhashlsh(text_list, threshold=0.9, num_perm=128):
    """
    create MinhashLSH from [text_list] --> lsh_dict
    - threshold is jaccard number. Default is 0.9 which can
    detect NearDup which is slightly changed in a few word.
    - For detecting ExactDup, recommend threshold=0.97
    - num_perm is number of permutation. Default is 128.
    Higher number, better accuracy,
    but trade-off with slower speed and higher memory
    """
    minhash_list = [generate_minhash_signature(t) for t in text_list]
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i in range(len(minhash_list)):
        lsh.insert(i, minhash_list[i])

    lsh_dict = {LSH: lsh, TEXT_LIST: text_list, MINHASH_LIST: minhash_list}
    return lsh_dict


def list_duplicated(lsh_dict, showtext=False):
    """
    list groups of duplication as python list
    - Must create lsh_dict (def generate_minhashLSH) first
    - showtext=False will return groups of duplication (list in list)
    - showtext=True will return groups of duplication and texts
    """
    text_list = lsh_dict[TEXT_LIST]
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
        dup_lists_text = [[li, [text_list[t] for t in li]] for li in dup_lists]
        return dup_lists_text
    return dup_lists


def remove_duplicated(lsh_dict, viewindex=False):
    """
    Remove duplication and return as python list
    - Must create lsh_dict (def generate_minhashLSH) first
    - viewindex=False will return list of deduplicated text
    - viewindex=True will return list of removed indices
    """
    text_list = lsh_dict[TEXT_LIST]
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
    dedup_texts = []
    for i in range(len(text_list)):
        if i not in remove_indexes:
            dedup_texts.append(text_list[i])
    return dedup_texts


# Example of usage
"""
# Read jsonline dataset and extract as text lists
import jsonlines

dat_jsonl = "oscar_th_10k.jsonl"
text_list = []

with jsonlines.open(dat_jsonl) as reader:
    for obj in reader:
        text_list.append(obj["text"])

# Generate minhashLSH first
lsh_dict = generate_minhashlsh(text_list,0.9)

# Get deduplicated text lists
dedup = remove_duplicated(lsh_dict)

"""
