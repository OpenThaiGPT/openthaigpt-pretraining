from openthaigpt_pretraining_data.core.constants import MINHASH_SEED
from nlpo3 import segment
from datasketch import MinHash

DEFAULT_MINHASH_COL_NAME = 'text'
DEFAULT_NUM_PERMUTATION = 128
N_GRAM = 5

def generate_minhash_signature(text, num_perm):
    minhash = MinHash(seed=MINHASH_SEED, num_perm = num_perm)
    tokens = segment(text, "newmm")
    n_gram = N_GRAM

    for i in range(len(tokens) - n_gram + 1):
        token_gram = "".join(tokens[i : i + n_gram])

        minhash.update(token_gram.encode("utf-8"))

    return minhash

def generate_minhash_signature_hf(doc, num_perm = DEFAULT_NUM_PERMUTATION, col_name = DEFAULT_MINHASH_COL_NAME):
    minhash = generate_minhash_signature(doc[col_name], num_perm)
    return {"hashvalues": minhash.hashvalues}