from openthaigpt_pretraining_data.core.constants import MINHASH_SEED
from nlpo3 import segment
from datasketch import MinHash


def generate_minhash_signature(text, num_perm):
    minhash = MinHash(seed=MINHASH_SEED, num_perm = num_perm)
    tokens = segment(text, "newmm")
    n_gram = 5

    for i in range(len(tokens) - n_gram + 1):
        token_gram = "".join(tokens[i : i + n_gram])

        minhash.update(token_gram.encode("utf-8"))

    return minhash

def generate_minhash_signature_hf(doc, num_perm = 128, col_name = 'text'):
    minhash = generate_minhash_signature(doc[col_name], num_perm)
    return {"hashvalues": minhash.hashvalues}