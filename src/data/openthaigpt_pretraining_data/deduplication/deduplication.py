import pickle

import numpy as np
from tqdm.auto import tqdm
from datasets import load_from_disk, Features, Sequence, Value
from datasketch import MinHashLSH, LeanMinHash
import pandas as pd

from openthaigpt_pretraining_data.core.constants import MINHASH_SEED
from openthaigpt_pretraining_data.core.minhash import generate_minhash_signature




def query_func(item, idx, index):
    neighbors = [
        str(dup_idx)
        for dup_idx in index.query(
            LeanMinHash(seed=MINHASH_SEED, hashvalues=item["hashvalues"]),
        )
    ]
    return {"__neighbors__": neighbors, "idx": idx}


def deduplicate(pretrain_data_args, deduplicate_args, global_config):
    pretrain_dataset = load_from_disk(pretrain_data_args.path_name)
    pretrain_dataset_minhash = load_from_disk(deduplicate_args.minhash_path)
    print(pretrain_dataset_minhash, "pretrain_dataset_minhash")

    empty_hashvalues = generate_minhash_signature("", global_config.num_perm).hashvalues

    minhash_index = MinHashLSH(
        threshold=deduplicate_args.thresold,
        num_perm=global_config.num_perm,
    )
    batch_size = 10000
    count = 0
    with minhash_index.insertion_session() as session:
        for i in tqdm(
            range(0, len(pretrain_dataset_minhash), batch_size), dynamic_ncols=True, desc="Iterating MinHashes..."  # noqa: E501
        ):
            batch = pretrain_dataset_minhash[i : i + batch_size]
            for j, Hs in enumerate(batch["hashvalues"]):
                count += 1
                key = i + j
                session.insert(key, LeanMinHash(seed=MINHASH_SEED, hashvalues=Hs))
    duplicate_results = []
    for i in tqdm(
      range(0, len(pretrain_dataset_minhash), batch_size), dynamic_ncols=True, desc="Iterating MinHashes..."  # noqa: E501
    ):
        batch = pretrain_dataset_minhash[i : i + batch_size]
        hashvalues = batch['hashvalues']
        for j in range(len(hashvalues)):
            key = i + j
            doc_hash_value = hashvalues[j]

            ## DO NOT DEDUPLICATE DATA THAT HAS EMPTY HASHVALUES
            if np.array_equal(doc_hash_value, empty_hashvalues):
                continue
            
            minhash = LeanMinHash(
                seed=MINHASH_SEED, hashvalues=doc_hash_value
            )
            neighbors = minhash_index.query(minhash)

            for neighbor in neighbors:
                if neighbor == key:
                    continue
                reference = pretrain_dataset_minhash[int(neighbor)]
                reference_signature = LeanMinHash(seed=MINHASH_SEED, hashvalues=reference["hashvalues"])
                score = minhash.jaccard(reference_signature)
                # print(score, 'score')
                if score > deduplicate_args.thresold:
                    duplicate_results.append(
                        {
                            "duplicate_id": neighbor,
                            "duplicate_text": reference['text'],
                            "duplicate_dataset": reference['source'],
                            "original_dataset": batch["source"][j],
                            "original_text": batch["text"][j],
                            "original_id": key,
                            "score": score,
                        }
                    )
                    break
    print(len(duplicate_results), "len(duplicate_results)")

    df = pd.DataFrame(duplicate_results)
    df.to_parquet(f"duplicate_results_{global_config.num_perm}.parquet")

    original_ids_to_remove = set()
    for item in duplicate_results:
        original_ids_to_remove.add(item["original_id"])

    pretrain_dataset[pretrain_data_args.split] = pretrain_dataset[pretrain_data_args.split].filter(
        lambda _, idx: idx not in original_ids_to_remove,
        desc="Filtering...",
        num_proc=global_config.num_process,
        with_indices=True,
    )
    print(pretrain_dataset)
    pretrain_dataset.save_to_disk(deduplicate_args.save_path)
