import pickle

import numpy as np
from tqdm.auto import tqdm
from datasets import load_from_disk, Features, Sequence, Value
from datasketch import MinHashLSH, LeanMinHash
import pandas as pd

from openthaigpt_pretraining_data.decontamination.constants import MINHASH_SEED
from openthaigpt_pretraining_data.decontamination.utils import (
    generate_minhash_signature,
    load_data,
)


empty_hashvalues = generate_minhash_signature("").hashvalues


def query_func(item, idx, index):
    neighbors = [
        str(dup_idx)
        for dup_idx in index.query(
            LeanMinHash(seed=MINHASH_SEED, hashvalues=item["hashvalues"]),
        )
    ]
    return {"__neighbors__": neighbors, "idx": idx}


def decontaminate(dataset_groups, pretrain_data_args, decontaminate_args):
    pretrain_dataset = load_data(pretrain_data_args)
    pretrain_dataset_minhash = load_from_disk(decontaminate_args.minhash_path)
    pretrain_dataset_minhash = pretrain_dataset_minhash
    print(pretrain_dataset, "pretrain_dataset")
    print(pretrain_dataset_minhash, "pretrain_dataset_minhash")

    duplicate_results = []
    for dataset_key in tqdm(dataset_groups.keys()):
        print(dataset_key)

        dataset_arg = dataset_groups[dataset_key]

        signature_path = f"./temp/{dataset_key}_{dataset_arg.split}_signature.pickle"
        file_path = f"./temp/{dataset_key}_{dataset_arg.split}_file.pickle"

        with open(file_path, "rb") as file:
            data = pickle.load(file)

        with open(signature_path, "rb") as file:
            signature = pickle.load(file)

        globals()[dataset_key] = MinHashLSH(
            threshold=decontaminate_args.thresold,
            num_perm=decontaminate_args.num_process,
        )
        with globals()[dataset_key].insertion_session() as session:
            for i, item in enumerate(signature):
                session.insert(i, item)

        pretrain_dataset_minhash_result = pretrain_dataset_minhash.map(
            lambda doc, idx: query_func(doc, idx, index=globals()[dataset_key]),
            desc="Querying...",
            num_proc=decontaminate_args.num_process,
            features=Features(
                {
                    **pretrain_dataset_minhash.features,
                    "__neighbors__": Sequence(Value("string")),
                    "idx": Value("int32"),
                }
            ),
            with_indices=True,
        ).filter(
            lambda x: len(x["__neighbors__"]) > 0
            and not np.array_equal(x["hashvalues"], empty_hashvalues),
            desc="Filtering...",
            num_proc=decontaminate_args.num_process,
        )
        print(pretrain_dataset_minhash_result, "pretrain_dataset_minhash_result")

        for doc in tqdm(
            pretrain_dataset_minhash_result, desc="Calculation Jaccard Distance..."
        ):
            neighbors = set(doc["__neighbors__"])
            for neighbor in neighbors:
                reference = data[int(neighbor)]
                reference_signature = signature[int(neighbor)]
                score = LeanMinHash(
                    seed=MINHASH_SEED, hashvalues=doc["hashvalues"]
                ).jaccard(reference_signature)
                if score > decontaminate_args.thresold:
                    duplicate_results.append(
                        {
                            "duplicate_id": neighbor,
                            "duplicate_text": reference,
                            "duplicate_dataset": dataset_key,
                            "original_dataset": doc["source"],
                            "original_text": doc["text"],
                            "original_id": doc["idx"],
                            "score": score,
                        }
                    )
                    break
        print(len(duplicate_results), "len(duplicate_results)")

    df = pd.DataFrame(duplicate_results)
    df.to_csv("duplicate_results.csv")

    original_ids_to_remove = set()
    for item in duplicate_results:
        original_ids_to_remove.add(item["original_id"])

    pretrain_dataset_decontaminated = pretrain_dataset.filter(
        lambda _, idx: idx not in original_ids_to_remove,
        desc="Filtering...",
        num_proc=decontaminate_args.num_process,
        with_indices=True,
    )
    print(pretrain_dataset_decontaminated)
    pretrain_dataset_decontaminated.save_to_disk(decontaminate_args.save_path)
