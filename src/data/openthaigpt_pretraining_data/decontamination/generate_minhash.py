from datasketch import LeanMinHash
from openthaigpt_pretraining_data.core.minhash import generate_minhash_signature, generate_minhash_signature_hf

from openthaigpt_pretraining_data.decontamination.utils import (
    MAPPER,
    load_data,
)
from tqdm.auto import tqdm

import pickle
from multiprocessing import Pool
from nlpo3 import load_dict


def generate_minhash_item(item):
    text, num_perm = item
    return generate_minhash_signature(text, num_perm)


def generate_minhash(dataset_groups, pretrain_data_args, minhash_config, global_config):
    load_dict(minhash_config.newmm_dict, "newmm")

    for dataset_key in dataset_groups.keys():
        dataset_arg = dataset_groups[dataset_key]
        dataset = load_data(dataset_arg)
        dataset1 = [
            MAPPER[dataset_key](item) for item in dataset[dataset_arg.split].to_list()
        ]
        dataset1 = list(set(dataset1))
        dataset1_map = [(item, global_config.num_perm)for item in dataset1]

        print(dataset1[:5], dataset_key)

        with Pool(processes=global_config.num_process) as pool:
            signatures = list(
                tqdm(
                    pool.imap(generate_minhash_item, dataset1_map),
                    total=len(dataset1_map),
                    desc="Processing dataset",
                )
            )
            signature_path = (
                f"./temp/{dataset_key}_{dataset_arg.split}_signature_{global_config.num_perm}.pickle"
            )
            file_path = f"./temp/{dataset_key}_{dataset_arg.split}_file_{global_config.num_perm}.pickle"

            with open(signature_path, "wb") as file:
                pickle.dump(signatures, file)
            with open(file_path, "wb") as file:
                pickle.dump(dataset1, file)

    pretrain_dataset = load_data(pretrain_data_args)

    dataset1 = pretrain_dataset[pretrain_data_args.split]
    signatures = dataset1.map(
        lambda x: generate_minhash_signature_hf(x, global_config.num_perm, pretrain_data_args.col_name), num_proc=global_config.num_process
    )
    signatures.save_to_disk(
        minhash_config.save_path, num_proc=global_config.num_process
    )
