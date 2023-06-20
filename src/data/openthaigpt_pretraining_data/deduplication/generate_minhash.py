from openthaigpt_pretraining_data.core.minhash import generate_minhash_signature_hf

from datasets import load_from_disk

from nlpo3 import load_dict


def generate_minhash(pretrain_data_args, minhash_config, global_config):
    load_dict(minhash_config.newmm_dict, "newmm")

    pretrain_dataset = load_from_disk(pretrain_data_args.path_name)

    dataset1 = pretrain_dataset[pretrain_data_args.split]
    signatures = dataset1.map(
        lambda x: generate_minhash_signature_hf(x, global_config.num_perm), num_proc=global_config.num_process
    )
    signatures.save_to_disk(
        minhash_config.save_path, num_proc=minhash_config.num_process
    )
