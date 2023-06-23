from datasets import load_from_disk

HF_TEXT_LABEL = {"openthaigpt": "text"}


def blind_pdpa(dataset_config, blind_config):
    hf_dataset = load_from_disk(dataset_config.path_name)[dataset_config.split]

    if blind_config.engine == "openthaigpt":
        from openthaigpt_pdpa import blind_pdpa_text

        result_data = hf_dataset.map(
            lambda doc: {HF_TEXT_LABEL: blind_pdpa_text(doc[HF_TEXT_LABEL])},
            num_proc=blind_config.num_proc,
        )
        result_data.save_to_disk(blind_config.save_path)

    else:
        raise NotImplementedError("Other Blind PDPA Engine will be supported soon.")
