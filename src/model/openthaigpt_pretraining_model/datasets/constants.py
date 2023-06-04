C4_DATASET = "c4"
MC4_DATASET = "mc4"

DATASET_ARGS = {
    C4_DATASET: {
        "path": "c4",
        "name": "en",
        "streaming": True,
    },
    MC4_DATASET: {
        "path": "mc4",
        "languages": ["th"],
        "streaming": True,
    },
}

SPLIT_TRAIN = "train"
SPLIT_VAL = "validation"
