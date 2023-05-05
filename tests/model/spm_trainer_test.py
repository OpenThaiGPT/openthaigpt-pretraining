from openthaigpt_pretraining_model.tokenizers.spm_trainer import train_tokenizer

OUTPUT_PATH = "./tokenizer"
VOCAB_SIZE_TEST_CASES = [5000, 7500]
NUM_DOCS_TEST_CASES = [500, 750]


def test_train_tokenizer():
    for i, num_docs in enumerate(NUM_DOCS_TEST_CASES):
        train_tokenizer(
            output_path=OUTPUT_PATH,
            vocab_size=VOCAB_SIZE_TEST_CASES[i],
            num_docs=num_docs,
        )
