## Usage

All functionality (inference included), should be launched using `deepy.py`, a wrapper around the `deepspeed` launcher.

We currently offer three main functions:
1. `train.py` is used for training and finetuning models.
2. `evaluate.py` is used to evaluate a trained model using the [language model evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness).
3. `generate.py` is used to sample text from a trained model.

which can be launched with:

```bash
sudo ./deepy.py [script.py] [./path/to/config_1.yml] [./path/to/config_2.yml] ... [./path/to/config_n.yml]
```

E.G To generate text unconditionally with the GPT-NeoX-20B model, you can use the following:
```bash
sudo ./deepy.py generate.py ./configs/20B.yml
```

Or optionally pass in a text file (e.g `prompt.txt`) to use as the prompt, which should be a plain `.txt` file with each prompt separated by newline characters, also passing in the path to an output file.

```bash
sudo ./deepy.py generate.py ./configs/20B.yml -i prompt.txt -o sample_outputs.txt
```

To reproduce our evaluation numbers on, for example, TriviaQA and PIQA use:

```bash
sudo ./deepy.py evaluate.py ./configs/20B.yml --eval_tasks triviaqa piqa
```
# Datasets

## Using Custom Data

To prepare your own dataset for training with custom data, format it as one large [jsonl](https://jsonlines.org/)-formatted file with each item in the list of dictionaries being a separate document. The document text should be grouped under one JSON key, i.e `"text"`. Any auxiliary data stored in other fields will not be used.

Next make sure to download the GPT2 tokenizer vocab, and merge files from the following links:

- Vocab: https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
- Merge: https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt

Or use the 20B tokenizer (for which only a single Vocab file is needed):

- Vocab: https://the-eye.eu/public/AI/models/GPT-NeoX-20B/slim_weights/20B_tokenizer.json

(alternatively, you can provide any tokenizer file that can be loaded by Hugging Face's tokenizers library with the `Tokenizer.from_pretrained()` command)

You can now pretokenize your data using `tools/preprocess_data.py`, the arguments for which are detailed below:

```
usage: preprocess_data.py [-h] --input INPUT [--jsonl-keys JSONL_KEYS [JSONL_KEYS ...]] [--num-docs NUM_DOCS] --tokenizer-type {HFGPT2Tokenizer,HFTokenizer,GPT2BPETokenizer,CharLevelTokenizer} [--vocab-file VOCAB_FILE] [--merge-file MERGE_FILE] [--append-eod] [--ftfy] --output-prefix OUTPUT_PREFIX
                          [--dataset-impl {lazy,cached,mmap}] [--workers WORKERS] [--log-interval LOG_INTERVAL]

optional arguments:
  -h, --help            show this help message and exit

input data:
  --input INPUT         Path to input jsonl files or lmd archive(s) - if using multiple archives, put them in a comma separated list
  --jsonl-keys JSONL_KEYS [JSONL_KEYS ...]
                        space separate listed of keys to extract from jsonl. Defa
  --num-docs NUM_DOCS   Optional: Number of documents in the input data (if known) for an accurate progress bar.

tokenizer:
  --tokenizer-type {HFGPT2Tokenizer,HFTokenizer,GPT2BPETokenizer,CharLevelTokenizer}
                        What type of tokenizer to use.
  --vocab-file VOCAB_FILE
                        Path to the vocab file
  --merge-file MERGE_FILE
                        Path to the BPE merge file (if necessary).
  --append-eod          Append an <eod> token to the end of a document.
  --ftfy                Use ftfy to clean text

output data:
  --output-prefix OUTPUT_PREFIX
                        Path to binary output file without suffix
  --dataset-impl {lazy,cached,mmap}
                        Dataset implementation to use. Default: mmap

runtime:
  --workers WORKERS     Number of worker processes to launch
  --log-interval LOG_INTERVAL
                        Interval between progress updates

```

For example:

```bash
sudo python tools/preprocess_data.py \
            --input ./data/mydataset.jsonl.zst \
            --output-prefix ./data/mydataset \
            --vocab ./data/gpt2-vocab.json \
            --merge-file gpt2-merges.txt \
            --tokenizer-type GPT2BPETokenizer \
```

You would then run training with the following settings added to your configuration file:

```yaml
  "data_path": "data/mydataset_text_document",
```

# Training

Training is launched using `deepy.py`, a wrapper around DeepSpeed's launcher, which launches the same script in parallel across many GPUs / nodes.

The general usage pattern is:

```bash
sudo python ./deepy.py train.py [path/to/config1.yml] [path/to/config2.yml] ...
```

You can pass in an arbitrary number of configs which will all be merged at runtime.

You can also optionally pass in a config prefix, which will assume all your configs are in the same folder and append that prefix to their path.

E.G:

```bash
sudo python ./deepy.py train.py -d configs 125M.yml local_setup.yml
```

This will deploy the `train.py` script on all nodes with one process per GPU. The worker nodes and number of GPUs are specified in the `/job/hostfile` file (see [parameter documentation](configs/README.md)), or can simply be passed in as the `num_gpus` arg if running on a single node setup.

Although this is not strictly necessary, we find it useful to define the model parameters in one config file (e.g `configs/125M.yml`) and the data path parameters in another (e.g `configs/local_setup.yml`).

