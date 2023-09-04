# Internet script

This folder contains a script `main.py` to process internet dataset including OSCAR, mc4, and cc100.

## How does the code work ?

1. Read data from provided source.
2. Filter data and replace some part using regex patterns created from the studies of mc4 and OSCAR to remove garbage texts.
3. The perplexity of remain texts will be computed and use to filter more garbage data. The clean texts **`C`** will be the remaining.
4. The subset **`S`** of garbage texts from step 3 will be randomly selected.
5. Concat **`C`** and **`S`** to construct the final dataset.
6. Create metadata for the result (`created date`, `updated date`, `id`, `metadata`)

For part 2-5, You can see each part in more details at `src/data/openthaigpt_pretraining_data/internet` 

## Running

You can also process the internet data via running `main.py` 

Before running. please download `core.zip` from this [link](https://drive.google.com/file/d/1OBbo21v_-esL31rxtNtsMHrA8T1JYqAd/view?usp=sharing) and extract it in `src/data/openthaigpt_pretraining_data/internet/perplexity` first.It contains an n-gram language model weight and Decision Tree classifier weight.

### Running Example
```bash
python main.py --input_file scripts/mc4_th_validation.json \
                --source mc4 \
                --num_proc 128 \
                --batch_size 1000
                --do_perplexity True \
                --sampled_back_ratio 0.2 \
                --output_file scripts/output.jsonl
```

### Arguments
- `--input_file` : Input file/directory name of the data. Should be the name of jsonl file in case of `--source` is set to `mc4`. Otherwise, it shoud be the ma,e of downloaded directory from huggingface.
- `--source` : Source of the data file ('mc4', 'cc100', 'oscar')
- `--num_proc` : Integer number of processor used to process data
- `--batch_size` : Size of data chunks to be process in a single step of loading.
- `--do_perplexity` : `True` or `False`. Indicates if we should do Step 3-4 in `What will the code do ?`
- `--sampled_back_ratio` : Float number in range 0-1. Indicates the ratio between the number bad data to be sampled back in step 4 and the number of all bad data. This is used only when `--do_perplexity` set to `True`
- `--output_file` : filename of the jsonl output (Ex. "output.jsonl"). The output file should contains these fields `text`, `source`, `source_id`, `created_date`, `updated_date`, and other metadata dict in `meta`

## Note

- Since the code didn't use any deep learning model, **_you don't need and shouldn't use the GPU_** to run the code
- The code was meant to process the original mc4, cc100, OSCAR on LANTA.
- If you want to process custom data using this pipeline, you should prepare your jsonl file with these required fields
    - `text` : A text document you want to process
    - `timestamp` : Timestamp of the created date of data
    - `source_id` : Integer id for each text document
    And run it with
    ```bash
    python main.py --input_file your_file.jsonl --source mc4 ...
- The reason we sample text back in step use sampled garbage text is **_to teach the LLM few inappropriate words_**.
- If `--do_perplexity` is set to True and you want to sampled some bad data back, the `--batch_size` should be large enough to form the distribution in step 4 (Greater than 1000 should be fine).
