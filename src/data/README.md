# Data

Folder `openthaigpt_pretraining_data` contains the preprocess function, EDA notebook, and examples for each dataset

Folder `scripts contains` the main program which will call function in folder `openthaigpt_pretraining_data`

If you want to code the function to clean new dataset.
1. Copy `core/preprocess.py` and code your function there.
2. Make sure your preprocess.py have function named `clean_dataset`.
3. Edit `scripts/main.py`.
3.1 import your `clean_dataset` function. Make sure you alias the function name.
3.2 Add your dataset name in choices for parser's engine argument (`parser.add_argument("--engine")`)
3.3 Add your dataset name and function as key-value pair of `CLEAN_FUNCTION`.
4. It would be great to write the testcase for your dataset. Add your test file in `tests/data`.




