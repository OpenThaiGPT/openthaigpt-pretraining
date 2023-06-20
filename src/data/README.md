# Data

Folder `openthaigpt_pretraining_data` contains the preprocess function, EDA notebook, and examples for each dataset

Folder `scripts` contains the main program which will call function in folder `openthaigpt_pretraining_data`

Before running `scripts/main.py`, please download model and lm weight from this link[https://drive.google.com/file/d/1OBbo21v_-esL31rxtNtsMHrA8T1JYqAd/view?usp=sharing] and extract it in `openthaigpt_pretraining_data/core` 

If you want to code the function to clean new dataset.
1. Copy `core/preprocess.py` and code your function there.
2. Make sure your preprocess.py have function named `clean_dataset`.
3. It would be great to write the testcase for your dataset. Add your test file in `tests/data`.




