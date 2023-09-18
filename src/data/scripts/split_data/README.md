# JSONL File Splitter
This Python script is designed to split a large JSONL (JSON Lines) file into three separate sets: train, test, and validation. It's a useful tool for preparing data for training, ensuring that you have distinct datasets for training, testing, and validation.

## Usage
In the SLURM script, modify the last line to specify the input JSONL file path and the desired test and validation set sizes:
```bash
python split_jsonl.py /path/to/your/jsonl/file.jsonl --test_size 0.001 --validation_size 0.001
```
Replace /path/to/your/jsonl/file.jsonl with the path to your JSONL file.
Adjust the --test_size and --validation_size options to set the desired proportions for the test and validation sets.

## Output
Once the SLURM job is completed, the script will split the input JSONL file into three separate files:

*_train.jsonl: Contains the training data.
*_test.jsonl: Contains the test data.
*_validation.jsonl: Contains the validation data.
The files will be saved in the following directories:

Data/train
Data/test
Data/validation

## Example
Here's an example SLURM script command:
```bash
python split_jsonl.py /scratch/lt200056-opgpth/large_data/mC4/mc4_th_new.jsonl --test_size 0.001 --validation_size 0.001
```
This command splits the mc4_th_new.jsonl file located in /scratch/lt200056-opgpth/large_data/mC4/ into train, test, and validation sets with a test size and validation size of 0.1% each.
