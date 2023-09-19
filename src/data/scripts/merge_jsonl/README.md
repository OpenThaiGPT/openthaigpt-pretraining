# JSONL File Merger

This Python script merges JSONL (JSON Lines) files from a specified folder into a single output file.

## Dependencies
Before using this script, make sure you have the following dependencies installed:

jsonlines>=3.1.0
You can typically install these dependencies using pip:

```bash
pip install jsonlines>=3.1.0
```

## Usage
To merge JSONL files in a folder and save the merged data to an output file, In the SLURM script, modify the last line to specify the input folder containing the JSONL files and the output file path:

```bash
python merge_jsonl_files.py folder_path output_file