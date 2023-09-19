# JSONL File Merger

This Python script merges JSONL (JSON Lines) files from a specified folder into a single output file.

## Usage
To merge JSONL files in a folder and save the merged data to an output file, In the SLURM script, modify the last line to specify the input folder containing the JSONL files and the output file path:

```bash
python merge_jsonl_files.py <folder_path> <output_file.jsonl>