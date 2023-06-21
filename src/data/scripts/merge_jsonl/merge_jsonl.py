import argparse
import jsonlines
import glob


def merge_jsonl_files(folder_path, output_file):
    # Get a list of JSONL files in the folder
    file_list = glob.glob(folder_path + "/*.jsonl")

    # Create an empty list to store the merged data
    merged_data = []

    # Iterate over each file and append its data to the merged_data list
    for file_path in file_list:
        with jsonlines.open(file_path) as reader:
            merged_data.extend(list(reader))

    # Write the merged data to the output file
    with jsonlines.open(output_file, mode="w") as writer:
        writer.write_all(merged_data)

    print(f"Merged data written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge JSONL files in a folder.")
    parser.add_argument(
        "folder_path", type=str, help="Path to the folder containing JSONL files"
    )
    parser.add_argument("output_file", type=str, help="Output file path")

    args = parser.parse_args()

    merge_jsonl_files(args.folder_path, args.output_file)
