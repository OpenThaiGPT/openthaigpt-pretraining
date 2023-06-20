import random
import jsonlines
import os
import argparse

# Create the train, test, and validation directories
train_directory = "Data/train"
test_directory = "Data/test"
validation_directory = "Data/validation"
os.makedirs(train_directory, exist_ok=True)
os.makedirs(test_directory, exist_ok=True)
os.makedirs(validation_directory, exist_ok=True)


# Function to split a list into train, test, and validation sets
def split_data(data, test_size, validation_size):
    random.shuffle(data)
    test_index = int(len(data) * test_size)
    validation_index = int(len(data) * (test_size + validation_size))
    return data[validation_index:], data[:test_index], data[test_index:validation_index]


# Function to split a .jsonl file into train, test, and validation files
def split_jsonl_file(file_path, test_size, validation_size):
    # Get the original file name
    file_name = os.path.basename(file_path)
    file_name_without_extension = os.path.splitext(file_name)[0]

    # Create the train, test, and validation file paths
    train_file_path = os.path.join(
        train_directory, f"{file_name_without_extension}_train.jsonl"
    )
    test_file_path = os.path.join(
        test_directory, f"{file_name_without_extension}_test.jsonl"
    )
    validation_file_path = os.path.join(
        validation_directory, f"{file_name_without_extension}_validation.jsonl"
    )

    # Open the input file for reading and the output files for writing
    try:
        with jsonlines.open(file_path, "r") as reader, jsonlines.open(
            train_file_path, "w"
        ) as train_writer, jsonlines.open(
            test_file_path, "w"
        ) as test_writer, jsonlines.open(
            validation_file_path, "w"
        ) as validation_writer:
            # Process the file in chunks
            chunk_size = 10000  # Adjust the chunk size based on available memory
            while True:
                chunk = []
                for line in iter(reader):  # Create an iterator from the reader
                    chunk.append(line)
                    if len(chunk) >= chunk_size:
                        break
                else:
                    break  # Exit the while loop if no more lines to read

                # Split the chunk into train, test, and validation sets
                train_chunk, test_chunk, validation_chunk = split_data(
                    chunk, test_size, validation_size
                )

                # Write the train data to the train file
                train_writer.write_all(train_chunk)

                # Write the test data to the test file
                test_writer.write_all(test_chunk)

                # Write the validation data to the validation file
                validation_writer.write_all(validation_chunk)

    except (OSError, jsonlines.InvalidLineError) as e:
        print(f"Error processing file: {file_path}. {str(e)}")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a .jsonl file into train, test, and validation files."
    )
    parser.add_argument("file_path", type=str, help="Path to the JSONL file to split.")
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.01,
        help="Percentage of data to use for the test set (default: 0.1)",
    )
    parser.add_argument(
        "--validation_size",
        type=float,
        default=0.001,
        help="Percentage of data to use for the validation set (default: 0.2)",
    )
    args = parser.parse_args()
    file_path = args.file_path
    test_size = args.test_size
    validation_size = args.validation_size

    print(f"Splitting file: {file_path}")
    split_jsonl_file(file_path, test_size, validation_size)

    print("Splitting of .jsonl file into train/test/validation sets completed.")
