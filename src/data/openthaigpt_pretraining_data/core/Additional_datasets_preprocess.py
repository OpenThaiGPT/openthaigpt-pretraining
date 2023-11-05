from datasets import load_dataset
import pandas as pd
import jsonlines
from tqdm import tqdm
import os
import re

"""This code block defines several variables
that will be used in the preprocessing of a dataset.
These variables include the dataset name,
target and origin languages, json path,
create and update dates, meta label, text column name,
file path, and folder path.
These variables will be used in the subsequent code blocks to load the dataset,
preprocess the text, and write the preprocessed text to a jsonline file."""

DUMMY_DATASET_NAME = ""  # dataset name
TARGET_LANGUAGE = ""  # target language
ORIGIN_LANGUAGE = ""  # origin language
JSONL_PATH = ""  # json path
DATASET_CREATE_DATE = ""  # create date
DATASET_UPDATE_DATE = ""  # update date
META_LABEL = ""  # meta label
TEXT_COLUMN = ""  # text column name
FILE_PATH = ""  # file path
FOLDER_PATH = ""  # folder path
NUM_OF_CHAR_THRESHOLD = int()  # number of character threshold

"""Load dataset"""
raw_dataset = load_dataset(DUMMY_DATASET_NAME)


def load_pickle(folder_path):
    """
    Load a pickle file as a pandas dataframe.

    Args:
    - folder_path: str, the path to the folder containing the pickle file.

    Returns:
    - pickle_df: pandas dataframe, the loaded pickle file.
    """
    pickle_df = pd.read_pickle(folder_path)
    return pickle_df


def combine_pickle(folder_path):
    """
    Combine multiple pickle files into a single pandas dataframe.

    Args:
    - FOLDER_PATH: str, the path to the folder containing the pickle files.

    Returns:
    - combined_df: pandas dataframe,
    the combined pickle files as a single dataframe.
    """
    folder_path = FOLDER_PATH
    pickle_files = [file for file in os.listdir(folder_path)
                    if file.endswith(".pkl")]
    pickle_df = []

    for file in pickle_files:
        file_path = os.path.join(folder_path, file)
        dataframe = pd.read_pickle(file_path)
        pickle_df.append(dataframe)

    combined_df = pd.concat(pickle_df, ignore_index=True)

    return combined_df


def text_summarize_processing(header, detail, summary=None):
    """
    Preprocesses text by summarizing it and concatenating the header,
    detail, and summary (if provided).

    Args:
    - header: str, the header of the text.
    - detail: str, the detail of the text.
    - summary: str (optional), the summary of the text.

    Returns:
    - compond_text: str, the preprocessed text.
    """
    header = header
    detail = detail
    if summary is None:
        compond_text = f"หัวข้อ: {header} + '\n' + เนื้อหา: {detail}"
    else:
        compond_text = (
            f"หัวข้อ:{header} + '\n' + เนื้อหา:{detail} +'\n' + สรุป:{summary}"
        )

    return compond_text


def translate_preprocessing(dataset, orgin_text, target_text):
    """
    Preprocesses text by translating it
    and concatenating the original and translated text.

    Args:
    - dataset: dict, the dataset containing the original and target text.
    - orgin_text: str, the original text.
    - target_text: str, the target text.

    Returns:
    - compond_text: str, the preprocessed text.
    """
    orgin_text = dataset[TARGET_LANGUAGE]
    target_text = dataset[ORIGIN_LANGUAGE]
    compond_text = f"Thai: {orgin_text} + '\n' + English: {target_text}"

    return compond_text


def drop_invalid_text_inlist(text_list):
    """
    Removes any text in a list that is less than 50 characters long.

    Args:
    - text_list: list, the list of text to process.

    Returns:
    - text_list: list, the processed list of text.
    """
    for index in range(len(text_list)):
        if len(text_list[index]) < NUM_OF_CHAR_THRESHOLD:
            text_list.pop(index)

    return text_list


def drop_invalid_text_df(text_df):
    """
    Removes any text in a pandas dataframe that is less than threshold.

    Args:
    - text_df: pandas dataframe, the dataframe to process.

    Returns:
    - clean_text_df: pandas dataframe, the processed dataframe.
    """
    clean_text_df = text_df[text_df[TEXT_COLUMN].str.len()
                            > NUM_OF_CHAR_THRESHOLD]
    clean_text_df = clean_text_df.reset_index(drop=True, inplace=True)

    return clean_text_df


def combine_translate(text_list):
    """
    Translates a list of text and
    sconcatenates the original and translated text.

    Args:
    - text_list: list, the list of text to process.

    Returns:
    - compond_text: str, the preprocessed text.
    """
    compond_text = []
    for index in range(len(text_list)):
        text_index = text_list[index]
        compond_text = translate_preprocessing(
            text_index[ORIGIN_LANGUAGE], text_index[TARGET_LANGUAGE]
        )
        compond_text.append(compond_text)

    return compond_text


def write_jsonline_with_index(text_list):
    """
    Writes a list of text to a jsonl file with an index for each row.

    Args:
    - text_list: list, the list of text to write to the jsonl file.

    Returns:
    - None
    """
    with jsonlines.open(f"{JSONL_PATH}.jsonl", mode="w") as writer:
        for index in tqdm(range(len(text_list))):
            # Create a dictionary for the row
            row_dict = {
                "text": text_list[index],
                "source": DUMMY_DATASET_NAME,
                "source_id": index,
                "created_date": DATASET_CREATE_DATE,
                "updated_date": DATASET_UPDATE_DATE,
                "meta": META_LABEL,
            }
            # Write the dictionary to the .jsonl file
            writer.write(row_dict)

    return None


def write_jsonline_with_df(text_df):
    """
    Writes a pandas dataframe of text to a jsonl file
    with an index for each row.

    Args:
    - text_df: pandas dataframe, the dataframe of text
    to write to the jsonl file.

    Returns:
    - None
    """
    with jsonlines.open(f"{JSONL_PATH}.jsonl", mode="w") as writer:
        for row, index in tqdm((text_df.iterrows())):
            # Create a dictionary for the row
            row_dict = {
                "text": row[index],
                "source": DUMMY_DATASET_NAME,
                "source_id": index,
                "created_date": DATASET_CREATE_DATE,
                "updated_date": DATASET_UPDATE_DATE,
                "meta": META_LABEL,
            }
            # Write the dictionary to the .jsonl file
            writer.write(row_dict)

    return None


def write_best_corpus(FILE_PATH, enccypedia=False):
    """
    Writes the best corpus to a jsonl file.

    Args:
    - FILE_PATH: str, the path to the file to write to.
    - enccypedia: bool, whether to preprocess the text as an encyclopedia.

    Returns:
    - None
    """
    with jsonlines.open(f"{FILE_PATH}.jsonl", mode="w") as writer:
        for index in os.listdir(f"{FOLDER_PATH}"):
            with open(f"{FILE_PATH}" + index, "r") as file:
                word = file.read()

                if enccypedia is True:
                    rows = word.split("\n")
                    data_list = [row.split("\t") for row in rows]
                    dataframe = pd.DataFrame(
                        data_list, columns=[
                            "text",
                            "label",
                            "label2",
                            "label3"]
                    )
                    text = dataframe["text"]
                    word = (
                        "".join(["<_>" if item == ""
                                else item for item in text])
                        .replace("<_>", "\n")
                        .replace("_", " ")
                    )

                else:
                    pass

                word = re.sub(r"[|]", "", word)
                word = re.sub(
                    r"^https?:\/\/.*[\r\n]*", "", word, flags=re.MULTILINE
                    )
                word = word.replace("</NE>", "")
                word = word.replace("<NE>", "")
                word = word.replace("<AB>", "")
                word = word.replace("</AB>", "")
                word = word.replace("<POEM>", "")
                word = word.replace("</POEM>", "")
                word = word.replace("\ufeff", "")

                row_dict = {
                    "text": word,
                    "source": FOLDER_PATH,
                    "source_id": index,
                    "created_date": DATASET_CREATE_DATE,
                    "updated_date": DATASET_UPDATE_DATE,
                    "meta": META_LABEL,
                }
                # Write the dictionary to the .jsonl file
                writer.write(row_dict)
                
    return None


def write_lst_corpus(Folder_PATH):
    """
    Writes the LST corpus to a jsonl file.

    Args:
    - Folder_PATH: str, the path to the folder containing
    the files to write to.

    Returns:
    - None
    """
    with jsonlines.open(f"{Folder_PATH}+{FILE_PATH}.jsonl",
                        mode="w") as writer:
        text_list = []

        for i in os.listdir(f"{Folder_PATH}"):
            with open(f"{Folder_PATH}/" + i, "r") as file:
                data_str = file.read()
                # Split the string into rows and create a list
                rows = data_str.split("\n")
                # Split each row into columns and create a list of lists
                data_list = [row.split("\t") for row in rows]
                # Convert the list of lists to a pandas dataframe
                dataframe = pd.DataFrame(
                    data_list, columns=["text", "label", "label1", "label3"]
                )
                text = dataframe["text"]
                word = (
                    "".join(["<_>" if item == "" else item for item in text])
                    .replace("<_>", "\n")
                    .replace("_", " ")
                )
                text_list.append(word)

        for i in tqdm(range(len(text_list))):
            # Create a dictionary for the row
            row_dict = {
                "text": text_list[i],
                "source": "LST",
                "source_id": i,
                "created_date": DATASET_CREATE_DATE,
                "updated_date": DATASET_UPDATE_DATE,
                "meta": META_LABEL,
            }
            # Write the dictionary to the .jsonl file
            writer.write(row_dict)

    return None
