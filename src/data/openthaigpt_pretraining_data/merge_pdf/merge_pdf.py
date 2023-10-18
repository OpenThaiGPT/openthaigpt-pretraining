# import os
import pandas as pd
import PyPDF2
import pdf_table2json.converter as converter
from difflib import SequenceMatcher
import re

# import numpy as np
# import statistics
# import math


def convert_correction_rules(filepath):
    """
    Description:
        Convert pdf_correction_rules.txt to
        list of tuples (wrong, right).
        This list will be used for correcting pdf-converted text.
    Args:
        filepath: string of file path.
    Returns:
        replace_list: List of tuples (incorrect_format , correct_format).
    """

    with open(filepath) as f:
        lines = f.readlines()

    replace_list = [
        eval(t.strip().replace("$line = str_replace", "").replace(",$line", "")[:-1])
        for t in lines
    ]
    return replace_list


def read_pdf_all_pages(pdf_file):
    """
    Description:
        Read all texts in every page of pdf file and return as string
        Require PyPDF2
    Args:
        pdf_file: string of pdf file path.
    Returns:
        read_text: string.
    """

    reader = PyPDF2.PdfReader(pdf_file)
    page_count = len(reader.pages)
    read_text = [reader.pages[i].extract_text() for i in range(page_count)]
    read_text = "".join(map(str, read_text))
    return read_text


def clean_dup_esc_char(str_text):
    """
    Description:
        Reduce duplicated escape characters into one
    Args:
        str_text: string.
    Returns:
        clean_dup: string.
    """

    clean_dup = re.sub(r"(\W)\1+", r"\1", str_text)
    return clean_dup


def clean_by_rules(str_text, rules_filepath):
    """
    Description:
        Clean input text that contain incorrect word structure by replacing with
        list of correct word structure which is written in pdf_correction_rules.txt
        as php format.
    Args:
        str_text: input string.
        rule_filepath: string of file path
    Returns:
        str_text: clean string.
    """

    replace_list = convert_correction_rules(rules_filepath)

    for tu in replace_list:
        str_text = str_text.replace(tu[0], tu[1])

    return str_text


def whitespace_ratio(text):
    """
    Description:
        Calculate ratio of whitespace in text
    Args:
        text: input string.
    Returns:
        whitespace ratio: float.
    """

    if len(text) == 0:
        return 0
    return len(text.split(" ")) / len(text)


def clean_whitespace(str_text):
    """
    Description:
        Clean whitespace in text if whitespace ratio is greater than 0.3
    Args:
        str_text: input string.
    Returns:
        text4: clean string.
    """
    # Not reccommend for working with table which requires duplicated

    if whitespace_ratio(str_text) > 0.3:
        text1 = re.sub(r"\s{2,}", "<WS>", str_text)
        text2 = re.sub(r"(?<=[A-z()])\s(?=[A-z()])", "<WS>", text1)
        text3 = re.sub(" ", "", text2)
        text4 = re.sub("<WS>", " ", text3)

    else:
        text2 = re.sub(r"(?<=[A-z()])\s(?=[A-z()])", "<WS>", str_text)
        text3 = re.sub(r"\s{2,}", " ", text2)
        text4 = re.sub("<WS>", " ", text3)

    return text4


def get_clean_json_tables(pdf_file, text_rule_file):
    """
    Description:
        Get json table from pdf file.
        Detectable table must have horizontal border line.
        Require pdf_table2json lib.
    Args:
        pdf_file: string of pdf file path.
        text_rule_file: string of file path to pdf_correction_rules.txt.
    Returns:
        result: list of dict.
    """

    dict_list = converter.main(pdf_file, json_file_out=False, image_file_out=False)
    result = eval(clean_by_rules(dict_list, text_rule_file))
    return result


def add_blank_fill(ls, max):
    """
    Description:
        Add blank member to list till length of list
        is equal to max
    Args:
        ls: list.
        max: int.
    Returns:
        ls: list with length of max.
    """

    while len(ls) < max:
        ls.append("")
    return ls


def get_clean_df(dict_ls):
    """
    Description:
        Create dataframe from list of dict that read from pdf file.
        Require Pandas.
    Args:
        dict_ls: list of dict.
    Returns:
        clean_df: dataframe.
    """

    max_key_cnt = max([len(dct.keys()) for dct in dict_ls])
    top_column = list(dict_ls[0].keys())
    new_column = add_blank_fill(top_column, max_key_cnt)

    clean_df = []

    for dct in dict_ls:
        if list(dct.keys()) == top_column:
            if len(clean_df) == 0:
                add_values = add_blank_fill(list(dct.values()), max_key_cnt)
                clean_df = pd.DataFrame([add_values], columns=new_column)
            else:
                clean_df.loc[len(clean_df.index)] = add_blank_fill(
                    list(dct.values()), max_key_cnt
                )
            prev_key = dct.keys()
        else:
            if len(dct.keys()) == 1:
                clean_df.loc[len(clean_df.index)] = add_blank_fill(
                    list(dct.values())[0], max_key_cnt
                )
            elif dct.keys() != prev_key:
                clean_df.loc[len(clean_df.index)] = add_blank_fill(
                    list(dct.keys()), max_key_cnt
                )
                clean_df.loc[len(clean_df.index)] = add_blank_fill(
                    list(dct.values()), max_key_cnt
                )
                prev_key = dct.keys()

            else:
                clean_df.loc[len(clean_df.index)] = add_blank_fill(
                    list(dct.values()), max_key_cnt
                )

    return clean_df


def most_similar_fuzzy(search_text, source_text):
    """
    Description:
        Get the most similar search-text from source-text
        Require difflib.
    Args:
        search_text: string of search word.
        source_text: string of source to lookup
    Returns:
        dict: {match content : match line number}.
    """

    lines = source_text.split("\n")
    max_sim = 0
    for i, line in enumerate(lines):
        words = line.split()

        for word in words:
            similarity = SequenceMatcher(None, word, search_text)
            if similarity.ratio() > max_sim:
                max_sim = similarity.ratio()
                match_line = i

    return {lines[match_line]: match_line}
