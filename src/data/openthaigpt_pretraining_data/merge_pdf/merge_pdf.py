import os
import pandas as pd
import PyPDF2
import pdf_table2json.converter as converter
import re
import numpy as np
import statistics
import math


def convert_correction_rules(filepath):
    """
    This function is to convert pdf_correction_rules.txt to
    list of tuples (wrong, right).
    This list will be used for correcting pdf-converted text.
    Output: list of tuples.
    """

    with open(filepath) as f:
        lines = f.readlines()

    replace_list = [
        eval(t.strip().replace("$line = str_replace", "").replace(",$line", "")[:-1])
        for t in lines
    ]
    return replace_list


def read_pdf_all_pages(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    page_count = len(reader.pages)
    read_text = [reader.pages[i].extract_text() for i in range(page_count)]
    read_text = "".join(map(str, read_text))
    return read_text
