import re
from pythainlp.util import countthai
from .keywords import (
    DOCUMENT_REMOVAL_KEYWORDS,
    PARTIAL_REMOVAL_KEYWORDS,
    TH_MONTHS,
    CODE_SPECIAL_CHARACTERS,
)
from typing import List, Dict
from datetime import datetime


def contains_document_removal_keywords(text: str) -> bool:
    """
    Description : Check if an input document contains any document removal keywords.
    Input text : An input document.
    Output : True if the document contains the keywords. Otherwise, False
    """

    pattern = "|".join(DOCUMENT_REMOVAL_KEYWORDS)

    return bool(re.search(pattern, text))


def check_ratio_bad_substring(text: str) -> bool:
    """
    Description : Check if the ratio between number of keywords and length of a document
                  is exceeds the threshold for each groups.

                  Group #1 : Name of months in Thai including abbreviations.
                  Group #2 : Special char that usually found in the code section.
                  Group #3 : Space.
                  Group #4 : Commar.

                  Note : Thresholds of each group are from the experiment on oscar.

    Input text : An input document.
    Output : True if a ratio of at least 1 group is above . Otherwise, False
    """

    n = len(text)

    if len(re.findall("|".join(TH_MONTHS), text)) / n > 0.015:
        return True

    if len(re.findall("|".join(CODE_SPECIAL_CHARACTERS), text)) / n > 0.075:
        return True

    if len(re.findall(" ", text)) / n > 0.13:
        return True

    if len(re.findall(",", text)) / n > 0.05:
        return True
    return False


def remove_partial_keywords(text: str) -> str:
    """
    Description : Remove partial removal keywords from the document.

    Input text : An input document.
    Output : A document after removed keywords.
    """

    return re.sub("|".join(PARTIAL_REMOVAL_KEYWORDS), "", text)


def clean_text(text: str) -> str:
    """
    Description : Clean an input document by these steps

                  1. Remove the whole document if
                    1.1. Contains any document removal keywords (ex. porn, gamble)
                    1.2. Contains too much TH months, code character, space and commar.
                    1.3. The percent of thai characters is less than 50%.
                  2. Remove partial removal keywords.

    Input text : An input document.
    Output : A clean document ("" if the whole document should be removed).
    """

    if (
        len(text) == 0
        or contains_document_removal_keywords(text)
        or check_ratio_bad_substring(text)
        or countthai(text) < 50
    ):
        return ""

    text = remove_partial_keywords(text).strip()

    return text


def clean_dataset(dataset: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Description : Call function clean_text to process the whole dataset.
    Input text : An input dataset having each element as a document in the dataset.
    Output : A clean dataset.
    """
    for i, data_point in enumerate(dataset):
        cleaned_text = clean_text(data_point["text"])
        if cleaned_text != dataset[i]["text"]:
            dataset[i]["text"] = cleaned_text
            dataset[i]["updated_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return [data_point for data_point in dataset if data_point["text"] != ""]
