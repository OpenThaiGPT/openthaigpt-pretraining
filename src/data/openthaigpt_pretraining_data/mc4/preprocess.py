# flake8: noqa:
from .pattern import (
    TOOLARGE_RE,
    NONECHAR_RE,
    NONE_TONE_MARK_RE,
    GAMBLE_RE,
    FOOTBALL_RE,
    HOTEL_AD_RE,
    SALE_URL_RE,
    SALE_SKIP_RE,
    SALE_RE,
    RENT_SKIP_RE,
    RENT_RE,
    JSON_RE,
    SCRIPT_RE,
    GARBAGE_RE,
    GHOST_RE,
    HEX_RE,
    PAGE_RE,
    EMBEDDED_SERVER_RE,
    U_RE,
    EMAIL_RE,
    URL_RE,
    MENU1_RE,
    MENU2_RE,
    MENU3_RE,
    MENU4_RE,
    SIDEBAR_RE,
    BLOCK_RE,
    HASHTAG_RE,
    MARKUP_RE,
    IFRAME_RE,
    IP_RE,
    TEL_RE,
    DATE1_RE,
    DATE2_RE,
    HTML_RE,
    REFINE1_RE,
    REFINE2_RE,
    REFINE3_RE,
    REFINE4_RE,
    REFINE5_RE,
    REFINE6_RE,
    REFINE7_RE,
    REFINE8_RE,
    REFINE9_RE,
    REFINE10_RE,
    REFINE11_RE,
    REFINE12_RE,
    REFINE13_RE,
    REFINE14_RE,
)
from datetime import datetime
from typing import List, Dict
import re


def clean_with_remove_document(text: str) -> bool:
    # ---- Clean too large unused lines
    # Limit matches list to 2 items only, enough
    matches = TOOLARGE_RE.findall(text)[:2]
    # Classify as toolarge row if number of matches = 2
    if len(matches) == 2:
        return True

    # ---- Clean none characters row
    # Limit matches list to 25 items
    matches = NONECHAR_RE.findall(text)[:25]
    # Classify as none character row if number of matches = 25
    if len(matches) == 25:
        return True

    # ---- Clean none tone mark row
    # Limit matches list to 25 items
    matches = NONE_TONE_MARK_RE.findall(text)[:25]
    # Classify as none tone mark row if number of matches = 25
    if len(matches) == 25:
        return True

    # ---- Clean Gamble ~ 9.2% of mC4 data
    # if found gamble word 2 times in a row, classify as gamble row
    # remove the row
    # Limit matches list to 2 items only, enough
    matches = GAMBLE_RE.findall(text)[:2]
    # Classify as gamble if number of matches = 2
    if len(matches) == 2:
        return True

    # ---- Clean Football data
    # if found gamble word 4 times in a row, classify as football data
    # remove the row
    # Limit matches list to 4 items only
    matches = FOOTBALL_RE.findall(text)[:4]
    if len(matches) == 4:
        return True

    # ---- Clean Hotel Advertising
    # if found hotel word 4 times in a row, classify as Hotel Ad. data
    # remove the row
    # Limit matches list to 4 items only, enough
    matches = HOTEL_AD_RE.findall(text)[:4]
    if len(matches) == 4:
        return True

    # ----  Clean Sale ~26% of mC4 data
    # Sale row data is diverse,
    # so the regex is not used in this case.
    # Rules:
    # 1. Remove row if it contains common specific Sale's URL
    # 2. Skip to next clean rule if it contains specific keywords, eg. "สอบราคา", "จัดซื้อจัดจ้าง, etc."
    # 3. If not found keywords in (2) then scan the row with sale keywords, if there are at leat 3 sale kewords found then remove the row.

    if SALE_URL_RE.search(text):
        return True

    if not SALE_SKIP_RE.search(text):
        # Classify as Sale data ( 3 matches, can be adjusted)
        matches = SALE_RE.findall(text)[:3]
        if len(matches) == 3:
            return True

    # ---- Clean Rent (พวกเช่า ~2% of mC4 data)
    # Rent use another rules
    # 1. find skip words in the row. If found, skip to next rule (not remove)
    # 2. if found rent word 2 times in a row, classify as rent row
    #    remove the row

    if not RENT_SKIP_RE.search(text):
        # Limit matches list to 2 items only, enough
        matches = RENT_RE.findall(text)[:2]
        if len(matches) == 2:
            return True

    # ---- Clean pattern (json like -> "abc": ~.5-1% )
    # 99% can classify as gabage: so remove them
    # match n items to make sure they are garbages n=20, can change
    matches = JSON_RE.findall(text)[:20]
    # if match only 20+, classify as garbage
    if len(matches) == 20:
        return True

    # ---- Clean script (Javascript, etc. ~.5% )
    # 99% can classify as gabage: so remove them
    matches = SCRIPT_RE.findall(text)[:10]
    # Classify as script if number of matches = 10
    if len(matches) == 10:
        return True

    # ---- Clean garbage (useless or not necessary ~.45%)
    # classify as gabage: so remove them
    matches = GARBAGE_RE.findall(text)[:4]
    # Classify as garbage if number of matches = 4
    if len(matches) == 4:
        return True

    # ---- Clean ghost language (~0.008% can cancel this clean)
    # classify as ghost : so remove them
    matches = GHOST_RE.findall(text)[:4]
    # Classify as ghost if number of matches = 4
    if len(matches) == 4:
        return True

    # ---- Clean HEX code
    # classify as HEX : so remove them
    matches = HEX_RE.findall(text)[:25]
    # Classify as HEX if number of matches = 25
    if len(matches) == 25:
        return True

    return False


def clean_text(text: str) -> str:
    text = PAGE_RE.sub(" ", text)
    text = EMBEDDED_SERVER_RE.sub(" ", text)
    text = U_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = MENU1_RE.sub(" ", text)
    text = MENU2_RE.sub(" ", text)
    text = MENU3_RE.sub(" ", text)
    text = MENU4_RE.sub(" ", text)
    text = SIDEBAR_RE.sub(" ", text)
    text = BLOCK_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    text = MARKUP_RE.sub(" ", text)
    text = IFRAME_RE.sub(" ", text)
    text = IP_RE.sub(" ", text)
    text = TEL_RE.sub(" ", text)
    text = DATE1_RE.sub(" ", text)
    text = DATE2_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)

    # --- Refinements (in sequence)
    text = REFINE1_RE.sub(" ", text)
    text = REFINE2_RE.sub(" ", text)
    text = REFINE3_RE.sub(" ", text)
    text = REFINE4_RE.sub(" ", text)
    text = REFINE5_RE.sub(" ", text)
    text = REFINE6_RE.sub(" ", text)
    text = REFINE7_RE.sub(" ", text)
    text = REFINE8_RE.sub(" ", text)
    text = REFINE9_RE.sub(" ", text)
    text = REFINE10_RE.sub(" ", text)
    text = REFINE11_RE.sub(" ", text)
    text = REFINE12_RE.sub(" ", text)
    text = REFINE13_RE.sub(" ", text)
    text = REFINE14_RE.sub(" ", text)

    # Split the text into lines and remove any empty lines
    lines = [line for line in text.split("\n") if line]

    # Initialize the list with the first line
    deduplicated_list = [lines[0]]

    # Iterate over the rest of the lines
    for i in range(1, len(lines)):
        # Find the common prefix between this line and the previous line
        common_prefix = ""
        for char1, char2 in zip(lines[i], lines[i - 1]):
            if char1 == char2:
                common_prefix += char1
            else:
                break

        # Remove the common prefix from this line and add it to the list
        deduplicated_list.append(lines[i][len(common_prefix) :])

    text = "\n".join(deduplicated_list)

    # Clean short lines
    # ( len(line) <= 30 characters , cut this line off)
    text = "\n".join(line for line in text.split("\n") if len(line) > 30)

    # ---- The scan row that passes all filter is written to disk
    # before write to disk, get rid of spaces by change them to single space (' ').

    text = re.sub("[ ]+", " ", text, 0, re.MULTILINE)
    text = re.sub("^[ ]", "", text, 0, re.MULTILINE)
    text = re.sub(r"\n\s*", "\n", text, 0, re.MULTILINE)

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
