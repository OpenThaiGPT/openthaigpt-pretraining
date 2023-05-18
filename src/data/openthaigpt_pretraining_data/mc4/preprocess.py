# flake8: noqa
from pattern import (
    toolarge_re,
    nonechar_re,
    none_tone_mark_re,
    gamble_re,
    football_re,
    hotel_ad_re,
    sale_url_re,
    sale_skip_re,
    sale_re,
    rent_skip_re,
    rent_re,
    json_re,
    script_re,
    garbage_re,
    ghost_re,
    url_re,
    menu1_re,
    menu2_re,
    menu3_re,
    menu4_re,
    hashtag_re,
    page_re,
    sidebar_re,
    markup_re,
    embedded_server_re,
    u_re,
    iframe_re,
    block_re,
    email_re,
    ip_re,
    tel_re,
    date1_re,
    date2_re,
    html_re,
    hex_re,
    refine1_re,
    refine2_re,
    refine3_re,
    refine4_re,
    refine5_re,
    refine6_re,
    refine7_re,
    refine8_re,
    refine9_re,
    refine10_re,
    refine11_re,
    refine12_re,
    refine13_re,
    refine14_re,
)
from typing import List
import re


def clean_text(text: str) -> str:

    # ---- Clean too large unused lines
    # Limit matches list to 2 items only, enough
    matches = toolarge_re.findall(text)[:2]
    # Classify as toolarge row if number of matches = 2
    if len(matches) == 2:
        return ""

    # ---- Clean none characters row
    # Limit matches list to 25 items
    matches = nonechar_re.findall(text)[:25]
    # Classify as none character row if number of matches = 25
    if len(matches) == 25:
        return ""

    # ---- Clean none tone mark row
    # Limit matches list to 25 items
    matches = none_tone_mark_re.findall(text)[:25]
    # Classify as none tone mark row if number of matches = 25
    if len(matches) == 25:
        return ""

    # ---- Clean Gamble ~ 9.2% of mC4 data
    # if found gamble word 2 times in a row, classify as gamble row
    # remove the row
    # Limit matches list to 2 items only, enough
    matches = gamble_re.findall(text)[:2]
    # Classify as gamble if number of matches = 2
    if len(matches) == 2:
        return ""

    # ---- Clean Football data
    # if found gamble word 4 times in a row, classify as football data
    # remove the row
    # Limit matches list to 4 items only
    matches = football_re.findall(text)[:4]
    if len(matches) == 4:
        return ""

    # ---- Clean Hotel Advertising
    # if found hotel word 4 times in a row, classify as Hotel Ad. data
    # remove the row
    # Limit matches list to 4 items only, enough
    matches = hotel_ad_re.findall(text)[:4]
    if len(matches) == 4:
        return ""

    # ----  Clean Sale ~26% of mC4 data
    # Sale row data is diverse,
    # so the regex is not used in this case.
    # Rules:
    # 1. Remove row if it contains common specific Sale's URL
    # 2. Skip to next clean rule if it contains specific keywords, eg. "สอบราคา", "จัดซื้อจัดจ้าง, etc."
    # 3. If not found keywords in (2) then scan the row with sale keywords, if there are at leat 3 sale kewords found then remove the row.

    if sale_url_re.search(text):
        return ""

    if not sale_skip_re.search(text):
        # Classify as Sale data ( 3 matches, can be adjusted)
        matches = sale_re.findall(text)[:3]
        if len(matches) == 3:
            return ""

    # ---- Clean Rent (พวกเช่า ~2% of mC4 data)
    # Rent use another rules
    # 1. find skip words in the row. If found, skip to next rule (not remove)
    # 2. if found rent word 2 times in a row, classify as rent row
    #    remove the row

    if not rent_skip_re.search(text):
        # Limit matches list to 2 items only, enough
        matches = rent_re.findall(text)[:2]
        if len(matches) == 2:
            return ""

    # ---- Clean pattern (json like -> "abc": ~.5-1% )
    # 99% can classify as gabage: so remove them
    # match n items to make sure they are garbages n=20, can change
    matches = json_re.findall(text)[:20]
    # if match only 20+, classify as garbage
    if len(matches) == 20:
        return ""

    # ---- Clean script (Javascript, etc. ~.5% )
    # 99% can classify as gabage: so remove them
    matches = script_re.findall(text)[:10]
    # Classify as script if number of matches > 10
    if len(matches) == 10:
        return ""

    # ---- Clean garbage (useless or not necessary ~.45%)
    # classify as gabage: so remove them
    matches = garbage_re.findall(text)[:4]
    # Classify as garbage if number of matches >= 4
    if len(matches) == 4:
        return ""

    # ---- Clean ghost language (~0.008% can cancel this clean)
    # classify as ghost : so remove them
    matches = ghost_re.findall(text)[:4]
    # Classify as ghost if number of matches >= 4
    if len(matches) == 4:
        return ""

    # ---------------------------------------------------------------
    # The row that falls down here is
    # the row that passed all romove filters.
    # Now, we will scan and REPLACE unwanted characters and patterns
    # with ' ' (blank)
    # ---------------------------------------------------------------

    text = url_re.sub(" ", text)
    text = menu1_re.sub(" ", text)
    text = menu2_re.sub(" ", text)
    text = menu3_re.sub(" ", text)
    text = menu4_re.sub(" ", text)
    text = hashtag_re.sub(" ", text)
    text = page_re.sub(" ", text)
    text = sidebar_re.sub(" ", text)
    text = markup_re.sub(" ", text)
    text = embedded_server_re.sub(" ", text)
    text = u_re.sub(" ", text)
    text = iframe_re.sub(" ", text)
    text = block_re.sub(" ", text)
    text = email_re.sub(" ", text)
    text = ip_re.sub(" ", text)
    text = tel_re.sub(" ", text)
    text = date1_re.sub(" ", text)
    text = date2_re.sub(" ", text)
    text = html_re.sub(" ", text)
    text = hex_re.sub(" ", text)

    # --- Refinements (in sequence)
    text = refine1_re.sub(" ", text)
    text = refine2_re.sub(" ", text)
    text = refine3_re.sub(" ", text)
    text = refine4_re.sub(" ", text)
    text = refine5_re.sub(" ", text)
    text = refine6_re.sub(" ", text)
    text = refine7_re.sub(" ", text)
    text = refine8_re.sub(" ", text)
    text = refine9_re.sub(" ", text)
    text = refine10_re.sub(" ", text)
    text = refine11_re.sub(" ", text)
    text = refine12_re.sub(" ", text)
    text = refine13_re.sub(" ", text)
    text = refine14_re.sub(" ", text)

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
    text = re.sub("\n\s*", "\n", text, 0, re.MULTILINE)

    return text


def clean_dataset(dataset: List[str]) -> List[str]:
    """
    Description : Call function clean_text to process the whole dataset.
    Input text : An input dataset having each element as a document in the dataset.
    Output : A clean dataset.
    """

    for i, data_point in enumerate(dataset):
        dataset[i] = clean_text(data_point)

    return [data_point for data_point in dataset if data_point != ""]
