import pandas as pd
import os
import json
import re
from FedMins import FederalReserveMins


# Remove participant name and normalize

def normalize_the_content(text_content):
    """
    this function clean the input by normalize the tabs and newline

    Parameters:
    text_content[str] -> input text that need to be clean

    Outputs:
    normalize_text[str] -> the text that was processed
    """
    normalize_text = re.sub(r' +', ' ', text_content)

    normalize_text = normalize_text.replace("\r\n\r", "\n")
    normalize_text = normalize_text.replace("\n\n", "\n")
    normalize_text = normalize_text.replace("\t\t", "\t")

    return normalize_text

# Save to jsonl


def convert_to_jsonl(file_path, dataset):
    """
    Save a list of text strings to a JSON Lines (JSONL) file.

    Parameters:
    - file_path[str]: The path to the output JSONL file.
    - dataset[DataFrame]: the dataframe that we want to covert to jsonl file

    Output:
    jsonl file save in the specified path
    """

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    for i, r in dataset.iterrows():
        file_name = os.path.join(file_path,
                                 f"fomc_{str(i).replace('-', '')[:8]}.jsonl")

        json_data = {'date-time': str(i),
                    'content': r['Text_normalize']}

        with open(file_name, "w") as file:
            file.write(json.dumps(json_data, ensure_ascii=True) + '\n')


if __name__ == "__main__":
    # Fedtools

    fed_mins = FederalReserveMins(
        main_url='https://www.federalreserve.gov',
        calendar_url='https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm',
        start_year=1990,
        historical_split=2014,
        verbose=True,
        thread_num=10)

    dataset = fed_mins.find_minutes()
    norm_text = [normalize_the_content(text) for text in dataset['Federal_Reserve_Mins']]

    dataset['Text_normalize'] = norm_text

    convert_to_jsonl("clean_scrape_fomc", dataset)
