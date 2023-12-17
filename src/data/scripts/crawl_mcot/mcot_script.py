import time
import pandas as pd
from datasets import Dataset, load_from_disk
from openthaigpt_pretraining_data.web_crawls_mcot.crawl_mcot import (
    get_response_with_retry,
    get_title_date,
    get_info,
)

from openthaigpt_pretraining_data.web_crawls_mcot.getpages_mcot import (
    get_main_href,
    get_page_no,
)

ROOT = "https://tna.mcot.net"


def process_news(text_response, time_delay):
    td_list = []
    href_list = get_main_href(text_response)

    for href in href_list:
        page_numbers = get_page_no(href_list, time_delay)

        for page_number in page_numbers:
            title_date = get_title_date(href, page_number, time_delay)
            news_details = get_info(href, page_number, time_delay)

            new_td_list = []
            for i, item in enumerate(title_date):
                if i < len(news_details) and news_details[i] is not None:
                    item["details"] = news_details[i]
                    new_td_list.append(item)

            td_list.extend(new_td_list)

    return td_list


def get_news(cur_url, time_delay):
    info_list = []

    res = get_response_with_retry(cur_url)
    res.encoding = "utf-8"

    if res.status_code == 200:
        info = process_news(res.text, time_delay)
        info_list.extend(info)

    time.sleep(time_delay)

    return info_list


mcot_news = get_news(ROOT, 0.5)
all_news = pd.DataFrame(mcot_news)
dataset = Dataset.from_pandas(all_news)
dataset.save_to_disk("mcot.arrow")
loaded_dataset = load_from_disk("mcot.arrow")
