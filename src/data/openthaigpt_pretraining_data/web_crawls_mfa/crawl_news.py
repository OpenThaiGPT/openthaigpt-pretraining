import requests
import time
from openthaigpt_pretraining_data.web_crawls_mfa.crawl_gov_achievements import (
    process_response,
    process_info,
)

ROOT = "https://www.mfa.go.th"
DIV_TAG = "div"
P_TAG = "p"
A_TAG = "a"
DATE_CLASS = "date"
INFO_CLASS = "p-3 col-md-4"
DETAIL_CLASS = "ContentDetailstyled__ContentDescription-sc-150bmwg-4 jWrYsI mb-3"


def get_title_date(cur_url, page_no, time_delay):
    """
    Description:
        Get data processed by the function process_response.
    Args:
        cur_url: The desired URL to be used as a root.
        page_no: The total number of pages.
        time_delay: Delay before another request (in second).
    Returns:
        news_list: A list containing titles and dates.
    """
    news_list = []

    for page in range(1, page_no + 1):
        url = f"{cur_url}&p={page}"
        res = requests.get(url)
        res.encoding = "utf-8"

        if res.status_code == 200:
            processed_data = process_response(res.text, time_delay)
            news_list.extend(processed_data)

        time.sleep(0.5)

    return news_list


def get_info(cur_url, page_no, time_delay):
    """
    Description:
        get data inside a link for every pafe
    Args:
        desired url and total of pages.
    Returns:
        info_list contains details of the news
    """
    info_list = []

    for page in range(1, page_no + 1):
        url = f"{cur_url}&p={page}"
        res = requests.get(url)
        res.encoding = "utf-8"

        if res.status_code == 200:
            processed_info = process_info(res.text, time_delay)
            info_list.extend(processed_info)

        time.sleep(0.5)

    return info_list
