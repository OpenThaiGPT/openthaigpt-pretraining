import requests
from bs4 import BeautifulSoup
import time

ROOT = "https://www.mfa.go.th"
DIV_TAG = "div"
P_TAG = "p"
A_TAG = "a"
DATE_CLASS = "date"
INFO_CLASS = "p-3 col-md-4"
DETAIL_CLASS = "ContentDetailstyled__ContentDescription-sc-150bmwg-4 jWrYsI mb-3"


def process_title_date(text_response):
    """
    Description:
        process titles and dates for news on every page.
    Args:
        text_response: Text response containing HTML content.
    Returns:
        news_list: A list containing titles and dates.
    """
    news_list = []
    
    soup = BeautifulSoup(text_response, "lxml")
    info = soup.find_all(DIV_TAG, class_=INFO_CLASS)

    for inf in info:

        # Get news titles
        title_list = inf.find(attrs={"title": True})
        title = title_list.get_text(strip=True)

        # Get news dates
        date_list = inf.find(P_TAG, class_=DATE_CLASS)
        indiv_date = date_list.get_text(strip=True, separator=' ')

        news_dict = {'title': title, 'date': indiv_date}
        news_list.append(news_dict)

    return news_list


def get_title_date(cur_url, page_no):
    """
    Description:
        Get data processed by the function process_response.
    Args:
        cur_url: The desired URL to be used as a root.
        page_no: The total number of pages.
    Returns:
        news_list: A list containing titles and dates.
    """
    news_list = []

    for page in range(1, page_no + 1):
        url = f"{cur_url}&p={page}"
        res = requests.get(url)
        res.encoding = "utf-8"

        if res.status_code == 200:
            processed_data = process_title_date(res.text)
            news_list.extend(processed_data)

        time.sleep(0.5)

    return news_list


def process_info(text_response):
    info_list = []
    href_list = []

    soup = BeautifulSoup(text_response, "lxml")
    info = soup.find_all(DIV_TAG, class_=INFO_CLASS)

    for branch in info:
        link = branch.find(A_TAG)
        if link:
            href_list.append(link["href"])

    for href in href_list:
        result = requests.get(f"{ROOT}{href}")
        content = result.text
        soup = BeautifulSoup(content, "lxml")

        details = soup.find_all(DIV_TAG, class_=DETAIL_CLASS)
        for element in details:
            detail = element.get_text(strip=True, separator=" ")
            info_list.append(detail)

        time.sleep(0.5)

    return info_list


def get_info(cur_url, page_no):
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
            processed_info = process_info(res.text)
            info_list.extend(processed_info)

        time.sleep(0.5)

    return info_list
