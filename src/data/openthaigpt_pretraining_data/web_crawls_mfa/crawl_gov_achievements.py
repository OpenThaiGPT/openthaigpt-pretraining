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


def process_response(text_response):
    """
    Description:
        process titles and dates for news on every page.
    Args:
        response
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
        indiv_date = date_list.get_text(strip=True, separator=" ")

        news_dict = {"title": title, "date": indiv_date}
        news_list.append(news_dict)

    return news_list


def get_title_date(cur_url, time_delay):
    """
    Description:
        Get data processed by the function process_response.
    Args:
        cur_url: The desired URL to be used as a root.
        time_delay: Delay before another request (in second).
    Returns:
        news_list: A list containing titles and dates.
    """
    news_list = []

    res = requests.get(cur_url)
    res.encoding = "utf-8"

    if res.status_code == 200:
        processed_data = process_response(res.text)
        news_list.extend(processed_data)

    time.sleep(time_delay)

    return news_list


def get_href(text_response):
    """
    Description:
        fetch href from text response.
    Args:
        response.
    Returns:
        href_list: A list containing href.
    """
    href_list = []

    soup = BeautifulSoup(text_response, "lxml")
    info = soup.find_all(DIV_TAG, class_=INFO_CLASS)

    for branch in info:
        link = branch.find(A_TAG)
        if link:
            href_list.append(link["href"])

    return href_list


def href_info(text_response):
    """
    Description:
        fetch news details.
    Args:
        response.
    Returns:
        info_list: A list containing news details.
    """
    info_list = []

    soup = BeautifulSoup(text_response, "lxml")
    details = soup.find_all(DIV_TAG, class_=DETAIL_CLASS)
    for element in details:
        detail = element.get_text(strip=True, separator=" ")
        info_list.append(detail)

    return info_list


def process_info(text_response, time_delay):
    info_list = []
    href_list = get_href(text_response)

    for href in href_list:
        res = requests.get(f"{ROOT}{href}")

        info = href_info(res.text)
        info_list.extend(info)

        time.sleep(time_delay)

    return info_list


def get_info(cur_url, time_delay):
    """
    Description:
        get data inside a link for every page.
    Args:
        desired url and time delay.
    Returns:
        info_list contains details of the news.
    """
    info_list = []

    res = requests.get(cur_url)
    res.encoding = "utf-8"

    if res.status_code == 200:
        processed_data = process_info(res.text, time_delay)
        info_list.extend(processed_data)

    time.sleep(time_delay)

    return info_list
