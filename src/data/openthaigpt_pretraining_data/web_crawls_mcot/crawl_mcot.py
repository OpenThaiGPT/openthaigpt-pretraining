import requests
from bs4 import BeautifulSoup
import time
import random

A_TAG = 'a'
H2_TAG = 'h2'
DIV_TAG = 'div'
HEADER_TAG = 'header'
HEADER_CLASS = 'entry-header'
TITLE_CLASS = 'entry-title'
CONTENT_CLASS = 'entry-content'
TIME_CLASS = 'time'

UNWANTED_CLASSES = [('social-box'),
                    ('entry-author'),
                    ('entry-tag'),
                    ('related-posts'),
                    ('readmore-box')]

UNWANTED_HREF = [('menu-item-479986'),
                 ('menu-item-459680'),
                 ('menu-item-482036'),
                 ('menu-item-859455')]

user_agents_list = [
    'Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.83 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4844.77 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_5_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.83 Safari/537.36',
    'Mozilla/5.0 (Linux; Android 11; SM-G975U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Mobile Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
]


def get_response_with_retry(url):
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # Delay in seconds
       
    for retries in range(MAX_RETRIES):
        try:
            res = requests.get(url, headers={
                'User-Agent': random.choice(user_agents_list)})
            if res is None:
                continue

            if res.status_code != 403:
                return res
        except requests.exceptions.RequestException:
            pass
               
        time.sleep(RETRY_DELAY)
    
    return None


def process_title_date(text_response):
    """
    Description:
        process titles and dates for news on every page.
    Args:
        response
    Returns:
        news_list: A list containing titles and dates.
    """
    news_list = []

    soup = BeautifulSoup(text_response, 'lxml')
    info = soup.find_all(HEADER_TAG, class_=HEADER_CLASS)

    for inf in info:
        title_list = inf.find(H2_TAG, class_=TITLE_CLASS)
        title = title_list.get_text(strip=True)

        date_list = inf.find(DIV_TAG, class_=TIME_CLASS)
        indiv_date = date_list.get_text(strip=True, separator=" ").split()[0]

        news_dict = {"title": title, "date": indiv_date}
        news_list.append(news_dict)

    return news_list


def get_title_date(cur_url, page_no, time_delay):
    """
    Description:
        Get data processed by the function process_response.
    Args:
        cur_url: The desired URL to be used as a root.
        time_delay: Delay before another request (in seconds).
    Returns:
        news_list: A list containing titles and dates.
    """

    info_list = []

    if page_no > 0:
        if page_no == 1:
            res = get_response_with_retry(cur_url)
            res.encoding = "utf-8"

            if res.status_code == 200:
                processed_data = process_title_date(res.text)
                info_list.extend(processed_data)

        else:
            for page in range(1, page_no + 1):
                url = f"{cur_url}/page/{str(page)}"
                res = get_response_with_retry(url)
                res.encoding = "utf-8"

                if res.status_code == 200:
                    processed_data = process_title_date(res.text)
                    info_list.extend(processed_data)

                time.sleep(time_delay)

    return info_list

