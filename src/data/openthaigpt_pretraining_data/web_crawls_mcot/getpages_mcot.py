import time
import locale
from bs4 import BeautifulSoup
from openthaigpt_pretraining_data.web_crawls_mcot.crawl_mcot import (
    get_response_with_retry,
)

A_TAG = "a"
LI_TAG = "li"
DIV_TAG = "div"
HREF_CLASS = "menu-main-menu-container"
PAGE_CLASS = "content-pagination"

ROOT = "https://tna.mcot.net"


UNWANTED_HREF = [
    ("menu-item-479986"),
    ("menu-item-459680"),
    ("menu-item-482036"),
    ("menu-item-859455"),
]


def get_main_href(text_response):
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
    info = soup.find(DIV_TAG, class_=HREF_CLASS)

    for data in info:
        for class_id in UNWANTED_HREF:
            unwanted = info.find_all(LI_TAG, id=class_id)
            for data in unwanted:
                data.extract()

        hrefs = [
            link["href"]
            for link in info.find_all("a", href=True)
            if "#" not in link["href"]
        ]
        href_list = [
            f"{ROOT}{href}" if href.startswith("/") else href for href in hrefs
        ]

    return href_list


def page_number_response(text_response):
    """
    Description:
        Fetch number of pages from the given response text.
    Args:
        response_text: The HTML response text to extract page number from.
    Returns:
        page_number: The page number extracted from the response.
    """
    soup = BeautifulSoup(text_response, "html.parser")
    pagination_div = soup.find_all(DIV_TAG, class_=PAGE_CLASS)

    if pagination_div:
        try:
            last_page_div = pagination_div[-1]
            last_page_elements = last_page_div.find_all(A_TAG)
            last_page_element = last_page_elements[-2]
            last_page_text = last_page_element.get_text(strip=True)

            # Set locale to handle numbers with comma as decimal separator
            locale.setlocale(locale.LC_NUMERIC, "")
            last_page_number = locale.atof(last_page_text)

            return int(last_page_number)
        except IndexError:
            return 1  # Set last_page_number to 1 when there are no pagination links
    else:
        return 0  # Set last_page_number to 0 when there is no pagination div


def get_page_no(href_list, time_delay):
    """
    Description:
        Fetch number of pages from each href.
    Args:
        list of href.
        time_delay: Delay in seconds after each request.
    Returns:
        page_numbers: A list containing page numbers.
    """
    page_numbers = []

    for href in href_list:
        print(f"Processing: {href}")
        
        res = get_response_with_retry(href)
        res.encoding = "utf-8"

        if res.status_code == 200:
            page_number = page_number_response(res.text)
            page_numbers.append(page_number)

        time.sleep(time_delay)

    return page_numbers
