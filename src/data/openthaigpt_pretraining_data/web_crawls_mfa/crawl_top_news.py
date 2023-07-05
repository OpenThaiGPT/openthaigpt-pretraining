import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

ROOT = 'https://www.mfa.go.th'
TOP_STORIES_URL = 'https://www.mfa.go.th/th/page/%E0%B8%82%E0%B9%88%E0%B8%B2%E0%B8%A7%E0%B9%80%E0%B8%94%E0%B9%88%E0%B8%99?menu=5d5bd3d815e39c306002aac4'
UNWANTED_CLASSES = [('div','d-inline-block'),('div','pt-3 col')]
DIV_TAG = 'div'
P_TAG = 'p'
A_TAG = 'a'
DATE_CLASS = 'date'
INFO_CLASS = 'p-3 col-md-4'
DETAIL_CLASS = 'ContentDetailstyled__ContentDescription-sc-150bmwg-4 jWrYsI mb-3'

def process_title_date(response):
    """
    Description:
        process titles and dates for news on every page.
    Args:
        response
    Returns:
        news_list: A list containing titles and dates.
    """
    soup = BeautifulSoup(response.content, 'lxml')
    info = soup.find_all(DIV_TAG, class_=INFO_CLASS)
    date_list = soup.find_all(P_TAG, class_=DATE_CLASS)

    news_list = []

    # Exclude unrelated data
    for inf in info:
        for tag_name, class_attributes in UNWANTED_CLASSES:
            unwanted_data = soup.find_all(tag_name, class_=class_attributes)
            for data in unwanted_data:
                data.extract()

        # Get news titles
        title = inf.get_text(strip=True, separator=' ')

        # Get news dates
        for indiv_date in date_list:
            indiv_date = indiv_date.get_text(strip=True, separator=' ')

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
        url = f'{cur_url}&p={page}'
        res = requests.get(url)
        res.encoding = "utf-8"

        if res.status_code == 200:
            processed_data = process_title_date(res)
            news_list.extend(processed_data)

        time.sleep(3)

    return news_list

def process_info(response):
  info_list = []
  href_list = []

  soup = BeautifulSoup(response.content, 'lxml')
  info = soup.find_all(DIV_TAG, class_=INFO_CLASS)

  for branch in info:
      link = branch.find(A_TAG)
      if link:
        href_list.append(link['href'])

  for href in href_list:
      result = requests.get(f'{ROOT}{href}')
      content = result.text
      soup = BeautifulSoup(content, 'lxml')   

      details = soup.find_all(DIV_TAG, class_=DETAIL_CLASS)
      for element in details:
          detail = element.get_text(strip=True, separator=' ')
          info_list.append(detail)     

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
      url = f'{cur_url}&p={page}'
      res = requests.get(url)
      res.encoding = "utf-8"

      if res.status_code == 200:
            processed_info = process_info(res)
            info_list.extend(processed_info)

      time.sleep(3)

    return info_list

news_title_date = get_title_date(cur_url = TOP_STORIES_URL,page_no = 216)
news_details = get_info(cur_url = TOP_STORIES_URL,page_no = 216)

for i, data_dict in enumerate(news_title_date):
    if i < len(news_details):
        data_dict.update({'detail': news_details[i]})
        
all_news = pd.DataFrame(news_title_date)