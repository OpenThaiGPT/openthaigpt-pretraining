import requests
from bs4 import BeautifulSoup
import pandas as pd

root = 'https://www.mfa.go.th'
top_stories_url = 'https://www.mfa.go.th/th/page/%E0%B8%82%E0%B9%88%E0%B8%B2%E0%B8%A7%E0%B9%80%E0%B8%94%E0%B9%88%E0%B8%99?menu=5d5bd3d815e39c306002aac4'

def get_title_date(cur_url, page_no):
    """
    Description:
        get titles and dates for news in every pafe
    Args:
        desired url to be used as a root and total of pages.
    Returns:
        news_list contains titles and dates
        """
    page = 1
    titles_list = []
    news_list = []
    date = []
    unwanted_classes = [('div','d-inline-block'),('div','pt-3 col')]

    while page != page_no+1:
      url = f'{cur_url}&p={page}'
      res = requests.get(url)
      res.encoding = "utf-8"
      soup = BeautifulSoup(res.content, 'lxml')

      info = soup.find_all('div', class_='p-3 col-md-4')
      date_list = soup.find_all('p', class_='date')

      # Exclude unrelated data 
      for inf in info:
          for tag_name, class_attributes in unwanted_classes:
                unwanted_data = soup.find_all(tag_name, class_=class_attributes)
                for data in unwanted_data:
                  data.extract()
          # Get news titles
          title = inf.get_text(strip=True, separator=' ')
          titles_list.append(title)
          # Get new dates
          for indiv_date in date_list:
            indiv_date = indiv_date.get_text(strip=True, separator=' ')
            date.append(indiv_date)

          news_dict = {'title': title, 'date': indiv_date}
          news_list.append(news_dict)

      page = page + 1


    return news_list


def get_info(cur_url, page_no):
    """
    Description:
        get data inside a link for every pafe
    Args:
        desired url and total of pages.
    Returns:
        info_list contains details of the news
        """
    page = 1
    info_list = []
    href_list = []

    while page != page_no+1:
      url = f'{cur_url}&p={page}'
      res = requests.get(url)
      res.encoding = "utf-8"
      soup = BeautifulSoup(res.content, 'lxml')

      info = soup.find_all('div', class_='p-3 col-md-4')

      for branch in info:
        link = branch.find('a')
        if link:
          href_list.append(link['href'])

      for href in href_list:
        result = requests.get(f'{root}{href}')
        content = result.text
        soup = BeautifulSoup(content, 'lxml')   

        details = soup.find_all('div', class_='ContentDetailstyled__ContentDescription-sc-150bmwg-4 jWrYsI mb-3')
        for element in details:
          detail = element.get_text(strip=True, separator=' ')
          info_list.append(detail)       

      page = page + 1

    return info_list

news_title_date = get_title_date(top_stories_url,1)
news_details = get_info(top_stories_url,1)
    
for i, data_dict in enumerate(news_title_date):
    if i < len(news_details):
        data_dict.update({'detail': news_details[i]})
        
all_news = pd.DataFrame(news_title_date)