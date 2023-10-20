from bs4 import BeautifulSoup
import requests
import pandas as pd
from tqdm.auto import tqdm

import re


def get_soup_by_url(url, if_verify=True):
    """
    Request.get url, for untrustes ssl: if_verify = False
    Return set , if succeess request -> (True, soup text), else --> (false, status code)
    """
    res = requests.get(
        url, verify=if_verify
    )  # requested url got untrusted SSL certificate have to ignore by verify = False
    res.encoding = "utf-8"

    if res.status_code == 200:
        return (True, BeautifulSoup(res.text, "html.parser"))
    else:
        print(res.status_code)
        return (False, res.status_code)


def get_soup_dict(soup_in):
    """
    get_soup_dict(soup_in) -> dict
    get contents in class = "panel-body" and extract as dict

    soup_in is soup in final url that contains detailed target
      (aka actual individual news)
    """
    # Extract body contents
    soup_in_pbody = soup_in.find_all("div", class_="panel-body")
    cur_pbody = soup_in_pbody[0]

    # Extract text on desired element
    cur_head = cur_pbody.find("div", class_="panel-heading clearfix Circular")
    cur_hidden = cur_pbody.find("p", class_="col-xs-8 remove-xs color7")
    cur_date = cur_pbody.find(
        "div", class_="col-xs-12 col-sm-6 col-md-5 news-2 font_level3 text-right"
    )
    if cur_date is not None:
        cur_date_clean = cur_date.text.strip()[: cur_date.text.strip().find("\n")]
    else:
        cur_date_clean = None

    cur_h3 = cur_pbody.h3
    cur_title = cur_pbody.find("p", class_="font_level2 Circular color3")
    cur_detail = cur_pbody.find("div", class_="col-xs-12 padding-sm1 news-2 Circular")

    # Create dict
    cur_content = {
        "panel_heading": cur_head.text.strip(),
        "hidden_date": cur_hidden.text.strip(),
        "date": cur_date_clean,
        "h3": cur_h3.text.strip(),
        "title": cur_title.text.strip(),
        "detail": cur_detail.text.strip(),
    }

    return cur_content


requests.packages.urllib3.disable_warnings(
    requests.packages.urllib3.exceptions.InsecureRequestWarning
)

# Get list for crawling, extract only the site with news_format contents
site_map = pd.read_csv("src/data/scripts/crawl_thaigov/Thaigov_crawl_lists.csv")
crawl_list = site_map.loc[site_map["news_format"] == "y", ["href"]]

# Create df for receiving data
content_news = pd.DataFrame(
    {
        "panel_heading": [],
        "url": [],
        "hidden_date": [],
        "date": [],
        "h3": [],
        "title": [],
        "detail": [],
    }
)
error_urls = pd.DataFrame({"url": [], "error_code": []})

for list in crawl_list.href:
    url_news_main = list
    # get soup (html content)
    soup = get_soup_by_url(url_news_main, False)

    # get contents from first page of url_news_main (No ?per_page=)
    if soup[0] is True:
        # list all 10 url of news content on each per_page
        soup_pbody = soup[1].find_all("div", class_="panel-body")
        soup_a = soup_pbody[0].find_all("a")

        # get soup_dict of each url in page
        for a in soup_a:
            soup_in_url = a.get("href")
            soup_in = get_soup_by_url(soup_in_url, False)
            if soup_in[0] is True:
                cur_dict = get_soup_dict(soup_in[1])
                cur_dict["url"] = soup_in_url
                # Add to content_news dataframe
                content_news.loc[len(content_news)] = cur_dict
            else:
                error_urls.loc[len(error_urls)] = {
                    "url": soup_in_url,
                    "error_code": soup_in[1],
                }

        # get number of max page for page looping
        soup_page = soup[1].find_all("ul", class_=re.compile("^pagination color"))
        soup_page_li = soup_page[0].find_all("li")
        soup_page_li = sorted(
            soup_page_li,
            key=lambda x: int(x.a["data-ci-pagination-page"])
            if "data-ci-pagination-page" in x.a.attrs
            else 0,
        )
        max_page = int(soup_page_li[-1].a["data-ci-pagination-page"])

        # Start looping next page
        url_get_page = url_news_main + "?per_page="

        for i in tqdm(range(1, max_page), total=max_page - 1):
            url_page = url_get_page + str(
                i * 10
            )  # req per_page is to load 10 contents on each page
            soup = get_soup_by_url(url_page, False)
            print(url_page)

            if soup[0] is True:
                soup_pbody = soup[1].find_all("div", class_="panel-body")
                soup_a = soup_pbody[0].find_all(
                    "a"
                )  # list all 10 url of news content on each per_page

                # get soup_dict of each url in page
                for a in soup_a:
                    soup_in_url = a.get("href")
                    soup_in = get_soup_by_url(soup_in_url, False)
                    if soup_in[0] is True:
                        cur_dict = get_soup_dict(soup_in[1])
                        cur_dict["url"] = soup_in_url
                        content_news.loc[len(content_news)] = cur_dict
                    else:
                        error_urls.loc[len(error_urls)] = {
                            "url": soup_in_url,
                            "error_code": soup_in[1],
                        }
            else:
                error_urls.loc[len(error_urls)] = {
                    "url": url_page,
                    "error_code": soup[1],
                }

    else:
        # print("URL error: "+url)
        error_urls.loc[len(error_urls)] = {"url": url_news_main, "error_code": soup[1]}


content_news.to_csv("test_crawl_lists.csv", index=False)
# error_urls
