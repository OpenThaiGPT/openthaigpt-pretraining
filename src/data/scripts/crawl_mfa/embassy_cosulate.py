import pandas as pd
from openthaigpt_pretraining_data.web_crawls_mfa.crawl_news import (
    get_title_date,
    get_info,
)

EMBASSY_CONSULATE_URL = "https://www.mfa.go.th/th/page/%E0%B8%82%E0%B9%88%E0%B8%B2%E0%B8%A7%E0%B8%81%E0%B8%B4%E0%B8%88%E0%B8%81%E0%B8%A3%E0%B8%A3%E0%B8%A1%E0%B8%AA%E0%B8%96%E0%B8%B2%E0%B8%99%E0%B9%80%E0%B8%AD%E0%B8%81%E0%B8%AD%E0%B8%B1%E0%B8%84%E0%B8%A3%E0%B8%A3%E0%B8%B2%E0%B8%8A%E0%B8%97%E0%B8%B9%E0%B8%95%E0%B9%81%E0%B8%A5%E0%B8%B0%E0%B8%AA%E0%B8%96%E0%B8%B2%E0%B8%99%E0%B8%81%E0%B8%87%E0%B8%AA%E0%B8%B8%E0%B8%A5%E0%B9%83%E0%B8%AB%E0%B8%8D%E0%B9%88?menu=5f2110a3c1d7dc1b17651cb2"

news_title_date = get_title_date(cur_url=EMBASSY_CONSULATE_URL, page_no=501)
news_details = get_info(cur_url=EMBASSY_CONSULATE_URL, page_no=501)

for i, data_dict in enumerate(news_title_date):
    if i < len(news_details):
        data_dict.update({"detail": news_details[i]})

all_news = pd.DataFrame(news_title_date)
all_news.to_csv("MFA_top_news.csv", index=False)
