import pandas as pd
from openthaigpt_pretraining_data.web_crawls_mfa.crawl_news import (
    get_title_date,
    get_info,
)

TOP_STORIES_URL = "https://www.mfa.go.th/th/page/%E0%B8%82%E0%B9%88%E0%B8%B2%E0%B8%A7%E0%B9%80%E0%B8%94%E0%B9%88%E0%B8%99?menu=5d5bd3d815e39c306002aac4"

news_title_date = get_title_date(cur_url=TOP_STORIES_URL, page_no=216)
news_details = get_info(cur_url=TOP_STORIES_URL, page_no=216)

for i, data_dict in enumerate(news_title_date):
    if i < len(news_details):
        data_dict.update({"detail": news_details[i]})

all_news = pd.DataFrame(news_title_date)
all_news.to_csv("MFA_top_news.csv", index=False)
