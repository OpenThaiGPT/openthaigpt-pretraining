import pandas as pd
from openthaigpt_pretraining_data.web_crawls_mfa.crawl_news import (
    get_title_date,
    get_info,
)

OTHER_NEWS_URL = "https://www.mfa.go.th/th/page/%E0%B8%82%E0%B9%88%E0%B8%B2%E0%B8%A7%E0%B8%AD%E0%B8%B7%E0%B9%88%E0%B8%99%E0%B9%86?menu=5d5bd3d815e39c306002aac7"

news_title_date = get_title_date(cur_url=OTHER_NEWS_URL, page_no=10)
news_details = get_info(cur_url=OTHER_NEWS_URL, page_no=10)

for i, data_dict in enumerate(news_title_date):
    if i < len(news_details):
        data_dict.update({"detail": news_details[i]})

all_news = pd.DataFrame(news_title_date)
all_news.to_csv("MFA_top_news.csv", index=False)
