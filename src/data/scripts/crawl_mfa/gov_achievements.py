import pandas as pd
from datasets import Dataset, load_from_disk
from openthaigpt_pretraining_data.web_crawls_mfa.crawl_gov_achievements import (
    get_title_date,
    get_info,
)

CUR_URL = "https://www.mfa.go.th/th/page/%E0%B8%A7%E0%B8%B5%E0%B8%94%E0%B8%B4%E0%B8%97%E0%B8%B1%E0%B8%A8%E0%B8%99%E0%B9%8C%E0%B8%9B%E0%B8%A3%E0%B8%B0%E0%B8%8A%E0%B8%B2%E0%B8%AA%E0%B8%B1%E0%B8%A1%E0%B8%9E%E0%B8%B1%E0%B8%99%E0%B8%98%E0%B9%8C%E0%B8%9C%E0%B8%A5%E0%B8%87%E0%B8%B2%E0%B8%99%E0%B9%80%E0%B8%94%E0%B9%88%E0%B8%99%E0%B8%95%E0%B8%B2%E0%B8%A1%E0%B8%99%E0%B9%82%E0%B8%A2%E0%B8%9A%E0%B8%B2%E0%B8%A2%E0%B8%A3%E0%B8%B1%E0%B8%90%E0%B8%9A%E0%B8%B2%E0%B8%A5"

news_title_date = get_title_date(CUR_URL, 0.5)
news_details = get_info(CUR_URL, 0.5)

for i, data_dict in enumerate(news_title_date):
    if i < len(news_details):
        data_dict.update({"detail": news_details[i]})

all_news = pd.DataFrame(news_title_date)
dataset = Dataset.from_pandas(all_news)
dataset.save_to_disk("mfa_gov_achievements.arrow")
loaded_dataset = load_from_disk("mfa_gov_achievements.arrow")
