import pandas as pd
from datasets import Dataset, load_from_disk
from openthaigpt_pretraining_data.web_crawls_mfa.crawl_news import (
    get_title_date,
    get_info,
)

SPEECHES_URL = "https://www.mfa.go.th/th/page/%E0%B8%AA%E0%B8%B8%E0%B8%99%E0%B8%97%E0%B8%A3%E0%B8%9E%E0%B8%88%E0%B8%99%E0%B9%8C?menu=5d5bd3d815e39c306002aacd"

news_title_date = get_title_date(cur_url=SPEECHES_URL, page_no=8)
news_details = get_info(cur_url=SPEECHES_URL, page_no=8)

for i, data_dict in enumerate(news_title_date):
    if i < len(news_details):
        data_dict.update({"detail": news_details[i]})

all_news = pd.DataFrame(news_title_date)
dataset = Dataset.from_pandas(all_news)
dataset.save_to_disk("MFA_speeches.arrow")
loaded_dataset = load_from_disk("MFA_speeches.arrow")

