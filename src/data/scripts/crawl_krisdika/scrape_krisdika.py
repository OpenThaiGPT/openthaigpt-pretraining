from selenium import webdriver
from bs4 import BeautifulSoup
from tqdm import tqdm
import requests
import json
import jsonlines
import time
import hydra


DOC_URL_FORMAT = "https://www.ocs.go.th/council-of-state/#/public/doc/{}"

NULL = "null"
TEXT_KEY = "text"
META_KEY = "meta"
SOURCE_KEY = "source"
SOURCE_ID_KEY = "source_id"
CREATED_DATE_KEY = "create_date"
UPDATED_DATE_KEY = "update_date"

SOURCE = "Krisdika"

TIME_LINE_ID_KEY = "timelineId"
PUBLISH_DATE_AD_KEY = "publishDateAd"
RESPOND_BODY_KEY = "respBody"
DATA_KEY = "data"


@hydra.main(version_base=None, config_path="./config", config_name="crawl_krisdika")
def main(cfg):
    driver = webdriver.Chrome()

    with open(cfg.requests_body, "r") as f:
        requests_body = json.load(f)

    res = requests.post(cfg.url, json=requests_body)

    respond_datas = res.json()[RESPOND_BODY_KEY][DATA_KEY]

    with jsonlines.open(cfg.output_path, "w") as writer:
        for data in tqdm(respond_datas):
            source_id = data[TIME_LINE_ID_KEY]
            date = data[PUBLISH_DATE_AD_KEY]

            doc_url = DOC_URL_FORMAT.format(source_id)
            driver.get(doc_url)

            time.sleep(cfg.delay)

            soup = BeautifulSoup(driver.page_source)
            raw_text = soup.find("div", {"class": "offset-1 col-10 mb-5 a4"})
            raw_text = raw_text.find_all("p")

            all_text = []
            for text in raw_text:
                all_text.append(text.text)

            document = "\n".join(all_text)

            data = {
                TEXT_KEY: document,
                SOURCE_KEY: SOURCE,
                SOURCE_ID_KEY: source_id,
                CREATED_DATE_KEY: date,
                UPDATED_DATE_KEY: date,
                META_KEY: NULL,
            }

            writer.write(data)


if __name__ == "__main__":
    main()  # type: ignore
