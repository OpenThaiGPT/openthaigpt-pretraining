import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

import warnings  # type: ignore
import jsonlines
import hydra
import time

warnings.filterwarnings("ignore")

NULL = "null"
TEXT_KEY = "text"
META_KEY = "meta"
SOURCE_KEY = "source"
SOURCE_ID_KEY = "source_id"
CREATED_DATE_KEY = "create_date"
UPDATED_DATE_KEY = "update_date"

SOURCE = "King Polity"

MAIN_URL = "https://www.krisdika.go.th/web/guest/thai-code-annotated"

URL_TEMPALTE = "https://www.krisdika.go.th/librarian/getfile?sysid={}&ext=htm"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"  # noqa
}

LAW_GROUP_KEY = "law_group"
NB_LAW_KEY = "nb_laws"
NB_PAGE_KEY = "nb_pages"
TITLE_KEY = "title"
LAW_URL_KEY = "law_url"
SUB_LAW_URL_KEY = "sub_law_url"
SYSID_KEY = "sysid"


def get_url_df():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=chrome_options)
    driver.get(MAIN_URL)
    soup = BeautifulSoup(driver.page_source)

    law_groups = pd.DataFrame(
        [i.text for i in soup.find_all("a", class_="ksdk-theme-bg-third-color")]
    )
    law_groups[LAW_GROUP_KEY] = law_groups[0].map(lambda x: x.split("(")[0][:-1])
    law_groups[NB_LAW_KEY] = law_groups[0].map(lambda x: int(x.split("(")[1][:-1]))
    law_groups[NB_PAGE_KEY] = np.ceil(law_groups[NB_LAW_KEY] / 10).astype(int)
    law_groups = law_groups.drop(0, axis=1)

    def get_law_urls(law_group, nb_pages):
        # open list page
        driver.get(MAIN_URL)
        # click law group
        link = driver.find_element(By.PARTIAL_LINK_TEXT, law_group)
        link.click()

        # check if max pagination button appeared
        try:
            WebDriverWait(driver, 10).until(
                EC.visibility_of_element_located((By.LINK_TEXT, str(nb_pages)))
            )
        except:  # noqa
            print("Max pagination button not found")

        # get law _urls
        laws = []
        law_urls = []
        sub_law_urls = []
        for nb_page in tqdm(range(1, nb_pages + 1)):
            link = driver.find_element(By.LINK_TEXT, str(nb_page))
            link.click()
            soup = BeautifulSoup(driver.page_source)
            laws += [i.text for i in soup.find_all("li", class_="thca-list-law-name")]
            law_urls += [
                i.find_all("li")[-1].find("a").get("href")
                for i in soup.find_all("ul", class_="thca-list-icon")
            ]
            sub_law_soup = [
                i
                for i in soup.find_all("li", class_="thca-list-sub-law")
                if i.find("a").text == "แสดงสารบัญลูกบทตามสารบัญกฎหมาย"
            ]
            sub_law_urls += [i.find("a").get("href") for i in sub_law_soup]

        df = pd.DataFrame(
            {TITLE_KEY: laws, LAW_URL_KEY: law_urls, SUB_LAW_URL_KEY: sub_law_urls}
        )
        df[SYSID_KEY] = df.law_url.map(lambda x: x.split("=")[-2].split("&")[0])
        df[LAW_GROUP_KEY] = law_group
        return df

    dfs = []
    for row in tqdm(law_groups.itertuples(index=False)):
        print(row[0])
        df = get_law_urls(row[0], row[2])
        dfs.append(df)

    law_url_df = pd.concat(dfs)

    return law_url_df


@hydra.main(version_base=None, config_path="./config", config_name="crawl_krisdika")
def main(cfg):
    config = cfg.king

    law_url_df = get_url_df()
    with jsonlines.open(config.output_path, "w") as writer:
        for i in tqdm(range(len(law_url_df))):
            sample = law_url_df.iloc[i]

            sysid = sample[SYSID_KEY]

            link = URL_TEMPALTE.format(sysid)
            reponse = requests.get(link, verify=False, headers=HEADERS)

            soup = BeautifulSoup(reponse.content, "html.parser")

            data = {
                TEXT_KEY: soup.text,
                SOURCE_KEY: SOURCE,
                SOURCE_ID_KEY: sysid,
                CREATED_DATE_KEY: NULL,
                UPDATED_DATE_KEY: NULL,
                META_KEY: NULL,
            }

            writer.write(data)

            time.sleep(config.delay)


if __name__ == "__main__":
    main()  # type: ignore
