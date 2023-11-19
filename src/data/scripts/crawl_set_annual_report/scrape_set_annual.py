from selenium import webdriver
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd
import requests
import zipfile  # type: ignore
import io  # type: ignore
import re  # type: ignore
import hydra

BASE_URL = "https://www.set.or.th/th/market/product/stock/quote/{}/company-profile/information"  # noqa: E501


@hydra.main(version_base=None, config_path="./config", config_name="crawl_set_annual")
def main(cfg):
    config = cfg.crawl_data
    csv_path = config.csv_path
    output_folder = config.output_folder

    companys = pd.read_csv(csv_path)["List of Listed Companies & Contact Information"][
        1:
    ].reset_index(drop=True)

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    driver = webdriver.Chrome(chrome_options)

    for company in tqdm(companys):
        url = BASE_URL.format(company)
        driver.get(url)
        html_code = driver.page_source

        soup = BeautifulSoup(html_code, "html.parser")
        download_tags = soup.find_all("a", {"class": "card-download-link d-flex"})
        for tag in download_tags:
            file_response = requests.get(tag["href"])

            date = tag.find_all("span")[1]
            date = re.search(r"\((.*?)\)", date.text).group(1)

            date = date.replace(" ", "-")

            output_path = "/".join([".", output_folder, company, date])

            with zipfile.ZipFile(
                io.BytesIO(file_response.content),
                mode="r",
            ) as zip_ref:
                zip_ref.extractall(output_path)


if __name__ == "__main__":
    main()  # type: ignore
