from bs4 import BeautifulSoup
from tqdm import tqdm
import requests
import jsonlines
import hydra

NULL = "null"
TEXT_KEY = "text"
META_KEY = "meta"
SOURCE_KEY = "source"
SOURCE_ID_KEY = "source_id"
CREATED_DATE_KEY = "create_date"
UPDATED_DATE_KEY = "update_date"

SOURCE = "Krisdika Polity"

MAIN_URL = "https://www.krisdika.go.th/web/guest/law?p_p_id=LawPortlet_INSTANCE_aAN7C2U5hENi&p_p_lifecycle=2&p_p_state=normal&p_p_mode=view&p_p_cacheability=cacheLevelPage&_LawPortlet_INSTANCE_aAN7C2U5hENi_lawTypeId=1&_LawPortlet_INSTANCE_aAN7C2U5hENi_javax.portlet.action=selectLawTypeMenu"  # noqa

HREF_KEY = "href"
ROW_KEY = "rows"
ID_KEY = "id"
DATA_KEY = "data"
VALUE_KEY = "value"


def save_jsonl_data(url_tag, id, date, writer):
    url_soup = BeautifulSoup(url_tag)
    link = url_soup.a[HREF_KEY]
    link = link.replace("get", "getfile")

    page = requests.get(link, verify=False)
    page_soup = BeautifulSoup(page.content)

    data = {
        TEXT_KEY: page_soup.text,
        SOURCE_KEY: SOURCE,
        SOURCE_ID_KEY: id,
        CREATED_DATE_KEY: date,
        UPDATED_DATE_KEY: date,
        META_KEY: NULL,
    }

    writer.write(data)


def reformat_id_to_date(id):
    date = id.replace("ร", "").split("-")[:-1]
    date = "-".join(date)

    return date


@hydra.main(version_base=None, config_path="./config", config_name="crawl_krisdika")
def main(cfg):
    config = cfg.polity
    old_date = None

    with jsonlines.open(config.output_path, "w") as writer:
        res = requests.get(MAIN_URL, verify=False)
        for data in tqdm(res.json()[ROW_KEY][0][ROW_KEY], desc="Main Process"):
            if data.get(ROW_KEY, False):
                for sub_data in tqdm(data[ROW_KEY], desc="Sub Process"):
                    id = sub_data[ID_KEY]

                    date = reformat_id_to_date(id)

                    if date == old_date:
                        continue

                    old_date = date

                    url_tag = sub_data[DATA_KEY][0][VALUE_KEY]
                    save_jsonl_data(url_tag, id, date, writer)

            else:
                id = data[ID_KEY]

                date = reformat_id_to_date(id)

                if date == old_date:
                    continue

                old_date = date

                url_tag = data[DATA_KEY][0][VALUE_KEY]
                save_jsonl_data(url_tag, id, date, writer)


if __name__ == "__main__":
    main()  # type: ignore
