import json
import glob
import pandas as pd
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader


def get_data_from_url(url_link, headers=None):
    response = requests.get(url_link, headers=headers)
    return response.json()


def clean_text(text):
    prefix_to_remove = "ลงลายมือชื่อ _________________\n"
    if text.startswith(prefix_to_remove):
        return text[len(prefix_to_remove) :]
    return text


def download_pdf_links(df_pdf):
    existing_ids = []
    for idx, row in df_pdf.iterrows():
        url = (
            f'https://www.set.or.th/th/market/news-and-alert/newsdetails?id={row["id"]}'
        )
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        links = soup.find_all("a", href=lambda href: href and ".pdf" in href)
        for i, link in enumerate(links, 1):
            response = requests.get(link["href"])
            with open(f"Extract{row['id']}-{i}.pdf", "wb") as pdf:
                pdf.write(response.content)
            print(f"File {row['id']}-{i} downloaded")
            existing_ids.append(row["id"])
    return existing_ids


def process_pdfs(df_pdf, existing_ids):
    error_id = []
    jsonl_data = []
    for id in existing_ids:
        keep_path_id = glob.glob(f"Extract{id}*.pdf")
        for pdf_file in keep_path_id:
            try:
                reader = PdfReader(pdf_file)
                pageObj = reader.pages[0]
                lines = pageObj.extract_text().strip()

                current_entry = {
                    "text": clean_text(lines),
                    "source": df_pdf.loc[df_pdf["id"] == id, "source"].iloc[0],
                    "source_id": "null",
                    "create_date": str(
                        df_pdf.loc[df_pdf["id"] == id, "datetime"].iloc[0]
                    ),
                    "update_date": str(
                        df_pdf.loc[df_pdf["id"] == id, "datetime"].iloc[0]
                    ),
                    "meta": "null",
                }
                jsonl_data.append(current_entry)
            except Exception as e:
                error_id.append(pdf_file.split(".")[0])
                print(f"Error processing file {pdf_file}:", e)
    return jsonl_data


start_date, end_date = "01/06/2023", "31/01/2024"
url_get_audit_comment = f"https://www.set.or.th/api/cms/v1/news/set?fromDate={start_date}&toDate={end_date}&orderBy=date&lang=th"  # noqa
header_ref_get_audit_comment = "https://www.set.or.th/th/market/news-and-alert/news"
header_xchannel_get_audit_comment = "WEB_SET"
header_cookie_get_audit_comment = "incap_ses_488_2046605=ye5cQaPThkd4o6XYmLrFBoL33mQAAAAAN2mcmSaQEKmhmZHKY6uekw==; nlbi_2046605=OfZ+PWbC/WgTMnNRx+ze1wAAAABat4l0bFK3cgiCNK18uuAg; visid_incap_2046605=rp6/FkR2R6aO8uirufGPSYG/1WQAAAAAQUIPAAAAAADt/xfivnBTm5CdFwUmnVq6"  # noqa

headers = {
    "Referer": header_ref_get_audit_comment,
    "Cookie": header_cookie_get_audit_comment,
    "x-channel": header_xchannel_get_audit_comment,
}

get_audit_comment = get_data_from_url(url_get_audit_comment, headers=headers)
df_pdf = pd.DataFrame(get_audit_comment["paginateNews"]["newsInfoList"])

existing_ids = download_pdf_links(df_pdf)
jsonl_data = process_pdfs(df_pdf, existing_ids)

with open("dataset.jsonl", "a") as f:
    for entry in jsonl_data:
        f.write(json.dumps(entry) + "\n")

for obj in jsonl_data:
    print(obj)
