import argparse
import glob
import tqdm
import jsonlines
import html
import re
import os

SOURCE = "source"
SOURCE_ID = "source_id"
TEXT = "text"
CREATED_DATE = "created_date"
UPDATED_DATE = "updated_date"

parser = argparse.ArgumentParser()

parser.add_argument("--input_folder", help="Name of an input folder", required=True)
parser.add_argument(
    "--output_folder",
    help='Name of an output folder (Default: "clean")',
    default="clean",
)

args = parser.parse_args()


def clean_data(text):
    # Replace <br> with newline
    text = text.replace("<br>", "\n")
    # Replace tab+colon with tab
    text = text.replace("\t:", "\t")
    # Decode HTML entities
    text = html.unescape(text)
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Strip leading and trailing whitespace
    text = text.strip()

    return text


def reformat_jsonl(input_file, output_file, source):
    """
    clean data and change format to
        "text": "Data",
        "source": "Source of the data",
        "source_id": "id of the original item in source data",
        "created_date": "Created date",
        "updated_date": "Updated date"
    """
    with jsonlines.open(input_file, "r") as reader, jsonlines.open(
        output_file, "w"
    ) as writer:
        current_tid = None
        current_text = ""

        for item in reader.iter(skip_invalid=True):
            tid = item["tid"]
            cid = item["cid"]
            desc = item.get("desc")

            if tid != current_tid:
                # Write the previous line (if any) to the output file
                if current_tid is not None:
                    data = {
                        SOURCE: source,
                        SOURCE_ID: tid,
                        TEXT: clean_data(current_text.strip()),
                        CREATED_DATE: item["updated_time"],
                        UPDATED_DATE: item["updated_time"],
                    }
                    writer.write(data)

                # Start a new line
                current_tid = tid
                current_text = ""

            if cid == "0":
                # Forum entry with title
                current_text += "กระทู้ {} เนื้อหา {} ".format(item["title"], desc)
            else:
                # Comment related to the current forum entry
                current_text += "ความคิดเห็นที่ {} {} ".format(cid, desc)

        # Write the last line to the output file
        if current_tid is not None:
            data = {
                SOURCE: source,
                SOURCE_ID: tid,
                TEXT: clean_data(current_text.strip()),
                CREATED_DATE: item["updated_time"],
                UPDATED_DATE: item["updated_time"],
            }
            writer.write(data)


if __name__ == "__main__":
    for path in tqdm.tqdm(glob.glob(args.input_folder + "/**/*.jsonl")):
        source = path.replace(args.input_folder, "")
        des_path = args.output_folder + source
        des_folder = des_path.split("\\")[:-1]
        des_folder = "\\".join(des_folder)

        os.makedirs(des_folder, exist_ok=True)

        reformat_jsonl(path, des_path, source)
