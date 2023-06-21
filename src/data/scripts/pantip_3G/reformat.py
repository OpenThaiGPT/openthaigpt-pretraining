import argparse
import glob
import tqdm
import jsonlines
import html
import re
import os
import gzip

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


def clean_text(value):
    if isinstance(value, str):
        # Replace tab+colon with tab
        value = value.replace("\t:", "\t")
        # Decode HTML entities
        value = html.unescape(value)
        # Replace <br> with newline
        value = value.replace("<br>", "\n")
        # Remove HTML tags
        value = re.sub(r"<.*?>", "", value)
        # Strip leading and trailing whitespace
        value = value.strip()

    return value


def reformat_jsonl(input_file, output_file, source):
    with gzip.open(input_file, "rt") as gzip_file, jsonlines.open(
        output_file, "w"
    ) as writer:
        reader = jsonlines.Reader(gzip_file)
        current_tid = None
        current_data = {
            "text": "",
            "source": source,
            "source_id": current_tid,
            "created_date": "",
            "updated_date": "",
            "meta": "",
        }

        for item in reader.iter(skip_invalid=True):
            tid = item["tid"]
            cid = item["cid"]
            desc = item.get("desc")

            if tid != current_tid:
                # Write the previous line (if any) to the output file
                if current_tid is not None:
                    current_data["text"] = clean_text(current_data["text"])
                    writer.write(current_data)

                # Start a new line
                current_tid = tid
                current_data = {
                    "text": "",
                    "source": source,
                    "source_id": current_tid,
                    "created_date": item.get("created_time"),
                    "updated_date": item.get("updated_time"),
                    "meta": None,
                }

            if cid == "0":
                # Forum entry with title
                current_data["text"] = "กระทู้ {} เนื้อหา {} ".format(
                    item["title"], desc
                )
                current_data["text"] += "ประเภท {} ".format(item.get("type", ""))
                current_data["text"] += "เกี่ยวกับ {} ".format(item.get("tags", ""))

                current_data["source"] = source
                current_data["source_id"] = tid
                current_data["updated_date"] = item.get("updated_time")
                current_data["created_date"] = item.get("created_time")

            else:
                # Comment related to the current forum entry
                current_data["text"] += "ความคิดเห็นที่ {} {} ".format(cid, desc)

        # Write the last line to the output file
        if current_tid is not None:
            current_data["text"] = clean_text(current_data["text"])
            writer.write(current_data)


if __name__ == "__main__":
    for path in tqdm.tqdm(glob.glob(args.input_folder + "/*.jsonl.gz")):
        source = path.replace(args.input_folder, "")
        des_path = args.output_folder + source
        des_folder = des_path.split(os.sep)[:-1]
        des_folder = os.sep.join(des_folder)

        os.makedirs(des_folder, exist_ok=True)

        des_path = des_path.replace(".gz", "")

        reformat_jsonl(path, des_path, "pantip_3g")
        print("done {}".format(path))
