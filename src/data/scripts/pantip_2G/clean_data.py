import argparse
import glob
import tqdm
import jsonlines
import html
import emoji
import re
import os

parser = argparse.ArgumentParser()

parser.add_argument("--input_folder", help="Name of an input folder", required=True)
parser.add_argument(
    "--output_folder",
    help='Name of an output folder (Default: "clean.txt")',
    default="clean",
)

args = parser.parse_args()


def clean_data(text):
    text = text.replace("<br>", "\n")
    text = text.replace("&nbsp;", "")
    text = text.replace("&quot;", '"')
    text = text.replace("\t:", "\t")
    text = re.sub(r"\\\(.*?\^\-.*?\^\\\)", "", text)
    text = emoji.demojize(text)
    text = html.unescape(text)
    text = re.sub(r"<.*?>", "", text)
    text = text.strip()

    return text


def reformat_jsonl(input_file, output_file, source):
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
                        "source": source,
                        "source_id": tid,
                        "text": clean_data(current_text.strip()),
                        "created_date": item["updated_time"],
                        "updated_date": item["updated_time"],
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
                "source": source,
                "source_id": tid,
                "text": clean_data(current_text.strip()),
                "created_date": item["updated_time"],
                "updated_date": item["updated_time"],
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
