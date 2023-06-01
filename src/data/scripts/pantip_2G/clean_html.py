import pandas as pd
import argparse
import glob
import tqdm
import json
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


def clean_text(text):
    text = re.sub(r"[\x01-\x1f]", " ", text)  # control char
    text = re.sub(r"<.*?>", "", text)

    text = text.replace("	:", " ")  # tab+colon
    text = text.replace("\t", " ")  # tab
    text = text.replace("\\", " ")  # from line 13849 emoji \(^-^\)
    text = text.replace("<br>", "")
    text = text.replace("&nbsp;", "")
    text = text.replace("&quot;", '"')

    return text


def load_and_clean_data(path, source):
    with open(path, "r", encoding="utf-8") as file:
        # Read the JSON data from the file
        data = json.dumps(file.read())

    # Print the JSON data
    j_file = json.loads(data).splitlines()
    test = []
    for jline in j_file:
        try:
            result = json.loads(jline)
            test.append(result)
        except Exception:
            pass

    data = {
        "source_id": [],
        "source": [],
        "text": [],
        "created_date": [],
        "updated_date": [],
    }
    n = 0
    for i in j_file:
        try:
            line_j = json.loads(i)
            if "title" not in line_j.keys():
                line_j["title"] = ""
            if "desc" not in line_j.keys():
                line_j["desc"] = ""
            line_j = json.loads(i)
            txt_title = line_j["title"]
            txt_desc = line_j["desc"]
            combine = txt_title + txt_desc
            combine = clean_text(combine)
            data["text"].append(combine)
            data["source"].append(source)
            data["source_id"].append(n)
            data["created_date"].append(line_j["updated_time"])
            data["updated_date"].append(line_j["updated_time"])
            n += 1
        except Exception:
            continue

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    for path in tqdm.tqdm(glob.glob(args.input_folder + "/**/*.jsonl")):
        source = path.replace(args.input_folder, "")
        data = load_and_clean_data(path, source)

        des_path = args.output_folder + source
        des_folder = des_path.split("\\")[:-1]
        des_folder = "\\".join(des_folder)

        os.makedirs(des_folder, exist_ok=True)

        data.to_json(
            des_path,
            orient="records",
            lines=True,
            force_ascii=False,
        )
