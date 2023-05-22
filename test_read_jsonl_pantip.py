import json
import pandas as pd
import re

pantip_df = pd.DataFrame({"tid": [], "cid": [], "title": [], "text": []})

rline = open("pantip_data/20230515_sample2G_1000_topics.jsonl", "r")

for line in rline:
    # clean err character
    line = re.sub(r"[\x01-\x1f]", " ", line)  # control char
    line = line.replace("	:", " ")  # tab+colon
    line = line.replace("\t", " ")  # tab
    line = line.replace("\\", " ")  # from line 13849 emoji \(^-^\)

    # read line as json
    line_j = json.loads(line)
    if line_j["cid"] != "0":
        # title = pantip_df["title"][len(pantip_df) - 1]
        title = pantip_df.loc[
            (pantip_df["cid"] == "0") & (pantip_df["tid"] == line_j["title"])
        ].title
    else:
        title = line_j["title"]
    line_dict = {
        "tid": line_j["tid"],
        "cid": line_j["cid"],
        "title": title,
        "text": line_j["desc"],
    }

    # keep in df
    pantip_df.loc[len(pantip_df)] = line_dict

pantip_df.to_csv("pantip_2g.csv")
