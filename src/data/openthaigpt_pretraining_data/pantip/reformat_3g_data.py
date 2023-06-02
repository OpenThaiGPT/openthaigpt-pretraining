import jsonlines

# import html
# import re
# import emoji


def clean_text(value):
    """if isinstance(value, str):
        # Remove control characters
        value = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', value)
        # Replace tab+colon with tab
        value = value.replace('\t:', '\t')
        # Remove tabs
        value = value.replace('\t', '')
        # Remove emoji
        value = emoji.demojize(value)
        # Decode HTML entities
        value = html.unescape(value)
        # Remove HTML tags
        value = re.sub(r'<.*?>', '', value)
        # Replace <br> with newline
        value = value.replace('<br>', '\n')
        # Strip leading and trailing whitespace
        value = value.strip()

    return value"""
    pass


def reformat_jsonl(input_file, output_file):
    with jsonlines.open(input_file, "r") as reader, jsonlines.open(
        output_file, "w"
    ) as writer:
        current_tid = None
        current_data = {}

        for item in reader:
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
                current_data = {}

            if cid == "0":
                # Forum entry with title
                current_data["text"] = "กระทู้ {} เนื้อหา {} ".format(
                    item["title"], desc
                )
                current_data["text"] += "ประเภท {} ".format(item.get("type", ""))
                current_data["text"] += "เกี่ยวกับ {} ".format(item.get("tags", ""))

                current_data["source"] = item.get("url")
                current_data["source_id"] = tid
                current_data["updated_time"] = item.get("updated_time")
                current_data["created_time"] = item.get("created_time")

            else:
                # Comment related to the current forum entry
                current_data["text"] += "ความคิดเห็นที่ {} {} ".format(cid, desc)

        # Write the last line to the output file
        if current_tid is not None:
            current_data["text"] = clean_text(current_data["text"])
            writer.write(current_data)
