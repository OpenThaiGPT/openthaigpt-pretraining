import jsonlines
import html
import re

SOURCE = "source"
SOURCE_ID = "source_id"
TEXT = "text"
CREATED_DATE = "created_date"
UPDATED_DATE = "updated_date"


def remove_website(text):
    """
    remove website: remove html and other condition of website.
    Input: pain text, List of data text[].
    Output: text removed website and html tag.
    """
    # Remove HTML tags
    text = re.sub(r"\<.*?\>", "", text, flags=re.MULTILINE)
    # Remove tags '[' text ']'
    text = re.sub(r"\[.*?\/.*?\]", "", text)
    # Remove website
    # Remove http
    text = re.sub(r"http\S+", " website", text, flags=re.MULTILINE)
    # remove www
    text = re.sub(r"(www|WWW).\S+", " website", text, flags=re.MULTILINE)
    # remove other condition
    text = re.sub(r".+\.com.\S+", " website", text, flags=re.MULTILINE)
    # Return cleaned website data
    return text


def remove_error(text):
    """
    remove error: To remove system error message out of data.
    Input: pain text, List of data text[].
    Output: data cleaned system error message.
    """
    check_point = 0
    minus_N = 10  # privote of range - minus_N
    while check_point < len(text) - minus_N:
        # Chekinf word "error" or "Error"
        if (text[check_point] == "e" or text[check_point] == "E") and (
            check_point < len(text) - minus_N
        ):
            if (
                text[check_point + 1] == "r"
                and text[check_point + 4] == "r"
                and not (
                    ("ก" < text[check_point + 7] < "๛")
                    or ("ก" < text[check_point - 3] < "๛")
                )
            ):
                tempsup = ""
                walk_count = 0
                # If it found error find untill the last line of error message
                while not ("ก" < text[check_point + walk_count] < "๛") and (
                    check_point + walk_count < len(text) - minus_N
                ):
                    tempsup += text[check_point + walk_count]
                    walk_count += 1
                # Replace error message
                text = text.replace(tempsup, "")
        # Update checkpoint
        check_point += 1
    # return cleaned data
    return text


def clean_data(text):
    """
    clean data: To clean nonnessecsory data.
    Input: pain text, List of data text[].
    Output: Cleaned data for readable data.
    """
    # Replace <br> with newline
    text = text.replace("<br>", "\n")
    # Replace tab+colon with tab
    text = text.replace("\t:", "\t")
    # Decode HTML entities
    text = html.unescape(text)
    # remove [spoil]
    text = text.replace("[Spoil] คลิกเพื่อดูข้อความที่ซ่อนไว้", "")

    # call remove_website function
    re_text = remove_website(text)
    # call remove_error function
    re_text = remove_error(re_text)

    # Strip leading and trailing whitespace
    re_text = re_text.strip()

    return re_text
    # return removed text


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
                    current_text = clean_data(current_text.strip())
                    current_text += "  ประเภท {} ".format(item.get("type", ""))
                    current_text += "เกี่ยวกับ {} ".format(item.get("tags", ""))
                    # actual tid is tid - 1
                    temp_tid = str(int(tid) - 1)
                    data = {
                        SOURCE: source,
                        SOURCE_ID: temp_tid,
                        TEXT: current_text,
                        CREATED_DATE: item["created_time"],
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
            # actual tid is tid - 1
            temp_tid = str(int(tid) - 1)
            data = {
                SOURCE: source,
                SOURCE_ID: temp_tid,
                TEXT: clean_data(current_text.strip()),
                CREATED_DATE: item["created_time"],
                UPDATED_DATE: item["updated_time"],
            }
            writer.write(data)
