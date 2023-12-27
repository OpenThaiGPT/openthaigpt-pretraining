from collections import Counter
import jsonlines
import re
import html
import emoji
from tqdm import tqdm

SOURCE = "source"
SOURCE_ID = "source_id"
TEXT = "text"
CREATED_DATE = "created_date"
UPDATED_DATE = "updated_date"
TID = "tid"
CID = "cid"
TITLE = "title"
DESC = "desc"
UPDATED_TIME = "updated_time"
FOLDER_PATH = "folder_path"
WRITED_FILE_NAME = "writed_file_name"
META = "meta"
LABEL = ""


def load_jsonl_to_list(file_path: str) -> list[dict]:
    """
    Description:
        open jsonl file and return list of dict
    Args:
        file_path: Input string of file path.
    Returns:
        result: List of dict
    """
    result = []
    with jsonlines.open(file_path, "r") as reader:
        for obj in reader.iter(type=dict, skip_invalid=True):
            result.append(obj)
    return result


def clean_data(text):
    """
    Description:
        Clean the text by replacing <br> with newline, removing emoji,
        HTML tags, and HTML entities,
        and stripping leading and trailing whitespace.
    Args:
        text: Input string of text.
    Returns:
        text: Cleaned string of text.
    """
    # Replace <br> with newline
    text = text.replace("<br>", "\n")
    # Replace tab+colon with tab
    text = text.replace("\t:", "\t")
    # Remove emoji \(^-^\)
    text = re.sub(r"\\\(.*?\^\-.*?\^\\\)", "", text)
    # Remove emoji
    text = emoji.demojize(text)
    # Decode HTML entities
    text = html.unescape(text)
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Strip leading and trailing whitespace
    text = text.strip()

    return text


for i in tqdm(input):
    list_new_format = []
    # Create list of topic id (tid_list)
    input_list = load_jsonl_to_list(f"{FOLDER_PATH}" + i)
    tid_list = [li[TID] for li in input_list]
    tid_counter = Counter(tid_list)
    tid_list = list(tid_counter.keys())
    tid_list.sort()

    for topic in tqdm(tid_list):
        current_text = ""
        # List all lines with the same tid
        topic_lists = [li for li in input_list if li[TID] == topic]

        # Get cid0
        cid0 = [li for li in topic_lists if li[CID] == "0"]
        # cid0 = cid0[0]

        # Check if cid0 works correctly
        if len(cid0) < 1:
            print(topic + " doesn't have cid0")
        else:
            if len(cid0) > 1:
                print(topic + " contain cid0 more than 1")
            cid0 = cid0[0]

            # Start combining topic text
            current_text += "กระทู้ {} เนื้อหา {} ".format
            (cid0[TITLE], cid0[DESC])
            topic_date = cid0[UPDATED_TIME]

            # Remove cid0
            cid0_index = topic_lists.index(cid0)
            topic_lists.pop(cid0_index)

        # Loop to get comment
        latest_cid = 0
        for comment in topic_lists:
            current_text += " ความคิดเห็นที่ {} {}".format
            (comment[CID], comment[DESC])
            if int(comment[CID]) > latest_cid:
                last_comment_date = comment[UPDATED_TIME]
                latest_cid = int(comment[CID])
            # print(current_text)

        # Create dict of new format
        data = {
            SOURCE: "pantip2G",
            SOURCE_ID: topic,
            TEXT: clean_data(current_text.strip()),
            # TEXT: current_text.strip(),
            CREATED_DATE: topic_date,
            UPDATED_DATE: last_comment_date,
            META: LABEL,
        }
        list_new_format.append(data)
        # print(clean_data(current_text))
        # print(data)

    with jsonlines.open(f"{FOLDER_PATH}+{WRITED_FILE_NAME}.jsonl", "w") as writer:
        writer.write_all(list_new_format)
