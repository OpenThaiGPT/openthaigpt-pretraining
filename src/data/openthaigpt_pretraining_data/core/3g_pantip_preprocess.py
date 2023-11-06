from pprint import pprint
# from bs4 import BeautifulSoup
from collections import Counter, defaultdict
# from pythainlp.corpus import thai_stopwords
import jsonlines
import multiprocessing
import re
import html
import emoji
import os
import gzip
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
FOLDER_PATH = 'folder_path'
WRITED_FILE_NAME = 'writed_file_name'
META = 'meta'
LABEL = ''


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
    with gzip.open(file_path,"r") as reader:
        reader = jsonlines.Reader(reader)
        for obj in reader.iter(type=dict, skip_invalid=True):
            result.append(obj)
    return result


def clean_data(text):
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
    
    # Remove website
    text = re.sub(r"^https?:\/\/.*[\r\n]*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^http?:\/\/.*[\r\n]*", "", text, flags=re.MULTILINE)
    text = re.sub(r"(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b", "", text, flags=re.MULTILINE)
    text = re.sub(r"<a+[*]+' a>'?:\/\/.*[\r\n]*", "", text, flags=re.MULTILINE)
    text = re.sub(r"<a>?:\/\/.*[\r\n]*", "", text, flags=re.MULTILINE)
    #remove [spoil]
    text = text.replace("[Spoil] คลิกเพื่อดูข้อความที่ซ่อนไว้", "")
    check_point = 0
    minus_N = 10
    while(check_point < len(text)-minus_N):
        #Chekinf word "error" or "Error"
        if((text[check_point]=='e' or text[check_point]=='E') and (check_point<len(text)-minus_N)):
            
            if(text[check_point+1]=='r' and text[check_point+2]=='r' and text[check_point+3]=='o' and text[check_point+4]=='r' and not(('ก' < text[check_point+5] < '๛') or ('ก' < text[check_point+6] < '๛') or ('ก' < text[check_point+7] < '๛') or ('ก' < text[check_point-1] < '๛') or ('ก' < text[check_point-2] < '๛')or ('ก' < text[check_point-3] < '๛'))):
                    tempsup=''
                    walk_count=0
                    while(  not('ก' < text[check_point+walk_count] < '๛') and (check_point+walk_count<len(text)-minus_N) ):
                        tempsup+=text[check_point+walk_count]
                        walk_count+=1
                    
                    text = text.replace(tempsup,"")
            
        check_point+=1

    # Strip leading and trailing whitespace
    text = text.strip()
    
    return text


def process_input(i):
    
    list_new_format = []
    # Create list of topic id (tid_list)
    input_list = load_jsonl_to_list(f'{FOLDER_PATH}'+i)
    tid_list = [li[TID] for li in input_list]
    tid_counter = Counter(tid_list)
    tid_list = list(tid_counter.keys()) 
    tid_list.sort()

    dictionary = defaultdict(lambda: [])

    for item in input_list:
        topic_id = item[TID]
        dictionary[topic_id].append(item)

    for topic in tid_list:
        current_text = ""
        # List all lines with the same tid
        # topic_lists = [li for li in input_list if li[TID] == topic]
        # topic_lists = [li for li in input_list if li[TID] == topic]
        topic_lists = dictionary[topic]

        # Get cid0 
        cid0 = [li for li in topic_lists if li[CID] == "0" ]
        # cid0 = cid0[0]
        
        topic_date = ''
        last_comment_date = ''

        # Check if cid0 works correctly
        if len(cid0) < 1:
            print(topic + " doesn't have cid0")
        else:
            if len(cid0) > 1:
                print(topic + " contain cid0 more than 1")
            cid0 = cid0[0]
            
            # Start combining topic text
            current_text += "หัวข้อ {} เนื้อหา {} ".format(cid0[TITLE], cid0[DESC])
            topic_date = cid0[UPDATED_TIME]

            # Remove cid0
            cid0_index = topic_lists.index(cid0)
            topic_lists.pop(cid0_index)

        # Loop to get comment
        for comment in topic_lists:
            current_text += " ความคิดเห็นที่ {} {}".format(comment[CID], comment[DESC])
            last_comment_date = comment[UPDATED_TIME]

            # print(current_text)
        
        # Create dict of new format
        data = {
                SOURCE: 'pantip3G',
                SOURCE_ID: topic,
                TEXT: clean_data(current_text.strip()),
                # TEXT: current_text.strip(),
                CREATED_DATE: last_comment_date,
                UPDATED_DATE: topic_date,
                META: LABEL
            }
        list_new_format.append(data)
        # print(clean_data(current_text))
        # print(data)
    return list_new_format


if __name__ == "__main__":
    input = os.listdir(FOLDER_PATH)

    with multiprocessing.Pool(128) as pool:
        input_data = input  # your list of inputs
        results = list(tqdm(pool.imap(process_input, input_data), total=len(input_data)))

    # Flatten list of results
    flat_results = [item for sublist in tqdm(results) for item in sublist]
    with jsonlines.open(f'{WRITED_FILE_NAME}.jsonl', "w") as writer:
        writer.write_all(flat_results)
