import html
import re


def remove_web_and_tag(text):
    """
    remove website: remove html and other condition of website.
    Input: pain text, List of data text[].
    Output: text removed website and html tag.
    """
    website = " website"
    # Remove HTML tags
    text = re.sub(r"\<.*?\>", "", text, flags=re.MULTILINE)
    # Remove tags '[' text ']'
    text = re.sub(r"\[.*?\/.*?\]", "", text)
    # Remove website
    # Remove http
    text = re.sub(r"http\S+", website, text, flags=re.MULTILINE)
    # remove www
    text = re.sub(r"(www|WWW).\S+", website, text, flags=re.MULTILINE)
    # remove other condition
    text = re.sub(r".+\.com.\S+", website, text, flags=re.MULTILINE)
    # Return cleaned website data
    return text


def remove_error(text):
    """
    remove error: To remove system error message out of data.
    Input: pain text, List of data text[].
    Output: data cleaned system error message.
    """
    text_lenght = len(text)
    # Finding index of [Ee]rror occured
    error_detected_index_object = re.finditer(pattern="[eE]rror", string=text)
    error_detected_index = [index.start() for index in error_detected_index_object]
    # error_detected_index.append(text_lenght - 1)
    # Set up range of thai character
    first_thai_character_order = ord("ก")
    last_thai_character_order = ord("๛")
    # Set up clean_flag if it 0 not clean, but 1 need to clean error
    clean_flag = 0
    previous_index = 0
    # This for store cleaned data
    error_cleaned = ""
    for index in error_detected_index:
        current_index = index
        # print(current_index)
        sub_sentence = text[previous_index:current_index]
        # check for to be clean or not
        # Thai character [^\u0E00-\u0E7F]=[ก-๛]"
        if clean_flag == 1:
            thai_char_pattern = re.compile(r"[\u0E00-\u0E7F]", re.UNICODE)
            matches = thai_char_pattern.findall(sub_sentence)

            if len(matches) > 0:
                cleaned_error_sub_sentence = re.sub(
                    r"Error|error.*?[^\u0E00-\u0E7F]+", "", sub_sentence
                )
                error_cleaned = error_cleaned + cleaned_error_sub_sentence
            else:
                error_cleaned = error_cleaned + ""

        else:
            error_cleaned = error_cleaned + sub_sentence

        # For check that 3 indexs before and after the website error not surrounding by thai website
        if (0 <= current_index - 3) and (current_index + 7 < text_lenght):
            three_index_before_error = ord(text[current_index - 3])
            three_index_after_error = ord(text[current_index + 7])
        # check Is it surrounding by thai website or not
        if (
            first_thai_character_order
            <= three_index_before_error
            <= last_thai_character_order
        ) or (
            first_thai_character_order
            <= three_index_after_error
            <= last_thai_character_order
        ):
            clean_flag = 0
        else:
            clean_flag = 1
        previous_index = current_index

    sub_sentence = text[previous_index:text_lenght]
    # check for to be clean or not
    if clean_flag == 1:
        thai_char_pattern = re.compile(r"[\u0E00-\u0E7F]", re.UNICODE)
        matches = thai_char_pattern.findall(sub_sentence)

        if len(matches) > 0:
            cleaned_error_sub_sentence = re.sub(
                r"Error|error.*?[^\u0E00-\u0E7F]+", "", sub_sentence
            )
            error_cleaned = error_cleaned + cleaned_error_sub_sentence
        else:
            error_cleaned = error_cleaned + ""

    else:
        error_cleaned = error_cleaned + sub_sentence
    # return cleaned data
    return error_cleaned


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
    re_text = remove_web_and_tag(text)
    # call remove_error function
    re_text = remove_error(re_text)

    # Strip leading and trailing whitespace
    re_text = re_text.strip()

    return re_text
    # return removed text
