import html
import re


def remove_web_and_tag(text):
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

    text_index = 0
    prevent_out_of_index = 10
    # privote of range - prevent_out_of_index
    len_text = len(text)
    while text_index < len_text - prevent_out_of_index:
        # Chekinf word "error" or "Error"
        if (text[text_index] == "e" or text[text_index] == "E") and (
            text_index < len_text - prevent_out_of_index
        ):
            if (
                text[text_index + 1] == "r"
                and text[text_index + 4] == "r"
                and not (
                    ("ก" < text[text_index + 7] < "๛")
                    or ("ก" < text[text_index - 3] < "๛")
                )
            ):
                temp_of_error_message = ""
                error_counter_index = 0
                # If it found error find untill the last line of error message
                while not ("ก" < text[text_index + error_counter_index] < "๛") and (
                    text_index + error_counter_index < len_text - prevent_out_of_index
                ):
                    temp_of_error_message += text[text_index + error_counter_index]
                    error_counter_index += 1
                # Replace error message
                text = text.replace(temp_of_error_message, "")
        # Update checkpoint
        text_index += 1
        len_text = len(text)
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
    re_text = remove_web_and_tag(text)
    # call remove_error function
    re_text = remove_error(re_text)

    # Strip leading and trailing whitespace
    re_text = re_text.strip()

    return re_text
    # return removed text
