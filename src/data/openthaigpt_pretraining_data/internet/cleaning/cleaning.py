from typing import Union, List
import re

TEXT_DATASET_KEY = "text"
COOKIE_KEYWORD = [
    "ใช้งานคุกกี้",
    "ใช้คุกกี้",
    "เว็บไซต์นี้ใช้คุกกี้",
    "นโยบายคุกกี้",
    "ตั้งค่าคุกกี้",
    "เทคโนโลยีคุกกี้",
    "คุกกี้บนเว็บไซต์",
    "ยินยอมให้เราเก็บคุกกี้ทั้งหมด",
    "นโยบายเกี่ยวกับคุกกี้",
]
SPECAIL_CHARACTOR_PATTERN = r"""[!@#$%^&*()_+={}\[\]:;"\'<>,.?/\|\\`~]"""
WHITE_SPACE_RATIO = 0.1
SPECAIL_CHARACTOR_RATIO = 0.05
MINIMUM_LENGHT = 128
N_LINE = 3
MIN_WORD_LENGHT = 30
MIN_LINE_RATIO = 0.3


def filter_short_texts(min_lenght: int = 256):
    """
    Description:
        Use map with huggingface dataset for filter short texts.
    Args:
        - min_lenght: minimun lenght of text for filter.
    """

    def filter(sample):
        return len(sample[TEXT_DATASET_KEY]) >= min_lenght

    return filter


def filter_keywords(keywords: Union[str, List[str]] = COOKIE_KEYWORD):
    """
    Description:
        Use map with huggingface dataset for filter text with have keywords.
    Args:
        - keywords: string or list of string which is the keyword that you want to delete.
    """
    if isinstance(keywords, str):
        keywords = [keywords]

    def filter(sample):
        for keyword in keywords:
            if keyword in sample[TEXT_DATASET_KEY]:
                return False
        return True

    return filter


def clean_no_meaningful(
    min_lenght: int = MIN_WORD_LENGHT,
    min_line_ratio: float = MIN_LINE_RATIO,
    white_space_ratio: float = WHITE_SPACE_RATIO,
    special_charactor_ratio: float = SPECAIL_CHARACTOR_RATIO,
    special_charactor_pattern: str = SPECAIL_CHARACTOR_PATTERN,
):
    """
    Description:
        Clean no meaningful sentence of document by check.
            - ratio between lenght space and lenght document.
            - ratio between lenght special charactor and lenght document.
            - ratio between lenght line after remove shot word and lenght full line.
    Args:
        - min_lenght: minimum of lenght word.
        - min_line_ratio: ratio between line after remove shot word and lenght full line that want to remove.
        - white_space_ratio: ratio between space charactor and total charactor that want to remove.
        - special_charactor_ratio: ratio between special charactor and total charactor that want to remove.
        - special_charactor_pattern: regex pattern for special charactor.
    """

    def get_ratio(text, total_lenght, pattern):
        """
        Calculate ratio between lenght text and lenght charactor in pattern.
        """
        return len(re.findall(pattern, text)) / total_lenght

    def get_line_ratio(line, pattern):
        """
        Calculate ratio between lenght line after remove shot word and lenght full line
        """
        sub_result = []
        sub_lines = re.split(pattern, line)
        if len("".join(sub_lines)) == 0:
            return 0.0
        for sub_line in sub_lines:
            if len(sub_line) > min_lenght:
                sub_result.append(sub_line)
        return len("".join(sub_result)) / len("".join(sub_lines))

    def cleaner(sample):
        text = sample[TEXT_DATASET_KEY]
        total_lenght = len(text)

        # Check white space
        if get_ratio(text, total_lenght, " ") >= white_space_ratio:
            return {TEXT_DATASET_KEY: ""}
        # Check special charactor
        if (
            get_ratio(text, total_lenght, special_charactor_pattern)
            >= special_charactor_ratio
        ):
            return {TEXT_DATASET_KEY: ""}

        # Check short word
        lines = text.split("\n")
        result = []
        for line in lines:
            if (
                get_line_ratio(line, SPECAIL_CHARACTOR_PATTERN) > min_line_ratio
                and get_line_ratio(line, " ") > min_line_ratio
            ):
                result.append(line)

        return {TEXT_DATASET_KEY: "\n".join(result)}

    return cleaner


def dedup_n_lines(n_lines: int = N_LINE):
    """
    Description:
        Deduplicate by check from n lines
    Args:
        - n_lines: number of line for check deduplicate
    """
    hash_history = []

    def dedup(sample):
        lines = sample["text"].split("\n")

        if len(lines) < n_lines:
            return {"text": sample["text"]}

        return_document = lines.copy()
        for i in range(len(lines) - 2):
            current_lines = lines[i : i + 3]
            hash_line = hash("\n".join(current_lines))
            if not hash_line in hash_history:
                hash_history.append(hash_line)
            else:
                for line in current_lines:
                    if line in return_document:
                        return_document.remove(line)
        return {"text": "\n".join(return_document)}

    return dedup
