from typing import Union, List  # type: ignore
import re  # type: ignore

from .constants import (
    TEXT_DATASET_KEY,
    NEWLINE_CHARACTOR,
    COOKIE_KEYWORD,
    SPECAIL_CHARACTOR_PATTERN,
    SPECAIL_CHARACTOR_RATIO,
    WHITE_SPACE_RATIO,
    MIN_LINE_RATIO,
    MINIMUM_LENGHT,
    MIN_WORD_LENGHT,
    N_LINE,
)


def filter_short_texts(min_lenght: int = MINIMUM_LENGHT):
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
    """  # noqa
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
    """  # noqa

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

        return {TEXT_DATASET_KEY: NEWLINE_CHARACTOR.join(result)}

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
        lines = sample[TEXT_DATASET_KEY].split(NEWLINE_CHARACTOR)

        if len(lines) < n_lines:
            return {TEXT_DATASET_KEY: sample[TEXT_DATASET_KEY]}

        return_document = lines.copy()
        for i in range(len(lines) - 2):
            current_lines = lines[i : i + 3]
            hash_line = hash("\n".join(current_lines))
            if hash_line not in hash_history:
                hash_history.append(hash_line)
            else:
                for line in current_lines:
                    if line in return_document:
                        return_document.remove(line)
        return {TEXT_DATASET_KEY: NEWLINE_CHARACTOR.join(return_document)}

    return dedup
