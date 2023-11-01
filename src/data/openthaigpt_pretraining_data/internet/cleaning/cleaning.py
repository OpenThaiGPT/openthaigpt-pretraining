from typing import Union, List  # type: ignore
import re  # type: ignore

from .constants import (
    TEXT_DATASET_KEY,
    NEWLINE_CHARACTER,
    COOKIE_KEYWORD,
    SPECAIL_CHARACTER_PATTERN,
    SPECAIL_CHARACTER_RATIO,
    WHITE_SPACE_RATIO,
    MIN_LINE_RATIO,
    MINIMUM_LENGTH,
    MIN_WORD_LENGTH,
    N_LINE,
)


def filter_short_texts(min_length: int = MINIMUM_LENGTH):
    """
    Description:
        Use map with huggingface dataset for filter short texts.
    Args:
        - min_length: minimun lenght of text for filter.
    """

    def filter(sample):
        return len(sample[TEXT_DATASET_KEY]) >= min_length

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
    min_length: int = MIN_WORD_LENGTH,
    min_line_ratio: float = MIN_LINE_RATIO,
    white_space_ratio: float = WHITE_SPACE_RATIO,
    special_character_ratio: float = SPECAIL_CHARACTER_RATIO,
    special_character_pattern: str = SPECAIL_CHARACTER_PATTERN,
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

    def get_ratio(text, total_length, pattern):
        """
        Calculate ratio between lenght text and lenght charactor in pattern.
        """
        return len(re.findall(pattern, text)) / total_length

    def get_line_ratio(line, pattern):
        """
        Calculate ratio between lenght line after remove shot word and lenght full line
        """
        sub_result = []
        sub_lines = re.split(pattern, line)
        if len("".join(sub_lines)) == 0:
            return 0.0
        for sub_line in sub_lines:
            if len(sub_line) > min_length:
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
            get_ratio(text, total_lenght, special_character_pattern)
            >= special_character_ratio
        ):
            return {TEXT_DATASET_KEY: ""}

        # Check short word
        lines = text.split("\n")
        result = []
        for line in lines:
            if (
                get_line_ratio(line, special_character_pattern) > min_line_ratio
                and get_line_ratio(line, " ") > min_line_ratio
            ):
                result.append(line)

        return {TEXT_DATASET_KEY: NEWLINE_CHARACTER.join(result)}

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
        lines = sample[TEXT_DATASET_KEY].split(NEWLINE_CHARACTER)

        if len(lines) < n_lines:
            return {TEXT_DATASET_KEY: sample[TEXT_DATASET_KEY]}

        return_document = lines.copy()
        for i in range(len(lines) - n_lines + 1):
            current_lines = lines[i : i + n_lines]
            hash_line = hash("\n".join(current_lines))
            if hash_line not in hash_history:
                hash_history.append(hash_line)
            else:
                for line in current_lines:
                    if line in return_document:
                        return_document.remove(line)
        return {TEXT_DATASET_KEY: NEWLINE_CHARACTER.join(return_document)}

    return dedup
