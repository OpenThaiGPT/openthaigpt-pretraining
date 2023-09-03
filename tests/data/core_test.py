from testcases.core_testcases import (
    DOCUMENT_REMOVE_TEST_CASES,
    CHECK_RATIO_TEST_CASES,
    REMOVE_KEYWORDS_TEST_CASES,
    CLEAN_TEXT_TEST_CASES,
    CLEAN_DATASET_TEST_CASES,
)

from openthaigpt_pretraining_data.internet.perplexity.preprocess import (
    contains_document_removal_keywords,
    check_ratio_bad_substring,
    remove_partial_keywords,
    clean_text,
    clean_dataset,
)

import copy


def test_document_remove():
    for test_case in DOCUMENT_REMOVE_TEST_CASES:
        assert (
            contains_document_removal_keywords(test_case["doc"]) == test_case["remove"]
        )


def test_check_ratio():
    for test_case in CHECK_RATIO_TEST_CASES:
        assert check_ratio_bad_substring(test_case["doc"]) == test_case["remove"]


def test_remove_partial_keywords():
    for test_case in REMOVE_KEYWORDS_TEST_CASES:
        assert remove_partial_keywords(test_case["doc"]) == test_case["new_doc"]


def test_clean_text():
    for test_case in CLEAN_TEXT_TEST_CASES:
        assert clean_text(test_case["doc"]) == test_case["new_doc"]


def test_clean_dataset():
    for test_case in CLEAN_DATASET_TEST_CASES:
        test_case = copy.deepcopy(test_case)
        assert clean_dataset(test_case["dataset"]) == test_case["new_dataset"]
