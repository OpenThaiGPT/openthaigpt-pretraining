from testcases.mc4_testcases import (
    DOCUMENT_REMOVE_TEST_CASES,
    CLEAN_TEXT_TEST_CASES,
    CLEAN_DATASET_TEST_CASES,
)

from openthaigpt_pretraining_data.mc4.preprocess import (
    clean_with_remove_document,
    clean_text,
    clean_dataset,
)
from utils_test import compare_dataset

import copy


def test_document_remove():
    for test_case in DOCUMENT_REMOVE_TEST_CASES:
        assert clean_with_remove_document(test_case["doc"]) == test_case["remove"]


def test_clean_text():
    for test_case in CLEAN_TEXT_TEST_CASES:
        assert clean_text(test_case["doc"]) == test_case["new_doc"]


def test_clean_dataset():
    for test_case in CLEAN_DATASET_TEST_CASES:
        test_case = copy.deepcopy(test_case)
        compare_dataset(clean_dataset(test_case["dataset"]), test_case["new_dataset"])
