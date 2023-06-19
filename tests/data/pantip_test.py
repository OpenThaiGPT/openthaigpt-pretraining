from openthaigpt_pretraining_data.pantip.text_cleaning import clean_html_tags

CLEAN_HTML_TAGS_TEST_CASES = [
    {"data": "<h1>Hello world</h1>", "expected_output": "Hello world"},
    {"data": "<h1>Hello world", "expected_output": "Hello world"},
    {"data": "Hello world", "expected_output": "Hello world"},
]


def test_clean_html_tags():
    for test_case in CLEAN_HTML_TAGS_TEST_CASES:
        assert clean_html_tags(test_case["data"]) == test_case["expected_output"]
