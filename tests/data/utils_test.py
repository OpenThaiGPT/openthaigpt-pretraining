import os
from openthaigpt_pretraining_data.utils import read_jsonl

FILE_DIR = os.path.dirname(__file__)

READ_JSONL_TEST_CASES = [
    {
        "filename": f"{FILE_DIR}/jsonl_data_example/example.jsonl",
        "expected_output": [
            {"text": "ufabet 88 แทงบอลได้ใน 1 อาทิตย์"},
            {
                "text": "เสือโคร่ง หรือ เสือลายพาดกลอน เป็นสัตว์เลี้ยงลูกด้วยน้ำนมอันดับสัตว์กินเนื้อ มีชื่อวิทยาศาสตร์ว่า Panthera tigris"  # noqa
            },
        ],
    }
]


def test_read_jsonl():
    for test_case in READ_JSONL_TEST_CASES:
        assert read_jsonl(test_case["filename"]) == test_case["expected_output"]


def compare_dataset(clean_dataset, test_dataset):
    test_dataset = [item for item in test_dataset if item["text"] != ""]

    for i in range(len(clean_dataset)):
        assert clean_dataset[i]["text"] == test_dataset[i]["text"]
