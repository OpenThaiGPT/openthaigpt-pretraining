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


def compare_dataset(dataset1, dataset2):
    for i in range(len(dataset1)):
        assert dataset1[i]['text'] == dataset2[i]['text']