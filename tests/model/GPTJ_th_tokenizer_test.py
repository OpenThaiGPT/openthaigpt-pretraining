# import time
from openthaigpt_pretraining_model.GPTJ_TH_tokenizer.tokenizer import (
    GPTJToken,
    GPT2Token,
    MergedToken,
)


TEXT_TEST_CASES = [
    "การวิจัยเกี่ยวกับรูปแบบภาษาที่ใหญ่",
    "น้องโลมานั้นจัดว่าเป็นสัตว์หายากใกล้สูญพันธุ์",
    "พฤษภาคมนี้ได้เวลาเที่ยวฉ่ำรับฤดูฝน",
    "อยากกินโมโม่พาราไดซ์",
    "รายละเอียดและหลักเกณฑ์การคัดเลือก",
]

TEXT_TEST_TOKEN = [
    "ResNets solve the famous known vanishing gradient.",
    "Each of the layers follow the same pattern.",
    "ResNet on the paper is mainly explained for ImageNet dataset.",
]


def test_merge_tokenizer():
    gptj_token = GPTJToken()
    gpt2_token = GPT2Token()
    m_token = MergedToken()

    for idx, test in enumerate(TEXT_TEST_CASES):
        assert [m_token.decode([token]) for token in m_token.encode(test)] == [
            gpt2_token.decode([token]) for token in gpt2_token.encode(test)
        ]

    for idx, token in enumerate(TEXT_TEST_TOKEN):
        assert m_token.encode(token) == gptj_token.encode(token)
