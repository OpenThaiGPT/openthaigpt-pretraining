from openthaigpt_pretraining_model.GPTJ_TH_tokenizer.tokenizer import (
    GPTJToken,
    GPT2Token,
    MergedToken,
)
from openthaigpt_pretraining_model.GPTJ_TH_tokenizer.constants import (
    GPT2_REPO,
    GPTJ_REPO,
    GPT2_LOCAL_DIR,
    GPTJ_LOCAL_DIR,
    GPT2_MERGE_DIR,
    GPTJ_MERGE_DIR,
)
from huggingface_hub import hf_hub_download


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
    # load vcab and merge rule
    hf_hub_download(repo_id=GPT2_REPO, filename="merges.txt", local_dir=GPT2_LOCAL_DIR)
    hf_hub_download(repo_id=GPTJ_REPO, filename="merges.txt", local_dir=GPTJ_LOCAL_DIR)

    gptj_token = GPTJToken(GPTJ_REPO)
    gpt2_token = GPT2Token(GPT2_REPO)
    m_token = MergedToken(GPTJ_REPO, GPT2_REPO, GPTJ_MERGE_DIR, GPT2_MERGE_DIR)

    for idx, test in enumerate(TEXT_TEST_CASES):
        assert [m_token.decode([token]) for token in m_token.encode(test)] == [
            gpt2_token.decode([token]) for token in gpt2_token.encode(test)
        ]

    for idx, token in enumerate(TEXT_TEST_TOKEN):
        assert m_token.encode(token) == gptj_token.encode(token)
