from openthaigpt_pretraining_model.gptj.gptj_model_xformers import get_output
import torch


def test_gptj_xformers_attention():
    pretrained_name = "hf-internal-testing/tiny-random-gptj"
    input_text = "Yo"
    output_xformers = get_output(
        pretrained_name=pretrained_name, use_xformers=True, input_text=input_text
    )
    output_originals = get_output(
        pretrained_name=pretrained_name, use_xformers=False, input_text=input_text
    )

    with torch.no_grad():
        assert torch.all(torch.abs(output_xformers - output_originals) <= 1e-6)
