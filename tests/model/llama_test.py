from openthaigpt_pretraining_model.llama.model import ExampleModel, ExampleModelModified


def test_llama_efficient_attention_parity():
    model1 = ExampleModel()
    model2 = ExampleModelModified()

    x = 5
    assert model1.predict(x) == model2.predict(x)

    x = 0
    assert model1.predict(x) == model2.predict(x)

    x = -3
    assert model1.predict(x) == model2.predict(x)

    x = 202
    assert model1.predict(x) == model2.predict(x)

    x = 1
    assert model1.predict(x) == model2.predict(x)
