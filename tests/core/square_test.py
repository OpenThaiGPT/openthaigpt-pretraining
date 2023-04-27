from openthaigpt_pretraining.example import SQUARE_TEST_CASES, CUBE_TEST_CASES


def test_square():
    for test_case in SQUARE_TEST_CASES:
        assert test_case["x"] ** 2 == test_case["y"]


def test_cube():
    for test_case in CUBE_TEST_CASES:
        assert test_case["x"] ** 3 == test_case["y"]


# sawadee krub pom:)
