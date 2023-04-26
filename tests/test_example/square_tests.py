SQUARE_TEST_CASES = [
    {"x": 1, "y": 1},
    {"x": 2, "y": 4},
    {"x": 3, "y": 9},
    {"x": 4, "y": 16},
    {"x": 5, "y": 25},
]

CUBE_TEST_CASES = [
    {"x": 1, "y": 1},
    {"x": 2, "y": 8},
    {"x": 3, "y": 27},
    {"x": 4, "y": 64},
    {"x": 5, "y": 125},
]


def test_square():
    for test_case in SQUARE_TEST_CASES:
        assert test_case["x"] ** 2 == test_case["y"]


def test_cube():
    for test_case in CUBE_TEST_CASES:
        assert test_case["x"] ** 3 == test_case["y"]
