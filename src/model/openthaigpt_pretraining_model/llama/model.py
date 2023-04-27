class ExampleModel:
    def __init__(self):
        pass

    def predict(self, x: int) -> int:
        return x**3


class ExampleModelModified:
    def __init__(self):
        pass

    def predict(self, x: int) -> int:
        return x * x * x