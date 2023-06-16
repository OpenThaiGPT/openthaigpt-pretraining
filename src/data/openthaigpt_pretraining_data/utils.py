import jsonlines


def read_jsonl(file_path: str):
    data = []
    with jsonlines.open(file_path) as reader:
        for line in reader.iter(skip_invalid=True):
            data.append(line)
    return data
