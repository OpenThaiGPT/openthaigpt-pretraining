import re


def is_seo_doc(text: str, max_limit=35, avg_limit=10):
    """
    Description:
        Check if doc is seo or menu content.
        This function is splitting text by spaces and marks, then
        check length of each word and get max length and average length.
        If max length is less than max_limit and average length is less
        than or equal to avg_limit, it is considered as SEO or Menu,
        and it will return True
        Recommend numbers for args (as default) are max_limit = 35,
        avg_limit = 10. These number is based on maximum character that
        Google reccomend to use as site link description when creating
        SEM ads.For the avg_limit number, it is based on the concept of
        UX/UI design which recommend menu label should be around 5-15
        characters
    Args :
        text: input string of content
        max_limit: interger for limit maximum length of SEO phrase
        avg_limit: interger for limit average length of words in doc
    Return:
        is_seo: boolean
    """
    # Split word by punctuation mark
    pattern = re.compile(r"\n|\t|\s|-|:")
    text_sp = pattern.sub(" ", text)
    text_split = text_sp.split()
    # List word len
    text_split_len = [len(item) for item in text_split]
    # Get max len
    max_len = max(text_split_len)
    # Get avg len
    avg_len = sum(text_split_len) / len(text_split_len)
    # Check seo by len and average
    if max_len <= max_limit and avg_len <= avg_limit:
        is_seo = True
    else:
        is_seo = False

    return is_seo


if __name__ == "__main__":
    import jsonlines

    def load_jsonl_to_list(file_path: str) -> list[dict]:
        """
        Description:
            open jsonl file and return list of dict
        Args:
            file_path: Input string of file path.
        Returns:
            result: List of dict
        """
        result = []
        with jsonlines.open(file_path, "r") as reader:
            for obj in reader.iter(type=dict, skip_invalid=True):
                result.append(obj)
        return result

    input_file = "oscar2301th_30000.jsonl"
    data_list = load_jsonl_to_list(input_file)
    id_seo = []
    for text in data_list:
        if is_seo_doc(text["text"]):
            id_seo.append(text["id"])

    len(id_seo)
