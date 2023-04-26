

def clean_text(line: str) -> str:
    # () is remainder after link in it filtered out
    return line.strip().replace("\n", " ").replace("()", "")

def filter_out_line(line: str) -> str:
    if len(line) < 80:
        return True
    return False