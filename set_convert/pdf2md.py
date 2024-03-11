import os

import pandas as pd
import tabula  # type: ignore


def _append_dataframe(dfs):
    result = pd.concat(dfs, ignore_index=True)
    return result


def read_pdf_table_only(input_filename: str, output_filename: str) -> bool:
    try:
        tables = tabula.read_pdf(input_filename, pages="all")
        concat = _append_dataframe(tables)
        concat.to_csv(output_filename)
    except Exception as ex:
        print(f"[pdf2md] Exception happened during read_pdf_table_only: {ex}")
        return False

    return True


# def read_pdf_table_with_text(input_filename: str, output_filename: str) -> bool:
#     try:
#         tables =


def _example_code():
    pdf_path = "./pdf"
    md_path = "./md"
    for file_name in os.listdir(pdf_path):
        input_file = os.path.join(pdf_path, file_name)
        output_file = os.path.join(md_path, os.path.splitext(file_name)[0] + ".md")
        read_pdf_table_only(input_file, output_file)
