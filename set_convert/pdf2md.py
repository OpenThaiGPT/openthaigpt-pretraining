import os

import pandas as pd
import tabula

pdf_path = './pdf'
md_path = './md'


def append_dataframe(dfs):
    result = pd.concat(dfs, ignore_index=True)
    return result


def read_pdf_table_only(input_filename, output_filename):
    tables = tabula.read_pdf(input_filename, pages='all')
    concat = append_dataframe(tables)
    concat.to_csv(output_filename)


for file_name in os.listdir(pdf_path):
    input_file = os.path.join(pdf_path, file_name)
    output_file = os.path.join(md_path, os.path.splitext(file_name)[0] + '.md')
    read_pdf_table_only(input_file, output_file)
