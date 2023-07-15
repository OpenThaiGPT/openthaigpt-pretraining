import os

docx_path = './docx'
md_path = './md'

for file_name in os.listdir(docx_path):
    input_file = os.path.join(docx_path, file_name)
    output_file = os.path.join(md_path, os.path.splitext(file_name)[0] + '.md')
    os.system(f'pandoc -f docx -t markdown {input_file} -o {output_file}')
