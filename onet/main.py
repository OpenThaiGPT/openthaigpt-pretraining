from docxlatex import Document
import json
import re
import os

# const
docx_folder = "docx"
result_file = 'result.json'
NL_BETWEEN_QUESTION = 13


def save_json(data, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


def remove_emptyline(lst):
    return [x for x in lst if x]


def is_something_list(lst):
    for item in lst:
        if item != '':
            return True
    return False


def read_docx(file_name):
    docx = Document(file_name)
    return docx.get_text()


def check_question_format(txt):
    pattern = r'\n{2,}'
    match = re.match(pattern, txt)
    return bool(match)


def separate_by_q_no(txt):
    pattern = r'\d+\.'
    return re.split(pattern, txt)


def transform_docx2json(file_name):
    doc_txt = read_docx(file_name)
    split_by_q = doc_txt.split('\n' * NL_BETWEEN_QUESTION)
    remove_empty = list(filter(is_something_list, split_by_q))

    output = []

    for question_no, question_text in enumerate(remove_empty):
        q = question_text.split('\n')
        questions = []
        answers = []

        for x in q:
            if check_question_format(x):
                answers.append(x)
            else:
                questions.append(x)
        output.append({
            'no': question_no,
            'question': ' '.join(questions).strip(),
            'answer': answers
        })

    return output


def list_files_in_folder(folder_path):
    file_list = []

    # Iterate over all the files in the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)

    return file_list


file_lst = list_files_in_folder(docx_folder)
for file_name in file_lst:
    q_a = transform_docx2json(file_name)
    save_json(q_a, result_file)
