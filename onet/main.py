from docxlatex import Document
import json
import re
import os

# const
docx_folder = "docx"


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
    pattern = r'^\d+\)'
    match = re.match(pattern, txt)
    return bool(match)


def separate_by_q(txt):
    return re.split('\n{5,}', txt)


def unwanted_questions(txt):
    patterns = [r'IMAGE#\d+-image\d+']
    for pattern in patterns:
        if re.findall(pattern, txt):
            return True
    return False


def transform_docx2json(file_name):
    doc_txt = read_docx(file_name)
    split_by_q = separate_by_q(doc_txt)
    remove_empty = list(filter(is_something_list, split_by_q))

    output = []
    q_no = 1

    for question_text in remove_empty:
        if unwanted_questions(question_text):
            continue

        q = question_text.split('\n')
        questions = []
        answers = []

        for x in q:
            if check_question_format(x):
                answers.append(x)
            else:
                questions.append(x)
        output.append({
            'no': q_no,
            'question': ' '.join(questions).strip(),
            'answers': answers,
            'answer': None
        })
        q_no += 1

    return output


def list_files_in_folder(folder_path):
    file_list = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)

    return file_list


file_lst = list_files_in_folder(docx_folder)
for file_name in file_lst:
    if '.docx' in file_name:
        q_a = transform_docx2json(file_name)
        save_json(q_a, f'result-{file_name.split(".")[0]}-result.json')
