from datasets import load_dataset
from tqdm.auto import tqdm
import jsonlines

LABEL = ""   # Add your label here

# Load the PRD_news dataset from the Hugging Face Datasets library
dataset = load_dataset('nakcnx/prd_news')

# Extract the training data from the dataset
data = dataset['train']
data_name = data['title']
data_text = data['text']

# Write the training data to a JSONL file
with jsonlines.open('PRD.jsonl', 'w') as writer:
    for i in tqdm(range(len(data))):
        """ Create a dictionary containing the text
        ,source, source_id,created_date,updated_date,
        and meta datafor the current training example"""
        train_dict = {
            'text': f'หัวข้อ: {data_name[i]}+\n+เนื้อหา: {data_text[i]}',
            'source': 'PRD',
            'source_id': i,
            'created_date': '2023-05-21',
            'updated_date': '2023-05-21', 'meta':  LABEL}
        # Write the dictionary to the JSONL file
        writer.write(train_dict)
