from datasets import load_dataset
import pandas as pd
from tqdm.auto import tqdm
import jsonlines

# Load the Thaisum dataset from the Hugging Face datasets library
dataset = load_dataset("thaisum")

# Split the dataset into validation, test, and train sets
validation_thaisum_df = dataset["validation"]
test_thaisum_df = dataset["test"]
train_thaisum_df = pd.DataFrame.from_dict(dataset["train"])

# Create sets of the bodies of the validation and test sets for comparison
validation_thaisum_set = set(validation_thaisum_df["body"])
test_thaisum_set = set(test_thaisum_df["body"])

# Identify the indices of the rows in the train set that have the same body as the validation or test set
idx = []
for i, item in tqdm(train_thaisum_df.iterrows(), total=len(train_thaisum_df)):
    if item["body"] in test_thaisum_set or item["body"] in validation_thaisum_set:
        idx.append(i)

# Create a new train set that excludes the rows with the same body as the validation or test set
train_thaisum_decontaminated_df = train_thaisum_df[~train_thaisum_df.index.isin(idx)]
train_thaisum_decontaminated_df.reset_index(drop=True, inplace=True)

# Write the new train set to a JSONL file with additional metadata
with jsonlines.open("train_thaisum_decontaminated.jsonl", "w") as writer:
    for i in tqdm(range(len(train_thaisum_decontaminated_df))):
        train_dict = {
            "text": f'หัวข้อ: {train_thaisum_decontaminated_df["title"][i]}+\n+เนื้อหา: {train_thaisum_decontaminated_df["body"][i]}+\n+สรุป: {train_thaisum_decontaminated_df["summary"][i]}',
            "source": "Thaisum",
            "source_id": i,
            "created_date": "2020-11-20",
            "updated_date": "2020-11-20",
            "meta": {
                "tag": None
                if train_thaisum_decontaminated_df["type"][i] == ""
                else train_thaisum_decontaminated_df["type"][i],
                "url": train_thaisum_decontaminated_df["url"][i],
            },
        }

        writer.write(train_dict)
