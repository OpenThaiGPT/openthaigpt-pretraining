# OpenThaiGPT Dataset Pipeline Overview

Folder `openthaigpt_pretraining_data` contains the preprocess function, EDA notebook, and examples for each dataset

Folder `notebook` contains the EDA or other experiment notebook.

Folder `scripts` contains the main program which will call function in folder `openthaigpt_pretraining_data`

## Description

The OpenThaiGPT Overview dataset pipeline is meticulously designed to fully preprocess raw data, extract essential information, and prepare the dataset in a trainable format. Here's an extended breakdown of its key components:

## Workflow

1. Obtain Raw Data in various formats such as CSV, PDF, etc., and convert it into plain text format. In our projects, we have Thai government dataset and we can obtain the raw dataset by webscraping. The pipeline to scrape can be found [here](scripts/crawl_thaigov)
[https://github.com/OpenThaiGPT/openthaigpt-pretraining/tree/main/src/data/scripts/crawl_thaigov]
2. Transform plain text data from each raw data source into the JSONL format.
    * Internet dataset would be found [here][https://github.com/OpenThaiGPT/openthaigpt-pretraining/tree/main/src/data/scripts/internet]
    * Pantip2G dataset would be found [here][https://github.com/OpenThaiGPT/openthaigpt-pretraining/tree/main/src/data/scripts/pantip_2G]
    * Pantip3G dataset would be found [here][https://github.com/OpenThaiGPT/openthaigpt-pretraining/tree/main/src/data/scripts/pantip_3G]
3. Consolidate all JSONL datasets into a single file. OSCAR and Pantip are required to merge jsonl. Here is the example of OSCAR and Pantip
    * oscar
        * oscar22.jsonl
        * oscar23.jsonl
        * oscar19.jsonl
        * oscarall.jsonl
    * pantip
        * pantip2g.jsonl
        * pantip3g.jsonl
        * pantipall.jsonl
This pipeline can be found [here][https://github.com/OpenThaiGPT/openthaigpt-pretraining/tree/main/src/data/scripts/merge_jsonl] 
4. Partition the JSONL dataset into training, validation, and test datasets. This pipeline can be found [here][https://github.com/OpenThaiGPT/openthaigpt-pretraining/tree/main/src/data/scripts/split_data] to split the data into train/test/eval
5. Anonymize Sensitive Data, which may include personal and residential information, in compliance with PDPA regulations. This pipeline can be found [here][https://github.com/OpenThaiGPT/openthaigpt-pretraining/tree/main/src/data/scripts/blind_pdpa]
6. Convert jsonl to Huggingface format. This pipline can be found [here][https://github.com/OpenThaiGPT/openthaigpt-pretraining/tree/main/src/data/scripts/huggingface_create]
7. Remove duplicate entries from Huggingface datasets through a deduplication process utilizing the N-Gram MinHash approach combined with Locality-Sensitive Hashing. This pipeline can be found [here][https://github.com/OpenThaiGPT/openthaigpt-pretraining/tree/main/src/data/scripts/deduplication] to remove duplication entries.
8. Implement Decontamination measures to prevent data leakage from the training dataset into the evaluation dataset, employing N-Gram MinHash and LSH techniques. This pipeline can be found [here][https://github.com/OpenThaiGPT/openthaigpt-pretraining/tree/main/src/data/scripts/decontamination] 


