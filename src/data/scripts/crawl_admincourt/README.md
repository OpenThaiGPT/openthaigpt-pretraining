## Config
### crawl_data:
- output_folder: path to output folder

### convert_to_jsonl:
- pdf_path: output folder from crawl_data
- text_rule_file: rule for fix pdf can use from this path (.\src\data\scripts\merge_pdf\pdf_correction_rules_new.txt)

## Run scrape SET annual report
```bash
python src\data\scripts\crawl_admincourt\scrape_admincourt.py
```

## Run convert to jsonl
```bash
python src\data\scripts\crawl_admincourt\convert_to_jsonl.py
```