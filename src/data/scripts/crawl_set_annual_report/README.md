## Config
### crawl_data:
- csv_path: path to csv file Can load csv file from [This](https://drive.google.com/file/d/1OGrFQ6Cpp3-olLMiqwsNGpeQZfeMyNrc/view) 
- output_folder: path to output folder

### convert_to_jsonl:
- pdf_path: output folder from crawl_data
- text_rule_file: rule for fix pdf can use from this path (.\src\data\scripts\merge_pdf\pdf_correction_rules_new.txt)

## Run scrape SET annual report
```bash
python src\data\scripts\crawl_set_annual_report\scrape_set_annual.py
```

## Run convert to jsonl
```bash
python src\data\scripts\crawl_set_annual_report\convet_to_jsonl.py
```