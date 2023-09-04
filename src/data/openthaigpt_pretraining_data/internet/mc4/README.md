# mc4 preprocessing code

This folder contains codes and regex patterns which are the result from the study of mc4 dataset.

The regex patterns are compiled in `pattern.py` and import to use in `preprocess.py`

## How does the code work ?

1. Check the garbage keywords and patterns in each text. If the count of patterns are above the thresholds, the text will be removed.

| Patterns        | Count Thresholds           |
| ------------- |:-------------:|
| Too large row (Larger than 1500 characters)      | 2 |
| Invalid characters (͹) |25|
| Missing tone marks row |25| 
| Gamble keywords |2| 
| Football team names |4| 
| Ads keywords |4| 
| Sale keywords |3| 
| URL that usually related to sales keyword |1| 
| Renting keywords |2| 
| Json-like patterns |20| 
| Programming code |10| 
| Thai specific spam keywords (Ex. ครีมฟอกสี, ยาลดน้ำหนัก) |4| 
| Ghost language (Ex. เธฅเธฐ) |4| 
| Hex code |25| 

3. If the text is not removed by step 1, less severe patterns will be check and remove partially.

| Patterns                          |
|----------------------------------|
| Pagination                        |
| Some HTML tag                     |
| Special unicode characters        |
| Email                             |
| URL                               |
| Hashtag                           |
| Webboard special characters («»)  |
| Menu bar                          |
| Markup                            |
| IP Address                        |
| Telephone number                  |
| Date indicator text ("เผยแพร่เมื่อ", "ประกาศเมื่อ") |
| Other spam ("แก้ไขครั้งสุดท้ายโดย", "แจ้งลิงก์เสีย") |

## Running

The code is imported in `src/data/scripts/internet` and you can use it together with OSCAR code there. 

This code is not meant to be run directly. If you want to run with your custom logic please create folder in `src/data/scripts` and import the function you want.

## Note

- The keywords and patterns are from an observation and experiments on the sample of mc4 dataset.
