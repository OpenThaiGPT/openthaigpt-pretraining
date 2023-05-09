# coding: utf-8
#
# -- readrows : read rows data from a row number to another row number.
#    You can use this program to check raw row data from Huggingface mC4
#


import datetime
from datasets import load_dataset

print("read mC4 data:")
while True:
    try:
        sknum = int(input("Start to read at rows: "))
        break
    except ValueError:
        print("That's not a valid number. Try again.")

while True:
    try:
        last_row_num = int(input("Enter the last row number to read: "))
        if last_row_num < sknum:
           continue
        break
    except ValueError:
        print("That's not a valid number. Try again.")


print('Please Wait...')
######## Start process

t0 = datetime.datetime.now()
dataset = load_dataset('mc4','th',split='train',  streaming=True)


last_row = last_row_num
skip_num = sknum
selected_rows = 0
stream = dataset.skip(skip_num)

for i, row in enumerate(stream):

	rownum = skip_num + i
	text = row['text']
	if rownum  >= skip_num:
		print('i=',rownum)
		print(text)
		print()
		if rownum >= last_row:
			break


# time spent
t1 = datetime.datetime.now()
tdelta = t1 - t0
print(tdelta)
