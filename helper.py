import csv

csv_in = open('old_data.csv', 'r')
csv_out = open('new_data.csv', 'w')

writer = csv.writer(csv_out)
counter = 0
last = 1003
for row in csv.reader(csv_in):
	if counter == 0:
		writer.writerow(row)
		counter += 1
		continue
	val = int(row[0].split(".")[0]) + last
	row[0] = "sperm_morph_" + str(val) + ".jpeg"
	print(row)
	writer.writerow(row)
	counter += 1
    

csv_in.close()
csv_out.close()