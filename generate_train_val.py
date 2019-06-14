import random

prefix = "../dataset/images/"
str_types = ["sperm_morph_{}_resized.jpeg", "sperm_morph_{}_resized_lr.jpeg", "sperm_morph_{}_resized_ud.jpeg", "sperm_morph_{}_resized_lr_ud.jpeg"]
total_num = 1111

names = []

for i in range(1, total_num + 1):
	if i == 998 or i == 492:
		continue
	names.append(i)
	# for s in str_types:
	# 	names.append(s.format("%02d" % i))

print(len(names))

random.shuffle(names)
num_train = int(total_num * 0.9)

counter = 0
with open("new_yolo/sperm_all_train_new.txt", "w") as txt:
	for name in names[:num_train]:
		for s in str_types:
			counter +=1
			txt.write(prefix + s.format("%02d" % name) + "\n")
	print("1", counter)

counter = 0
with open("new_yolo/sperm_all_val_new.txt", "w") as txt:
	print()
	for name in names[num_train:]:
		for s in str_types:
			counter +=1
			txt.write(prefix + s.format("%02d" % name) + "\n")

		# txt.write(prefix + name + "\n")
	print("2", counter)