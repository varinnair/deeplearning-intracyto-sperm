import random

nums = []
num_images = 1003
for i in range(1, num_images + 1):
	if i == 998:
		continue
	nums.append(i)

random.shuffle(nums)
num_train = int(1003 * 0.8)


# ../dataset/images/sperm_morph_722_resized_16_0.jpeg
format_str = "../dataset/images/sperm_morph_{}_resized_{}_{}.jpeg\n"
with open("sperm_all_train2.txt", "w") as txt:
	for i in nums[:num_train]:
		if i == 2:
			print("hid")
		for j in [0, 24]:
			for k in [0, 16]:
				name = format_str.format("%02d" % i, j, k)
				txt.write(name)

with open("sperm_all_val2.txt", "w") as txt:
	for i in nums[num_train:]:
		if i == 2:
			print("hi")
		for j in [0, 24]:
			for k in [0, 16]:
				name = format_str.format("%02d" % i, j, k)
				txt.write(name)