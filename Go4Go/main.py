import os, random


def remove_items(test_list, item):
 
    # using list comprehension to perform the task
    res = [i for i in test_list if i != item]
 
    return res

#opens random .sgf file
file = open(random.choice(os.listdir(r"C:\Users\Orbix\OneDrive\Desktop\Python_Projects\Go_NN\Go4Go")))
lines = file.readlines()

print(file)

#reads/formats the file data
data = lines[3].split(";")
b = []
w = []
black = []
white = []

new_move_list = []

for i in range(1, len(data)):
    if (i % 2) == 1:
        b.append(data[i])
    else:
        w.append(data[i])

for x in b:
    black.append(x[2:4])

for y in w:
    white.append(y[2:4])

move_list = list(zip(black,white))
file.close()

remove_items(move_list, "B")
remove_items(move_list, "W")

for move in move_list:
    for i in move:
        if ord(i[0])
        x = ord(i[0]) - 96
        y = ord(i[1]) - 96
        new_move_list.append([x,y])

print(new_move_list)
print(ord("i"))
