import random
non_empty_episode = "non-empty-video-v4.txt"

lines = open(non_empty_episode, 'r').readlines()

f_train = "train.txt"
f_test = "test.txt"
f_train = open(f_train, 'w')
f_test = open(f_test, 'w')

for line in lines:
    if random.random() < 0.2:
        f_test.write(line)
    else:
        f_train.write(line)

