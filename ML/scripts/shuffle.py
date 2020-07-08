# needs to be ran in dl-gpu environment with Tensorflow version
# tf.__version__ : '1.8.0'

import random

random.shuffle(sequence, function)

number_list = [7, 14, 21, 28, 35, 42, 49, 56, 63, 70]
print("Original list:", number_list)

random.shuffle(number_list)
print("List after first shuffle:", number_list)

random.shuffle(number_list)
print("List after second shuffle:", number_list)
