import sys
import random

num_chars=10
if len(sys.argv).1:
    num_chars=int(sys.argv[1])
password=""
for index in range(num_chars):
    char_index=random.randint(0,52)
    char_index+=ord('a')
    password+=chr(char_index)
print(password)
#is the end

