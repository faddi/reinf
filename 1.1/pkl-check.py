

import _pickle as cpickle
import gzip


counter = 0

with gzip.GzipFile('Pong-ram-v0_memory.pkl', 'rb') as f:
  while f.peek(1):
    row = cpickle.load(f)
    counter += 1
    # if row[2] != 0:
    # print(row[2])
    # print(row[2] != 0 )


print(counter)
