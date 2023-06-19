import math
import time

LOOPS = 10 ** 6

print("time.time_ns(): %s" % time.time_ns())
print("time.time(): %s" % time.time())

min_dt = [abs(time.time_ns() - time.time_ns())
          for _ in range(LOOPS)]
min_dt = min(filter(bool, min_dt))
print("min time_ns() delta: %s ns" % min_dt)

min_dt = [abs(time.time() - time.time())
          for _ in range(LOOPS)]
min_dt = min(filter(bool, min_dt))
print("min time() delta: %s ns" % math.ceil(min_dt * 1e9))
