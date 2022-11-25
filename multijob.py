import sys
import numpy as np

args = sys.argv[1:]
job_index = int(args[0])
num_jobs = int(args[1])

ls = np.arange(21)
ls = np.array_split(ls, num_jobs)[job_index]

print(f"job_index: {job_index}\nnum_jobs: {num_jobs}\nsub_list: {ls}")
