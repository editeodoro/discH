import numpy as np
import  multiprocessing as mp
import time

def f(R,Z):

    x=0
    for r in R:
        for z in Z:

            x+=r+z

    return x


Ra=np.linspace(0,10,1000)
Za=np.linspace(0,10,1000)
t1=time.time()
print(f(Ra,Za))
print(time.time()-t1)

def f2(R):

    x=0
    for r in R:
        for z in Za:

            x+=r+z

    return x


pool=mp.Pool(processes=2)
t1=time.time()
print(np.sum(pool.apply_async(f2,args=(Ra,))))
print(time.time()-t1)

nproc=2
process = np.empty(nproc, dtype=object)
dim = int(len(Ra) / nproc)
print(dim)

def fpa(R,Z):

    mp.Queue()


for i in range(nproc - 1):
    start = int(dim * i)
    end = int(dim * (i+1))

    process[i] = mp.Process(target=f, args=(Ra[start:end], Za))
    print(process)
process[-1]=mp.Process(target=f, args=(Ra[end:], Za))

for p in process:
    p.start()
for p in process:
    p.join()

