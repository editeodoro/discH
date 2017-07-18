import numpy as np
import multiprocessing as mp

class ParDo:
    '''
    Manage Multiprocess things
    '''
    def __init__(self, nproc):
        '''
        nproc=number of processor involved
        '''
        self.n=nproc
        self.process=list(np.zeros(nproc))
        self.output=mp.Queue()

    def initialize(self):
        self.output=mp.Queue()

    def run(self,array,target,args):
        #Initialize process
        if self.n==1: self.process[0]=mp.Process(target=target, args=(array[:],)+args)
        else:
            dim=int(len(array)/self.n)
            for i in range(self.n-1):
                start=int(dim*i)
                end=int(dim*(i+1))
                self.process[i]=mp.Process(target=target, args=(array[start:end],))
            self.process[-1]= mp.Process(target=target, args=(array[end:],))
        #Run
        for p in self.process:
            p.start()
        for p in self.process:
            p.join()
        results=np.concatenate([self.output.get() for p in self.process])
        indxsort=np.argsort(results[:,0], kind='mergesort')
        return results[indxsort]