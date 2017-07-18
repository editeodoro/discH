import numpy as np
import multiprocessing as mp

class ParDo:
    '''
    Manage Multiprocess things
    '''
    def __init__(self, nproc, func=None):
        '''
        nproc=number of processor involved
        '''
        self.n=nproc
        self.process=list(np.zeros(nproc))
        self.output=mp.Queue()

        if func is not None: self.set_func(func=func)

    def initialize(self):
        self.output=mp.Queue()

    def set_func(self,func):

        self.func=func

    def _target(self,*args):

        self.output.put(self.func(*args))


    def run(self, array, args):

            target=self._target
            # Initialize process
            if self.n == 1:
                self.process[0] = mp.Process(target=target, args=(array[:],) + args)
            else:
                dim = int(len(array) / self.n)
                for i in range(self.n - 1):
                    start = int(dim * i)
                    end = int(dim * (i + 1))
                    self.process[i] = mp.Process(target=target, args=(array[start:end],) + args)
                self.process[-1] = mp.Process(target=target, args=(array[end:],) + args)

            # Run
            ##start
            for p in self.process:
                p.start()
            ##dequeue
            results = np.concatenate([self.output.get() for p in self.process])
            ##Join
            for p in self.process:
                p.join()
            ##Order
            indxsort = np.argsort(results[:, 0], kind='mergesort')

            return results[indxsort]
        #return results