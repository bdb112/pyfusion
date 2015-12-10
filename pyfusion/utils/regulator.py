from time import time as seconds
from time import sleep

class Regulator():
    """ crate a regulator=Regulato(maxcpu) and put regulator.wait() into a loop
    and it will wait so that max cpu is controlled.
    """
    def __init__(self, maxcpu=0.5):
        self.maxcpu = maxcpu
        self.maxcpu = min(self.maxcpu,0.98)
        self.maxcpu = max(self.maxcpu,0.01)
        if self.maxcpu != maxcpu:
            print('changed maxcpu from {i} to {a}'.format(i=maxcpu, a=self.maxcpu))

        self.count = 0
        self.st = seconds()
        self.last_wait = 0
        self.waited = 0
    def wait(self):
        self.count += 1
        avg_time = (seconds() - self.st - self.waited)/self.count
        self.last_wait = max(avg_time*(1-self.maxcpu)/self.maxcpu,0.01)
        print(avg_time, self.last_wait, self.count)
        self.waited += self.last_wait
        sleep(self.last_wait)

