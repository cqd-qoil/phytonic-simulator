import numpy as np
import sympy as sp
from string import ascii_lowercase
from operator import mul
from functools import reduce
import random

def co(*args):
    return sp.symbols('x',cls=sp.IndexedBase)[args]

class Experiment:
    def __init__(self):
        self.sources = []
        self.objects = []
        self.detectors = []
        self.step = 0
        self.totalSteps = 0
        print('Experiment initialised.')

    def addModes(self,*args):
        self.modes = [mode for mode in args]

    def addPhotons(self,*args):
        self.sources = [photon for photon in args]

    def generateState(self):
        photons = self.sources
        totalPathModes = sum([photon.pathModes for photon in photons])
        self.pathModes = [sp.symbols(letter,cls=sp.IndexedBase) for letter in list(ascii_lowercase)[:totalPathModes]]
        ltr = iter(letters)
        photonStates = []
        for photon in photons:
            if photon.mixed == True:
                sample = random.choices(list(photon.state.keys()),
                                        weights=photon.state.values(),
                                        k=1)
                state = sample[0]
            else:
                state = photon.state
            for pathdof in range(photon.pathModes):
                state = state.subs({'p{}'.format(pathdof+1):next(ltr)},
                                   simultaneous=True)
            photonStates.append(state)
        try:
            self.state = reduce(mul, photonStates)
        except TypeError:
            raise('No photon states!')

class Photon:
    instances = []
    def __init__(self,pathModes,dofs,state):
        self.pathModes = pathModes
        self.dofs = dofs
        self.o = np.zeros(self.pathModes)
        self.instances.append(self)
        self.id = id(self)
        self.generatePhoton(state)

    def generatePhoton(self,state):
        if type(state) == dict:
            self.mixed = True
        else:
            self.mixed = False
        self.state = state

class Element:
    def __init__(self,rule,dofs):
        self.id = id(self)
        self.rule = rule
        self.dofs = dofs

    def replaceTupleVal(self,tup,ix,val):
        lst = list(tup)
        if hasattr(ix,'__iter__'):
            for i in ix:
                lst[i] = val[i]
        else:
            lst[ix] = val
        return tuple(lst)

    def genIdx(self,modes,indices):
        newIdx = {}
        _id = None
        for m in modes:
            if hasattr(m,'__iter__'):
                if set(m).issubset(set(indices)):
                    _id = m
                    ix = [indices.index(item) for item in m]
            else:
                if m in indices:
                    _id = m
                    ix = indices.index(m)
        for m in modes:
            newIdx[m] = self.replaceTupleVal(indices,ix,m)
        return newIdx,_id

    def genUnitary(self,state):
        unitary = {}
        for arg in sp.preorder_traversal(state):
            if arg.is_Indexed:
                for m in self.rule.keys():
                    flag = False
                    if hasattr(m,'__iter__'):
                        flag = set(m).issubset(set(arg.indices))
                    else:
                        flag = m in indices
                    if flag:
                        break
                if flag:
                    idx, loc = self.genIdx(self.rule.keys(),arg.indices)
                    unitary[arg] = self.rule[loc](*idx.values())
        self.u = unitary

    def updateRules(self,verbose=False):
        oldKeys = list(self.rule.keys())
        self.rule = dict(zip(self.dofs,self.rule.values()))
        if verbose:
            print(' DOF updated:',oldKeys,'=>',self.dofs)
            self.prettify()

    def prettify(self):
        args = [(item,) for item in self.rule.keys()]
        for key,func in self.rule.items():
            print(co(key),'--->',func(*args))

class BS(Element):
    def __init__(self):
        dofs = [a,b]
        rule = {a: lambda a,b: co(*a)+co(*b),
                b: lambda a,b: co(*a)-co(*b)}

        Element.__init__(self,rule,dofs)

class HWP(Element):
    def __init__(self,th):
        dofs = [H,V]
        rule = {H: lambda H,V: sp.cos(2*th)*co(*H)+sp.sin(2*th)*co(*V),
                V: lambda H,V: sp.sin(2*th)*co(*H)-sp.cos(2*th)*co(*V)}
        Element.__init__(self,rule,dofs)

class PBS(Element):
    def __init__(self):
        dofs = [(a,H),(a,V),(b,H),(b,V)]
        rule = {(a,H): lambda aH,aV,bH,bV: co(*aH),
                (a,V): lambda aH,aV,bH,bV: sp.I*co(*bV),
                (b,H): lambda aH,aV,bH,bV: co(*bH),
                (b,V): lambda aH,aV,bH,bV: sp.I*co(*aV)}
        Element.__init__(self,rule,dofs)
