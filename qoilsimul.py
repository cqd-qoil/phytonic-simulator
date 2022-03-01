import numpy as np
import sympy as sp
from string import ascii_lowercase
from operator import mul
from functools import reduce
import random
from itertools import product

def co(*args):
    return sp.symbols('x',cls=sp.IndexedBase)[args]

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

    def addElements(self,*args):
        self.timeline = {i:arg for i,arg in enumerate(args)}

    def generateState(self):
        photons = self.sources
        letters = list(ascii_lowercase)
        totalPathModes = sum([photon.pathModes for photon in photons])
        self.pathModes = [sp.symbols(letter,cls=sp.IndexedBase) for letter in letters[:totalPathModes]]
        ltr = iter(letters)
        photonStates = []
        for photon in photons:
            modes = []
            next_ = {}
            if photon.mixed == True:
                sample = random.choices(list(photon.state.keys()),
                                        weights=photon.state.values(),
                                        k=1)
                state = sample[0]
            else:
                state = photon.state
            for pathdof in range(photon.pathModes):
                m = next(ltr)
                state = state.subs({'p{}'.format(pathdof+1):m},
                                   simultaneous=True)
                modes.append(sp.symbols(m,cls=sp.IndexedBase))
                next_[sp.symbols(m,cls=sp.IndexedBase)] = None
            photonStates.append(state)
            photon.o = modes
            photon.next = next_
        try:
            self.state = reduce(mul, photonStates)
        except TypeError:
            raise('No photon states!')

class Photon:
    instances = []
    def __init__(self,pathModes,dof,state):
        self.pathModes = pathModes
        self.dof = dof
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
    _ref = {'path':None,'pol':[H,V]}
    def __init__(self,rule,dof,pathmodes):
        self.id = id(self)
        self.rule = rule
        self.dof = {'path':None} | {k:v for k,v in self._ref.items() if k in dof}
        self.pathmodes = pathmodes
        self.o = np.array(range(pathmodes),dtype=object)
        self._i = np.array(range(pathmodes),dtype=object)

    @property
    def i(self):
        return self._i

    @i.setter
    def i(self,newValue):
        self._i = newValue
        self.o = newValue
        self.dof['path'] = newValue
        keys = list(product(*self.dof.values()))
        self.updateRules(newKeys=keys)

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
        return newIdx, _id

    def genUnitary(self,state):
        unitary = {}
        for arg in sp.preorder_traversal(state):
            if arg.is_Indexed:
                for m in self.rule.keys():
                    flag = False
                    if hasattr(m,'__iter__'):
                        flag = set(m).issubset(set(arg.indices))
                    else:
                        flag = m in arg.indices
                    if flag:
                        break
                if flag:
                    idx, loc = self.genIdx(self.rule.keys(),arg.indices)
                    unitary[arg] = self.rule[loc](*idx.values())
        self.u = unitary

    def updateRules(self,newKeys=None,verbose=False):
        if not newKeys:
            print('dof')
            newKeys = self.dof
        oldKeys = list(self.rule.keys())
        self.rule = dict(zip(newKeys,self.rule.values()))
        if verbose:
            print(' DOF updated:',oldKeys,'=>',newKeys)
            self.prettify()

    def prettify(self):
        args = [(item,) for item in self.rule.keys()]
        for key,func in self.rule.items():
            print(co(key),'--->',func(*args))

class BS(Element):
    def __init__(self):
        dof = ['path']
        pathmodes = 2
        rule = {a: lambda a,b: co(*a)+co(*b),
                b: lambda a,b: co(*a)-co(*b)}
        Element.__init__(self,rule,dof,pathmodes)

class HWP(Element):
    def __init__(self,th):
        dof = ['pol']
        pathmodes = 1
        s = (1/sp.sqrt(2))
        rule = {H: lambda H,V: sp.cos(2*th)*co(*H)+sp.sin(2*th)*co(*V),
                V: lambda H,V: sp.sin(2*th)*co(*H)-sp.cos(2*th)*co(*V)}
        Element.__init__(self,rule,dof,pathmodes)

class PBS(Element):
    def __init__(self):
        dof = ['path','pol']
        pathmodes = 2
        rule = {(a,H): lambda aH,aV,bH,bV: co(*aH),
                (a,V): lambda aH,aV,bH,bV: sp.I*co(*bV),
                (b,H): lambda aH,aV,bH,bV: co(*bH),
                (b,V): lambda aH,aV,bH,bV: sp.I*co(*aV)}
        Element.__init__(self,rule,dof,pathmodes)

class QWP(Element):
    def __init__(self,th):
        dof = ['pol']
        pathmodes = 1
        s = (1/sp.sqrt(2))
        rule = {H: lambda H,V: s*((1+sp.I*sp.cos(2*th))*co(*H)+sp.I*sp.sin(2*th)*co(*V)),
                V: lambda H,V: s*((sp.I*sp.sin(2*th)*co(*H)+(1-sp.I*sp.cos(2*th))*co(*V)))}
        Element.__init__(self,rule,dof,pathmodes)
