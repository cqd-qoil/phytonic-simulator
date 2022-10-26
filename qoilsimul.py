import numpy as np
import sympy as sp
from string import ascii_lowercase
from operator import mul
from functools import reduce
import random
from itertools import product,chain

H,V = sp.symbols('H V', cls=sp.IndexedBase)
p1,p2 = sp.symbols('p1 p2', cls=sp.IndexedBase)

def co(*args):
    return sp.symbols('x',cls=sp.IndexedBase)[args]

class Experiment:
    def __init__(self):
        self.sources = []
        self.elements = {}
        self.detectors = []
        self.runs = 1
        print('Experiment initialised.')

    def emptyMode(self):
        m = sp.symbols(next(self._ltr),cls=sp.IndexedBase)
        self.pathModes.append(m)
        return m

    def addModes(self,*args):
        self.modes = [mode for mode in args]

    def addPhotons(self,*args):
        self.sources = [photon for photon in args]
        self.generateState()

    def addElements(self,*args):
        self.elements = {(i+1):arg for i,arg in enumerate(args)}

    def addDetectors(self,detectors):
        self.detectors = detectors

    def generateState(self):
        photons = self.sources
        letters = list(ascii_lowercase)
        totalPathModes = sum([photon.pathModes for photon in photons])
        self.pathModes = [sp.symbols(letter,cls=sp.IndexedBase) for letter in letters[:totalPathModes]]
        self._ltr = iter(letters)
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
                m = next(self._ltr)
                state = state.subs({'p{}'.format(pathdof+1):m},
                                   simultaneous=True)
                modes.append(sp.symbols(m,cls=sp.IndexedBase))
                next_[sp.symbols(m,cls=sp.IndexedBase)] = None
            photonStates.append(state)
            photon.o = modes
        try:
            self.state = reduce(mul, photonStates)
        except TypeError:
            raise('No photon states!')

    def generateCircuit(self):
        if len(self.sources) == 0:
            print('No photon sources in the experiment.')
        elif len(self.elements) == 0:
            print('No optical elements in the circuit.')
        else:
            self.circuit = ''
            for m in self.pathModes:
                symb = '.'
                st = '{} : '.format(m)
                if m in list(chain.from_iterable([p.o for p in self.sources])):
                    symb = '-'
                    st += 'p -'
                else:
                    st += symb*3
                for op in self.elements.values():
                    step = len(op.label)
                    if m in op.i:
                        symb = '-'
                        st+= op.label
                    else:
                        st += step*symb
#                     step = '---'
                    st += symb
                st += '--'

                if m in self.detectors.i:
                    st += 'D'
                else:
                    st += '-'
                self.circuit += st + '\n'

    def build(self):
        states = {}
        # Run experiment
        self.generateState()
        states[0] = self.state
        for i,element in self.elements.items():
            element.genUnitary(self.state)
            self.state = self.state.subs(element.u,simultaneous=True)
            states[i] = self.state
        self.result = states
        self.coincidence()

    def simulate(self):
        print('Simulating experiment...')
        print('> Total runs: ',self.runs)

        resultsDict = dict()
        resultsDict['sources'] = {i:photon.state for i,photon in enumerate(self.sources)}
        resultsDict['elements'] = {i:[element.label,
                                      element.dof] for i,element in self.elements.items()}
        resultsDict['data'] = dict()
        for run in range(self.runs):
            individualRunDict = dict()
            self.build()
            individualRunDict['timeline'] = self.result
            individualRunDict['startState'] = self.result[0]
            resultsDict['data'][run] = individualRunDict
        print('Simulation done.')
        self.summary = resultsDict

    def _calculateTermAmplitudes(self):
        amplitudes = {}
        for arg in sp.preorder_traversal(self.state.expand()):
            if arg.is_Mul:
                amp,factors = sp.factor_list(arg)
                # Create boolean mask for creation operators in term
                mask = [sp.Symbol('x') in i[0].free_symbols for i in factors]
                terms = [f for f,m in zip(factors,mask) if m]
                if not len(terms):
                    # If no modes, pass
                    continue
                currArg = [item[0]**item[1] for item in terms]
                currArg = reduce((lambda x, y: x * y),currArg)

                # Symbolic amplitudes (e.g. cos(theta), 2k, etc.)
                symbAmp = [f for f,m in zip(factors,mask) if not m]

                # Mode labels (e.g. a, b, H, V, t1, etc.)
                idx = [x for y in terms for x in y[0].indices]

                termAmplitude = amp
                for item in symbAmp:
                    termAmplitude *= (item[0]**item[1])

                if currArg in amplitudes.keys():
                    amplitudes[currArg]['amp'] += termAmplitude
                else:
                    amplitudes[currArg] = {'amp':termAmplitude,
                                          'idx':set(idx)}
        self._stateAmplitudes = amplitudes

    def coincidence(self):
        self._calculateTermAmplitudes()
        coincidenceProb = 0
        postSelectedState = 0
        for term,values in self._stateAmplitudes.items():
            if set(self.detectors.i).issubset(values['idx']):
                coincidenceProb += values['amp']**2
                postSelectedState += values['amp']*term

        self.successProbability = coincidenceProb
        self.postSelectedState = postSelectedState

    def _coincidence(self):
        self.detectors.coincidence(self.state)
        self.successProbability = self.detectors.coincidenceProb
        self.postSelectedState = self.detectors.finalState

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

class Detectors:
    def __init__(self,numberOfDetectors):
        self.id = id(self)
        self.i = np.array([None]*numberOfDetectors,dtype=object)
        self.numberOfDetectors = numberOfDetectors

    def coincidence(self,state):
        projState = 0
        if not any(self.i):
            print('No detectors connected')
            return
        else:
            prob = 0
            for arg in sp.preorder_traversal(state.expand()):
                if arg.is_Mul:
                    # Separate each element of the state in terms of its factors
                    amp,factors = sp.factor_list(arg)
                    idx = []
                    symbAmp = 1
                    for f in factors:
                        if (sp.symbols('x') in f[0].atoms()):
                            # If 'x' is in a factor it means that it's a creation
                            # operator and then we take the indices. Else we just
                            # treat it as a "symbolic amplitude" factor.
                            idx += f[0].indices
                        else:
                            symbAmp *= f[0]**f[1]
                    if set(self.i).issubset(set(idx)):
                        # If the detector inputs are found in the expression then
                        # that means it belongs to a coincidence. We then take
                        # the associated amplitude and add the element to the
                        # projected state.
                        prob += (amp*symbAmp)**2
                        projState += arg
            self.coincidenceProb = prob
            self.finalState = projState

class Element:
    _ref = {'path':None,'pol':[H,V]}
    def __init__(self,rule,dof,pathmodes):
        self.id = id(self)
        self.rule = rule
        self.dof = {'path':None}
        self.dof = self.dof | {k:v for k,v in self._ref.items() if k in dof}
        self.pathmodes = pathmodes
        self.o = np.array([None]*pathmodes,dtype=object)
        self._i = np.array([None]*pathmodes,dtype=object)

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

class HWP(Element):
    def __init__(self,th):
        dof = ['pol']
        pathmodes = 1
        rule = {H: lambda H,V: sp.cos(2*th)*co(*H)+sp.sin(2*th)*co(*V),
                V: lambda H,V: sp.sin(2*th)*co(*H)-sp.cos(2*th)*co(*V)}
        self.label = 'HWP'
        Element.__init__(self,rule,dof,pathmodes)

class QWP(Element):
    def __init__(self,th):
        dof = ['pol']
        pathmodes = 1
        s = (1/sp.sqrt(2))
        rule = {H: lambda H,V: s*((1+sp.I*sp.cos(2*th))*co(*H)+sp.I*sp.sin(2*th)*co(*V)),
                V: lambda H,V: s*((sp.I*sp.sin(2*th)*co(*H)+(1-sp.I*sp.cos(2*th))*co(*V)))}
        self.label = 'QWP'
        Element.__init__(self,rule,dof,pathmodes)

class BS(Element):
    def __init__(self,r=1/sp.Rational(2)):
        dof = ['path']
        pathmodes = 2
        rs = sp.sqrt(r)
        ts = sp.sqrt(1-r)
        rule = {p1: lambda a,b: ts*co(*a)+rs*co(*b),
                p2: lambda a,b: rs*co(*a)-ts*co(*b)}
        self.label = 'BS'
        Element.__init__(self,rule,dof,pathmodes)

class PBS(Element):
    def __init__(self):
        dof = ['path','pol']
        pathmodes = 2
        rule = {(p1,H): lambda aH,aV,bH,bV: co(*aH),
                (p1,V): lambda aH,aV,bH,bV: sp.I*co(*bV),
                (p2,H): lambda aH,aV,bH,bV: co(*bH),
                (p2,V): lambda aH,aV,bH,bV: sp.I*co(*aV)}
        self.label = 'PBS'
        Element.__init__(self,rule,dof,pathmodes)

class PPBSH(Element):
    def __init__(self,r = 1/sp.Rational(2)):
        dof = ['path','pol']
        pathmodes = 2
        sr = sp.sqrt(r)
        st = sp.sqrt(1-r)
        rule = {(p1,H): lambda aH,aV,bH,bV: st*co(*aH)+sr*sp.I*co(*bH),
                (p1,V): lambda aH,aV,bH,bV: co(*aV),
                (p2,H): lambda aH,aV,bH,bV: sr*sp.I*co(*aH)+st*co(*bH),
                (p2,V): lambda aH,aV,bH,bV: co(*bV)}
        self.label = 'PPBSH'
        Element.__init__(self,rule,dof,pathmodes)

class PPBSV(Element):
    def __init__(self,r = 1/sp.Rational(2)):
        dof = ['path','pol']
        pathmodes = 2
        sr = sp.sqrt(r)
        st = sp.sqrt(1-r)
        rule = {(p1,H): lambda aH,aV,bH,bV: co(*aH),
                (p1,V): lambda aH,aV,bH,bV: st*co(*aV)+sr*sp.I*co(*bV),
                (p2,H): lambda aH,aV,bH,bV: co(*bH),
                (p2,V): lambda aH,aV,bH,bV: sr*sp.I*co(*aV)+st*co(*bV)}
        self.label = 'PPBSV'
        Element.__init__(self,rule,dof,pathmodes)

class PhaseShifter(Element):
    def __init__(self,theta):
        dof = ['path']
        pathmodes = 1
        rule = {p1: lambda a: sp.exp(sp.I*theta)*co(*a)}
        self.label = 'PSh'
        Element.__init__(self,rule,dof,pathmodes)

class Attenuator(Element):
    def __init__(self,theta):
        dof = ['path']
        pathmodes = 1
        rule = {p1: lambda a: theta*co(*a)}
        self.label = 'Att'
        Element.__init__(self,rule,dof,pathmodes)

class Mirror(Element):
    def __init__(self):
        dof = ['pol']
        pathmodes = 1
        rule = {V: lambda V: -co(*V)}
        self.label = 'M'
        Element.__init__(self,rule,dof,pathmodes)
