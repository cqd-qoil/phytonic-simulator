import numpy as np
import sympy as sp
from string import ascii_lowercase
from operator import mul
from functools import reduce
import random
from itertools import product,chain

H,V = sp.symbols('H V', cls=sp.IndexedBase)
p1, p2 = sp.symbols('p1 p2', cls=sp.IndexedBase)
# sp.symbols('p1:20',cls_=sp.IndexedBase)

def co(*args):
    return sp.symbols('x', cls=sp.IndexedBase)[args]

class Experiment:
    def __init__(self):
        self.sources = []
        self.elements = {}
        self.detectors = []
        self.runs = 1
        print('Experiment initialised.')

    def emptyMode(self):
        m = sp.symbols(next(self._ltr), cls=sp.IndexedBase)
        self.pathModes.append(m)
        return m

    def nextMode(self, pathmode):
        pass

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
                # probability = real**2 + imag**2, otherwise sympy gets weird with phases
                probability = sum([item**2 for item in values['amp'].as_real_imag()])
                coincidenceProb += probability
                # coincidenceProb += values['amp']*(values['amp'].conjugate())
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
    """
    Base class representing an optical element.

    The Element class serves as the base class for different types of optical elements. It provides basic functionality and methods used by various optical elements.

    Parameters:
        rule (dict): A dictionary defining the rules for the optical element. The keys represent specific combinations
                     of input modes, and the values are lambda functions that calculate the output state based on the
                     given input state(s).
        dof (dict): A dictionary representing the degrees of freedom associated with the optical element. The 'path' key
                    is reserved to track the selected path/mode through the element.
        pathmodes (int): An integer representing the number of paths/modes in the optical element.

    Attributes:
        id (int): The unique identifier for the optical element.
        rule (dict): A dictionary defining the rules for the optical element.
        dof (dict): A dictionary representing the degrees of freedom associated
                    with the optical element.
        pathmodes (int): An integer representing the number of paths/modes in
                    the optical element.
        o (numpy array): An array representing the output state of the optical
                    element.
        _i (numpy array): An internal array representing the input state of the
                    optical element.
        u (dict): A dictionary containing the unitary operation associated with
                    the optical element.

    Properties:
        i (numpy array): The input state of the optical element.

    Methods:
        replaceTupleVal(tup, ix, val): Helper method to replace values in a tuple at specific indices.
        genIdx(modes, indices): Helper method to generate new indices based on selected modes.
        genUnitary(state): Generates unitary operations based on the input state and the element's rule.
        updateRules(newKeys=None, verbose=False): Updates the rule of the optical element with new keys if provided.
        prettify(): Prints a prettified version of the rule for the optical element.

    Example:
        # Creating an instance of the Element class
        >>> rule = {('p1',): lambda a: co(*a) * (1 + 2j), ('p2',): lambda b: co(*b) * (3 - 4j)}
        >>> dof = {'path': None, 'some_other_param': 42}
        >>> pathmodes = 2
        >>> element = Element(rule, dof, pathmodes)
    """
    def __init__(self, rule, dof, pathmodes):
        self.id = id(self)
        self.rule = rule
        self.dof = {'path':None}
        self.dof.update(dof)
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
        """
        Replace values in a tuple at specified indices with new values. Used to determine the rules of an optical element that handles some particular degrees of freedom without affecting others.
        Args:
        - tup (tuple): The original tuple.
        - ix (int or iterable of ints): The index or indices at which to replace values.
        - val (any or iterable): The new value(s) to replace at the specified indices.

        Returns:
        tuple: A new tuple with values replaced at the specified indices.

        Example:
        >>> t = [path1,H,time1,freq1,oam1]
        >>> replaceTupleVal(t, [0,2], [path2,time4])
        [path2,H,time4,freq1,oam1]
        """
        lst = list(tup)
        if hasattr(ix,'__iter__'):
            for i,j in zip(ix, range(len(val))):
                lst[i] = val[j]
        else:
            lst[ix] = val
        return tuple(lst)

    def genIdx(self,modes,indices):
        """
        Generate new indices for specified modes in a list of indices.
        Args:
        - modes (iterable): Modes for which new indices will be generated.
        - indices (list): List of original indices.
        Returns:
        tuple: A tuple containing two elements:
        1. A dictionary where keys are the modes and values are the new indices.
        2. The original mode(s) that were found in the 'indices' list.
        """
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
                    unitary[arg] = self.rule[loc](list(idx.values()))
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
            print(co(key),'--->',func(args))

class HWP(Element):
    def __init__(self,theta,label='HWP'):
        dof = {'pol':[H,V]}
        pathmodes = 1                # modes = [H,V]
        rule = {H: lambda modes: sp.cos(2*theta)*co(*modes[0])+sp.sin(2*theta)*co(*modes[1]),
                V: lambda modes: sp.sin(2*theta)*co(*modes[0])-sp.cos(2*theta)*co(*modes[1])}
        self.label = label
        Element.__init__(self,rule,dof,pathmodes)

class QWP(Element):
    def __init__(self,theta,label='QWP'):
        dof = {'pol':[H,V]}
        pathmodes = 1
        s = (1/sp.sqrt(2))           # modes = [H,V]
        rule = {H: lambda modes: s*((1+sp.I*sp.cos(2*theta))*co(*modes[0])+sp.I*sp.sin(2*theta)*co(*modes[1])),
                V: lambda modes: s*((sp.I*sp.sin(2*theta)*co(*modes[0])+(1-sp.I*sp.cos(2*theta))*co(*modes[1])))}
        self.label = label
        Element.__init__(self,rule,dof,pathmodes)

class BS(Element):
    def __init__(self,r=1/sp.Rational(2),label='BS'):
        dof = {'path':None}
        pathmodes = 2
        rs = sp.sqrt(r)
        ts = sp.sqrt(1-r)
        rule = {p1: lambda modes: ts*co(*modes[0])+rs*co(*modes[1]), # modes = [a,b]
                p2: lambda modes: rs*co(*modes[0])-ts*co(*modes[1])} # modes = [a,b]
        self.label = label
        Element.__init__(self,rule,dof,pathmodes)

class PBS(Element):
    def __init__(self,label='PBS'):
        dof = {'pol':[H,V]}
        pathmodes = 2
        rule = {(p1,H): lambda modes: co(*modes[0]),      # modes = [aH,aV,bH,bV]
                (p1,V): lambda modes: sp.I*co(*modes[3]),
                (p2,H): lambda modes: co(*modes[2]),
                (p2,V): lambda modes: sp.I*co(*modes[1])}
        self.label = label
        Element.__init__(self,rule,dof,pathmodes)

class UBS(Element):
    """ Universal beam splitter
    """
    def __init__(self, user_modes, label="UBS", dof_label='user'):
        dof = {dof_label:user_modes}
        pathmodes = 2
        rule = {(p1,user_modes[0]): lambda modes: co(*modes[0]),      # modes = [at1,at2,bt1,bt2]
                (p1,user_modes[1]): lambda modes: sp.I*co(*modes[3]),
                (p2,user_modes[0]): lambda modes: co(*modes[2]),
                (p2,user_modes[1]): lambda modes: sp.I*co(*modes[1])}
        self.label = label
        Element.__init__(self, rule, dof, pathmodes)

class PPBSH(Element):
    def __init__(self,r = 1/sp.Rational(2),label='PPBSH'):
        dof = {'pol':[H,V]}
        pathmodes = 2
        sr = sp.sqrt(r)
        st = sp.sqrt(1-r)
        rule = {(p1,H): lambda modes: st*co(*modes[0])+sr*sp.I*co(*modes[2]),   # modes = [aH,aV,bH,bV]
                (p1,V): lambda modes: co(*modes[1]),
                (p2,H): lambda modes: sr*sp.I*co(*modes[0])+st*co(*modes[2]),
                (p2,V): lambda modes: co(*modes[3])}
        self.label = label
        Element.__init__(self,rule,dof,pathmodes)

class PPBSV(Element):
    def __init__(self,r = 1/sp.Rational(2),label='PPBSV'):
        dof = {'pol':[H,V]}
        pathmodes = 2
        sr = sp.sqrt(r)
        st = sp.sqrt(1-r)
        rule = {(p1,H): lambda modes: co(*modes[0]),                            # modes = [aH,aV,bH,bV]
                (p1,V): lambda modes: st*co(*modes[1])+sr*sp.I*co(*modes[3]),
                (p2,H): lambda modes: co(*modes[2]),
                (p2,V): lambda modes: sr*sp.I*co(*modes[1])+st*co(*modes[3])}
        self.label = label
        Element.__init__(self,rule,dof,pathmodes)

class BD_full(Element):
    def __init__(self, pathmodes=1, walking_pol=H,label='BD'):
        dof = {'path':None, 'pol':[H, V]}
        # Since a single BD can be used for many different paths (and it can always "increase" the total number of path modes), it needs some special consideration.
        paths = sp.symbols('p1:20', cls_=sp.IndexedBase)[:pathmodes+1]
        pols = [H, V]
        modes = list(product(paths, pols)) # [aH,aV,bH,bV,cH,cV, ...] depends on the number of input paths
        rule = dict()
        for j, jdof in enumerate(modes):
            # The last input will always be empty, but we still need to define a rule for both polarisations
            # for consistency purposes. Hence why I check for j < total_modes-2 below.
            if j < len(modes)-2 and jdof[1] == walking_pol:
                # This is the rule for the polarisation that gets displaced.
                # aH -> bH -> cH, etc. (if H walks off)
                # Setting j=j makes sure that the function is always called correctly, instead of depending on the
                # global value of j (figuring this out was a headscratcher...)
                func = lambda modes, j=j: co(*modes[j+2])
                r = {jdof: func}
            else:
                # Not-walking polarisation goes trough, same as both polarisation in last input mode (should be empty)
                func = lambda modes, j=j: co(*modes[j])
                r = {jdof: func}
            rule = {**rule, **r}
        self.label = label
        Element.__init__(self,rule,dof,pathmodes)

class PhaseShifter(Element):
    def __init__(self,theta,label='PSh'):
        dof = {'path':None}
        pathmodes = 1
        rule = {p1: lambda modes: sp.exp(sp.I*theta)*co(*modes[0])} # mode = a
        self.label = label
        Element.__init__(self,rule,dof,pathmodes)

class Attenuator(Element):
    def __init__(self,theta,label='Att'):
        dof = {'path':None}
        pathmodes = 1
        rule = {p1: lambda modes: theta*co(*modes[0])} # same as phase shifter
        self.label = label
        Element.__init__(self,rule,dof,pathmodes)

class Mirror(Element):
    def __init__(self,label='M'):
        dof = {'pol':[H,V]}
        pathmodes = 1
        rule = {V: lambda modes: -co(*modes[0])} # same as phase shifter
        self.label = label
        Element.__init__(self,rule,dof,pathmodes)

class MZI(Element):
    """
    Mach-Zehnder Interferometer (MZI) class.
    Parameters:
        theta (list of reals): A list of two real numbers representing the phase angles of the MZI.
    """
    def __init__(self,theta,label ='MZI'):
        dof = {'path':None}
        pathmodes = 2
        phase_1 = sp.exp(sp.I*theta[0])
        phase_2 = sp.exp(sp.I*theta[1])   # modes = [a,b]
        rule = {p1: lambda modes: co(*modes[0])*(phase_1/2)*(phase_2+1)+co(*modes[1])*(phase_1/2)*(phase_2-1),
                p2: lambda modes: co(*modes[0])*(1-phase_2)/2+co(*modes[1])*(1+phase_2)/2}
        self.label = label
        Element.__init__(self,rule,dof,pathmodes)
