from .fsa import FiniteStateAutomaton

def load_fsa(name:str):

    if name == "DeliveryEval-v0-task1":
        return fsa_delivery1()
    elif name == "DeliveryEval-v0-task2":
        return fsa_delivery2()
    elif name == "DeliveryEval-v0-task3":
        return fsa_delivery3()
    elif name == "OfficeEval-v0-task1":
        return fsa_office1()
    elif name == "OfficeEval-v0-task2":
        return fsa_office2()
    elif name == "OfficeEval-v0-task3":
        return fsa_office3()
    elif name == "OfficeRSEval-v0-task1":
        return fsa_office1()
    elif name in ("DoubleSlitEval-v0-task1", "DoubleSlitEval-v1-task1", 'DoubleSlitRSEval-v0-task1', "IceCorridorEval-v0-task1"):
        return fsa_double_slit1()
    
def fsa_double_slit1():
    
    symbols_to_phi =  {"O1": 0, 
                       "O2": 1}

    fsa = FiniteStateAutomaton(symbols_to_phi)

    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")

    fsa.add_transition("u0", "u1", ["O1"])
    fsa.add_transition("u0", "u2", ["O2"])

    return fsa
    
def fsa_office1():

    # Sequential: Get coffe, then mail, then go to an office
    # COFFEE -> MAIL -> OFFICE

    symbols_to_phi =  {"C1": 0, 
                       "C2": 1, 
                       "O1": 2, 
                       "O2": 3, 
                       "M1": 4, 
                       "M2": 5, }

    fsa = FiniteStateAutomaton(symbols_to_phi)

    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")
    fsa.add_state("u3")

    fsa.add_transition("u0", "u1", ["C1", "C2"])
    fsa.add_transition("u1", "u2", ["M1", "M2"])
    fsa.add_transition("u2", "u3", ["O1", "O2"])

    return fsa

def fsa_office2():

    # OR: Get coffee or mail, then go to an office
    # (COFFEE v MAIL) -> OFFICE

    symbols_to_phi =  {"C1": 0, 
                       "C2": 1, 
                       "O1": 2, 
                       "O2": 3, 
                       "M1": 4, 
                       "M2": 5, }

    fsa = FiniteStateAutomaton(symbols_to_phi)

    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")
    fsa.add_state("u3")

    fsa.add_transition("u0", "u1", ["C1", "C2"])
    fsa.add_transition("u0", "u2", ["M1", "M2"])
    fsa.add_transition("u1", "u3", ["O1", "O2"])
    fsa.add_transition("u2", "u3", ["O1", "O2"])

    return fsa



def fsa_office3():

    # Composed: Get coffee AND email in any order, then go to office
    # (COFFEE ^ MAIL) -> OFFICE
    
    symbols_to_phi =  {"C1": 0, 
                       "C2": 1, 
                       "O1": 2, 
                       "O2": 3, 
                       "M1": 4, 
                       "M2": 5, }

    fsa = FiniteStateAutomaton(symbols_to_phi)

    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")
    fsa.add_state("u3")
    fsa.add_state("u4")

    fsa.add_transition("u0", "u1", ["C1", "C2"])
    fsa.add_transition("u0", "u2", ["M1", "M2"])
    fsa.add_transition("u1", "u3", ["M1", "M2"])
    fsa.add_transition("u2", "u3", ["C1", "C2"])
    fsa.add_transition("u3", "u4", ["O1", "O2"])

    return fsa

def fsa_delivery1():

    # Composite: Get coffe or mail in any order, then go to an office
    # (COFFEE ^ MAIL) -> OFFICE

    symbols_to_phi = {"A": 0, 
                      "B": 1, 
                      "C": 2, 
                      "H": 3}
    
    fsa = FiniteStateAutomaton(symbols_to_phi)

    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")
    fsa.add_state("u3")
    fsa.add_state("u4")

    fsa.add_transition("u0", "u1", ["A"] )
    fsa.add_transition("u1", "u2", ["B"] )
    fsa.add_transition("u2", "u3", ["C"] )
    fsa.add_transition("u3", "u4", ["H"] )

    return fsa

def fsa_delivery2():

    # OR: Go to A "OR" B, then C, then H.
    # (A v B ) -> C -> H

    
    symbols_to_phi = {"A": 0, 
                      "B": 1, 
                      "C": 2, 
                      "H": 3}
    
    fsa = FiniteStateAutomaton(symbols_to_phi)

    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")
    fsa.add_state("u3")
    fsa.add_state("u4")

    fsa.add_transition("u0", "u1", ["A"] )
    fsa.add_transition("u0", "u2", ["B"] )
    fsa.add_transition("u1", "u3", ["C"] )
    fsa.add_transition("u2", "u3", ["C"] )
    fsa.add_transition("u3", "u4", ["H"] )

    return fsa

def fsa_delivery3():

    # OR: Go to A "OR" B, then C, then H.
    # (A v B ) -> C -> H
    
    symbols_to_phi = {"A": 0, 
                      "B": 1, 
                      "C": 2, 
                      "H": 3}
    
    fsa = FiniteStateAutomaton(symbols_to_phi)

    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")
    fsa.add_state("u3")
    fsa.add_state("u4")
    fsa.add_state("u5")

    fsa.add_transition("u0", "u1", ["A"] )
    fsa.add_transition("u0", "u2", ["B"] )
    fsa.add_transition("u1", "u3", ["B"] )
    fsa.add_transition("u2", "u3", ["A"] )
    fsa.add_transition("u3", "u4", ["C"] )
    fsa.add_transition("u3", "u4", ["H"] )

    return fsa
