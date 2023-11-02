from .fsa import FiniteStateAutomaton

def load_fsa(name:str):

    if name == "DeliveryEval-v0-task1":
        return fsa_delivery1()
    elif name == "DeliveryEval-v0-task2":
        return fsa_delivery2()
    elif name == "DeliveryEval-v0-task3":
        return fsa_delivery3()
    elif name == "DeliveryEval-v0-task4":
        return fsa_delivery4()
    elif name == "OfficeComplexEval-v0-task1":
        return fsa_office1()
    elif name == "OfficeRSEval-v0-task1":
        return fsa_office1()
    
def fsa_office1():
    
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
    fsa.add_state("u6")
    fsa.add_state("u7")


    fsa.add_transition("u0", "u1", ["A"])
    fsa.add_transition("u0", "u2", ["B"])
    fsa.add_transition("u0", "u3", ["C"])
    fsa.add_transition("u1", "u4", ["B"])
    fsa.add_transition("u3", "u4", ["B"])
    fsa.add_transition("u1", "u5", ["C"])
    fsa.add_transition("u2", "u5", ["C"])
    fsa.add_transition("u2", "u6", ["A"])
    fsa.add_transition("u3", "u6", ["A"])
    fsa.add_transition("u4", "u7", ["H"])
    fsa.add_transition("u5", "u7", ["H"])
    fsa.add_transition("u6", "u7", ["H"])

    return fsa

def fsa_delivery4():

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
    fsa.add_state("u6")
    fsa.add_state("u7")
    fsa.add_state("u8")


    fsa.add_transition("u0", "u1", ["A"])
    fsa.add_transition("u0", "u2", ["B"])
    fsa.add_transition("u1", "u3", ["H"])
    fsa.add_transition("u2", "u3", ["H"])
    fsa.add_transition("u3", "u4", ["A"])
    fsa.add_transition("u3", "u5", ["B"])
    fsa.add_transition("u4", "u6", ["H"])
    fsa.add_transition("u5", "u6", ["H"])
    fsa.add_transition("u6", "u7", ["C"])
    fsa.add_transition("u7", "u8", ["H"])

    return fsa