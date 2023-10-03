from .fsa import FiniteStateAutomaton


def fsa_office1():
    
    symbols_to_phi = {"COFFEE": [0, 1], 
                      "OFFICE":[2, 3], 
                      "MAIL": [4, 5],
                      "DECORATION":[6, 7]}
    
    fsa = FiniteStateAutomaton(symbols_to_phi)

    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")
    fsa.add_state("u3")
    fsa.add_state("u4")

    fsa.add_transition("u0", "u1", "COFFEE")
    fsa.add_transition("u0", "u2", "MAIL")
    fsa.add_transition("u1", "u3", "MAIL")
    fsa.add_transition("u2", "u3", "COFFEE")
    fsa.add_transition("u3", "u4", "OFFICE")

    return fsa

def fsa_delivery_mini1():

    symbols_to_phi = {"A": [0], 
                      "B":[1], 
                      "H": [2]}
    
    fsa = FiniteStateAutomaton(symbols_to_phi)

    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")
    fsa.add_state("u3")

    fsa.add_transition("u0", "u1", "A")
    fsa.add_transition("u1", "u2", "B")
    fsa.add_transition("u2", "u3", "H")

    return fsa


def fsa_delivery1():


    symbols_to_phi = {"A": [0], 
                      "B": [1], 
                      "C": [2], 
                      "H": [3]}
    
    fsa = FiniteStateAutomaton(symbols_to_phi)

    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")
    fsa.add_state("u3")
    fsa.add_state("u4")

    fsa.add_transition("u0", "u1", "A")
    fsa.add_transition("u1", "u2", "B")
    fsa.add_transition("u2", "u3", "C")
    fsa.add_transition("u3", "u4", "H")

    return fsa

def fsa_delivery2():

    symbols_to_phi = {"A": [0], 
                      "B": [1], 
                      "C": [2], 
                      "H": [3]}
    
    fsa = FiniteStateAutomaton(symbols_to_phi)

    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")
    fsa.add_state("u3")
    fsa.add_state("u4")

    fsa.add_transition("u0", "u1", "A")
    fsa.add_transition("u0", "u2", "B")
    fsa.add_transition("u1", "u3", "C")
    fsa.add_transition("u2", "u3", "C")
    fsa.add_transition("u3", "u4", "H")

    return fsa

def fsa_delivery3():

    symbols_to_phi = {"A": [0], 
                      "B": [1], 
                      "C": [2], 
                      "H": [3]}
    
    fsa = FiniteStateAutomaton(symbols_to_phi)

    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")
    fsa.add_state("u3")
    fsa.add_state("u4")
    fsa.add_state("u5")
    fsa.add_state("u6")
    fsa.add_state("u7")


    fsa.add_transition("u0", "u1", "A")
    fsa.add_transition("u0", "u2", "B")
    fsa.add_transition("u0", "u3", "C")
    fsa.add_transition("u1", "u4", "B")
    fsa.add_transition("u3", "u4", "B")
    fsa.add_transition("u1", "u5", "C")
    fsa.add_transition("u2", "u5", "C")
    fsa.add_transition("u2", "u6", "A")
    fsa.add_transition("u3", "u6", "A")
    fsa.add_transition("u4", "u7", "H")
    fsa.add_transition("u5", "u7", "H")
    fsa.add_transition("u6", "u7", "H")

    return fsa