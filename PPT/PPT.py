import random

rock="piedra"
paper="papel"
scissors="tijeras"
jugador_eleccion=None
maquina_eleccion=None
action1=None
action2=None
opciones = [rock, paper, scissors]

def Eleccion_maquina():
    return random.choice(opciones)

def Eleccion_jugador():
    jugador_eleccion=input("R/P/T:  ")
    if   jugador_eleccion=="R":
        return rock
    elif   jugador_eleccion=="P":
        return paper
    else :
        return scissors
   

def juego(action1, action2):
    if action1 == action2:
        return 1

    elif (action1 == "piedra" and action2 == "tijeras") or \
         (action1 == "papel" and action2 == "piedra") or \
         (action1 == "tijeras" and action2 == "papel"):
            return 2
    else:
        return 0

    
