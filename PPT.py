import random

rock="piedra"
paper="papel"
scissors="tijeras"

opciones = [rock, paper, scissors]

def Eleccion_maquina():
    print(random.choice(opciones))

def Eleccion_jugador():
    jugador_eleccion=input("R/P/T")
    if   jugador_eleccion=="R":
        jugador_eleccion=rock
    if   jugador_eleccion=="P":
        jugador_eleccion=paper
    else :
        jugador_eleccion=scissors
    print(jugador_eleccion)

def ganador():
    