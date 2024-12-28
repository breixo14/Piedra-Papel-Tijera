import random

rock="piedra"
paper="papel"
scissors="tijeras"
jugador_eleccion=None
maquina_eleccion=None

opciones = [rock, paper, scissors]

def Eleccion_maquina():
    maquina_eleccion=random.choice(opciones)

def Eleccion_jugador():
    jugador_eleccion=input("R/P/T")
    if   jugador_eleccion=="R":
        jugador_eleccion=rock
    if   jugador_eleccion=="P":
        jugador_eleccion=paper
    else :
        jugador_eleccion=scissors
    print(jugador_eleccion)

def juego():
    if jugador_eleccion==maquina_eleccion:
        print("Empate")
    if jugador_eleccion==rock and maquina_eleccion==scissors: 
        print("Ganaste")
    if jugador_eleccion==paper and maquina_eleccion==rock:
        print("Ganaste")
    if jugador_eleccion==scissors and maquina_eleccion==paper:
        print("Ganaste")
    else:
        print("Perdiste")
