import random

rock="piedra"
paper="papel"
scissors="tijeras"
jugador_eleccion=None
maquina_eleccion=None

opciones = [rock, paper, scissors]

def Eleccion_maquina():
    return random.choice(opciones)

def Eleccion_jugador():
    jugador_eleccion=input("R/P/T:  ")
    if   jugador_eleccion=="R":
        return rock
    if   jugador_eleccion=="P":
        return paper
    else :
        return scissors
   

def juego():
    jugador=Eleccion_jugador()
    maquina=Eleccion_maquina()

    print("La maquina eligío {} y tu elegiste {}".format(maquina, jugador))
    
    if jugador==maquina:
        print("Empate")
    elif (jugador == "piedra" and maquina == "tijeras") or \
         (jugador == "papel" and maquina == "piedra") or \
         (jugador == "tijeras" and maquina == "papel"):
        print("¡Ganaste!")
    else:
        print("Perdiste")

    
juego()
