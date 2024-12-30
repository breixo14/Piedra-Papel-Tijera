import tensorflow as tf
import numpy as np
import time

# Constantes del juego
ACTIONS = ["Piedra", "Papel", "Tijera"]
NUM_ACTIONS = len(ACTIONS)
HISTORY_WINDOW = 3  # Número de jugadas previas consideradas
EXPLORATION_RATE = 0.1  # Probabilidad de exploración

# Crear modelo
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(NUM_ACTIONS * HISTORY_WINDOW,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(NUM_ACTIONS, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model

# Determinar ganador
def get_winner(action1, action2):
    if action1 == action2:
        return 0  # Empate
    elif (action1 == 0 and action2 == 2) or \
         (action1 == 1 and action2 == 0) or \
         (action1 == 2 and action2 == 1):
        return 1  # Gana agente 1
    else:
        return 2  # Gana agente 2

# Inicializar agentes
agent1 = create_model()
agent2 = create_model()

# Historial de jugadas
history_agent1 = []
history_agent2 = []

# Configuración
num_games = 1000
batch_size = 32
training_count = 0

# Medir tiempo
start_time = time.time()

# Simulación
win_counts_agent1 = 0

for game in range(num_games):
    # Acción del agente 1
    if len(history_agent1) >= HISTORY_WINDOW:
        input_agent1 = np.zeros((1, NUM_ACTIONS * HISTORY_WINDOW))
        for i in range(HISTORY_WINDOW):
            input_agent1[0, i * NUM_ACTIONS + history_agent2[-(i + 1)]] = 1
        if np.random.rand() < EXPLORATION_RATE:
            action1 = np.random.choice(NUM_ACTIONS)
        else:
            action1_probs = agent1.predict(input_agent1, verbose=0)
            action1 = np.argmax(action1_probs)
    else:
        action1 = np.random.choice(NUM_ACTIONS)  # Acción inicial aleatoria

    # Acción del agente 2
    if len(history_agent2) >= HISTORY_WINDOW:
        input_agent2 = np.zeros((1, NUM_ACTIONS * HISTORY_WINDOW))
        for i in range(HISTORY_WINDOW):
            input_agent2[0, i * NUM_ACTIONS + history_agent1[-(i + 1)]] = 1
        if np.random.rand() < EXPLORATION_RATE:
            action2 = np.random.choice(NUM_ACTIONS)
        else:
            action2_probs = agent2.predict(input_agent2, verbose=0)
            action2 = np.argmax(action2_probs)
    else:
        action2 = np.random.choice(NUM_ACTIONS)  # Acción inicial aleatoria

    # Determinar ganador
    winner = get_winner(action1, action2)

    # Mostrar resultados del juego
    print(f"Juego {game + 1}: Agente 1 ({ACTIONS[action1]}) vs Agente 2 ({ACTIONS[action2]}). Ganador: {winner}")
    if winner == 1:
        win_counts_agent1 += 1

    # Guardar historial
    history_agent1.append(action1)
    history_agent2.append(action2)

    # Entrenar modelos
    if len(history_agent1) >= batch_size:
        x_agent1 = np.zeros((batch_size, NUM_ACTIONS * HISTORY_WINDOW))
        y_agent1 = np.array(history_agent2[-batch_size:])

        x_agent2 = np.zeros((batch_size, NUM_ACTIONS * HISTORY_WINDOW))
        y_agent2 = np.array(history_agent1[-batch_size:])

        for i in range(batch_size):
            for j in range(HISTORY_WINDOW):
                # Asegurarse de que no estamos fuera de rango
                if len(history_agent1) - batch_size + i - j >= 0:
                    x_agent1[i, j * NUM_ACTIONS + history_agent2[-(batch_size - i + j)]] = 1
                    x_agent2[i, j * NUM_ACTIONS + history_agent1[-(batch_size - i + j)]] = 1

        agent1.train_on_batch(x_agent1, y_agent1)
        agent2.train_on_batch(x_agent2, y_agent2)
        training_count += 2

# Medir tiempo total
end_time = time.time()
training_duration = end_time - start_time

# Resumen del entrenamiento
print(f"\nEntrenamiento completado en {training_duration:.2f} segundos.")
print(f"Modelos entrenados {training_count} veces.")
print(f"Victorias del agente 1: {win_counts_agent1} de {num_games} juegos.")
