from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import numpy as np
import matplotlib.pyplot as plt
from mesa import Agent


# Clase para representar a los agentes (niños)
class ChildAgent(Agent):
    def __init__(self, unique_id, model):
        self.unique_id = unique_id
        self.model = model

        # Atributos cognitivos
        self.memory = np.random.randint(5, 8)  # Iniciar con un valor aleatorio
        self.attention_span = np.random.uniform(10, 20)
        self.metacognition = np.random.uniform(0.6, 0.9)

        # Atributos emocionales
        self.self_regulation = np.random.uniform(0.5, 1)
        self.frustration = 0
        self.self_esteem = np.random.uniform(1, 5)

        # Atributos interpersonales
        self.communication = np.random.uniform(0.5, 1)
        self.collaboration = np.random.uniform(0.5, 1)
        self.empathy = np.random.uniform(0.5, 1)

        self.pos = None  # Posición inicial (se asignará en el modelo)

    def step(self):
        # Actualización de atributos en cada paso
        self.memory = min(10, self.memory + 0.1)  # Incremento gradual de memoria
        self.attention_span = max(5, self.attention_span - 0.1)  # Atención disminuye ligeramente
        if np.random.random() < self.metacognition:
            self.memory += 0.05  # Incremento extra por metacognición

        # Emociones
        self.frustration = max(0, self.frustration - 0.05)  # Reducir frustración
        if np.random.random() > self.self_regulation:
            self.frustration += 0.1
        self.self_esteem += 0.1 if self.frustration < 0.3 else -0.1  # Ajustar autoestima

        # Interacciones
        # Interacciones con vecinos (cercanía en la grilla)
        for neighbor in self.model.grid.get_neighbors(self.pos, moore=True, include_center=False):
            if isinstance(neighbor, ChildAgent):
                # Interacción de colaboración: compartir niveles de colaboración
                self.collaboration = (self.collaboration + neighbor.collaboration) / 2
                
                # Interacción de frustración: la frustración de un agente puede afectar a los vecinos
                if self.frustration > 0.7:
                    neighbor.frustration = min(1, neighbor.frustration + 0.05)  # Aumentar frustración del vecino

                # Interacción de empatía: si un agente tiene alta empatía, reduce la frustración de sus vecinos
                if self.empathy > 0.7:
                    neighbor.frustration = max(0, neighbor.frustration - 0.05)  # Reducir frustración del vecino

        # Actualización de atributos personales
        self.memory = min(10, self.memory + 0.1)  # Incremento gradual de la memoria
        self.attention_span = max(5, self.attention_span - 0.1)  # La atención disminuye con el tiempo

        # Metacognición: si el agente reflexiona (probabilidad dada por metacognition)
        if np.random.random() < self.metacognition:
            self.memory += 0.05

        # Emociones y autorregulación
        self.frustration = max(0, self.frustration - 0.05)  # Reducir frustración
        if np.random.random() > self.self_regulation:
            self.frustration += 0.1  # Aumentar frustración si la autorregulación falla

        # Ajuste de autoestima
        if self.frustration < 0.3:
            self.self_esteem += 0.1
        else:
            self.self_esteem -= 0.1

        # Interacciones para colaboración y empatía
        self.collaboration = min(1, self.collaboration + 0.01)
        self.empathy = min(1, self.empathy + 0.01)

# Clase para representar el modelo
class ClassroomModel(Model):
    def __init__(self, num_agents, width, height):

        self.random = np.random.RandomState()

        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)

        # Listas para recopilar datos
        self.all_memory = []
        self.all_frustration = []
        self.all_attention = []

        # Crear agentes
        for i in range(self.num_agents):
            agent = ChildAgent(i, self)
            self.schedule.add(agent)

            # Colocar a los agentes en posiciones aleatorias dentro de la grilla
            x = self.random.randint(0, self.grid.width)
            y = self.random.randint(0, self.grid.height)
            self.grid.place_agent(agent, (x, y))

    def step(self):
        # Recopilar los datos de todos los agentes por cada paso
        memory_data = []
        frustration_data = []
        attention_data = []

        for agent in self.schedule.agents:
            memory_data.append(agent.memory)  # Agregar la memoria de cada agente
            frustration_data.append(agent.frustration)  # Agregar la frustración de cada agente
            attention_data.append(agent.attention_span)  # Agregar la atención de cada agente

        # Guardar los datos de este paso
        self.all_memory.append(memory_data)
        self.all_frustration.append(frustration_data)
        self.all_attention.append(attention_data)

        # Avanzar al siguiente paso en la simulación
        self.schedule.step()

# Función para graficar resultados
# Función para graficar resultados
def plot_evolution(model):
    # Graficar la evolución de la Memoria
    plt.figure(figsize=(10, 6))
    for agent_id in range(model.num_agents):
        plt.plot([time_step[agent_id] for time_step in model.all_memory], label=f'Agente {agent_id} - Memoria')
    plt.title("Evolución de la Memoria")
    plt.xlabel("Tiempo")
    plt.ylabel("Memoria")
    plt.legend()
    plt.show()

    # Graficar la evolución de la Frustración
    plt.figure(figsize=(10, 6))
    for agent_id in range(model.num_agents):
        plt.plot([time_step[agent_id] for time_step in model.all_frustration], label=f'Agente {agent_id} - Frustración')
    plt.title("Evolución de la Frustración")
    plt.xlabel("Tiempo")
    plt.ylabel("Frustración")
    plt.legend()
    plt.show()

    # Graficar la evolución de la Atención
    plt.figure(figsize=(10, 6))
    for agent_id in range(model.num_agents):
        plt.plot([time_step[agent_id] for time_step in model.all_attention], label=f'Agente {agent_id} - Atención')
    plt.title("Evolución de la Atención")
    plt.xlabel("Tiempo")
    plt.ylabel("Atención")
    plt.legend()
    plt.show()


# Código principal
if __name__ == "__main__":
    model = ClassroomModel(10, 10, 10)  # 10 agentes en una grilla 10x10
    for _ in range(100):  # Simular 100 pasos
        model.step()

    plot_evolution(model)
