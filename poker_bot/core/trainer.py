import jax
import jax.numpy as jnp
import numpy as np
import logging
import pickle
from dataclasses import dataclass
from . import full_game_engine as fge
from .full_game_engine import GameState # Importar GameState para type hinting
from jax import Array
from functools import partial

logger = logging.getLogger(__name__)

@dataclass
class TrainerConfig:
    batch_size: int = 128
    num_actions: int = 14
    max_info_sets: int = 50000

class PokerTrainer:

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.iteration = 0
        self.regrets = jnp.zeros((config.max_info_sets, config.num_actions))
        self.strategies = jnp.ones((config.max_info_sets, config.num_actions)) / config.num_actions
        logger.info("PokerTrainer inicializado con el nuevo motor CFR y aprendizaje real.")

    def train(self, num_iterations: int, save_path: str, save_interval: int):
        logger.info(f"🚀 Iniciando entrenamiento por {num_iterations} iteraciones...")
        
        # Envolvemos el train_step estático con la config para que JIT funcione
        # Nota: La lógica de backtracking con bucles de Python no es compilable,
        # así que llamaremos a la función directamente sin JIT por ahora.
        # jit_train_step = jax.jit(partial(PokerTrainer.train_step, config=self.config))

        for it in range(num_iterations):
            key = jax.random.PRNGKey(self.iteration)
            
            # Llamamos a la función directamente. Pasamos los arrays como argumentos
            # para mantener el patrón de pureza funcional.
            self.regrets, self.strategies = self.train_step(key, self.regrets, self.strategies)
            
            self.iteration += 1
            if (self.iteration % save_interval == 0):
                self.save_model(f"{save_path}_iter_{self.iteration}.pkl")
            logger.info(f"Iteración {self.iteration}/{num_iterations} completada.")
        
        logger.info("🎉 Entrenamiento finalizado.")

    def _cfr_backtracking(self, payoffs: Array, histories: Array, initial_states: GameState):
        """
        Implementa un algoritmo CFR real.
        Recorre las trayectorias hacia atrás para calcular y acumular regrets.
        """
        all_indices = []
        all_regrets = []

        # Este bucle se ejecuta en Python, procesando una partida a la vez.
        for i in range(self.config.batch_size):
            history = histories[i]
            payoff_vector = payoffs[i]
            
            # 1. Reconstruir la trayectoria de estados visitados
            states_trajectory = []
            # Obtenemos el estado inicial para esta partida específica
            # jax.tree_util.tree_map es la forma correcta de indexar un Pytree
            current_state = jax.tree_util.tree_map(lambda x: x[i], initial_states)
            
            # Usamos el historial de acciones para recrear la secuencia de estados
            for t in range(fge.MAX_GAME_LENGTH):
                action = int(history[t])
                if action == -1:
                    break # Fin del historial para esta partida
                states_trajectory.append(current_state)
                current_state = fge.step(current_state, action)

            # 2. Backtracking: recorremos la trayectoria hacia atrás para calcular regrets
            if not states_trajectory: continue

            for t in reversed(range(len(states_trajectory))):
                state_t = states_trajectory[t]
                action_t = int(history[t])
                player_t = int(state_t.current_player_idx[0])

                # El valor de estar en este estado es el payoff final que obtuvo el jugador.
                state_value = payoff_vector[player_t]

                # Bucketing/Indexación: ¿A qué 'cajón' de nuestra memoria corresponde este estado?
                # Por ahora, usamos una simplificación: el cajón es simplemente el ID del jugador.
                info_set_index = player_t 
                
                # Calculamos el arrepentimiento para cada acción posible desde este estado.
                action_regrets = np.zeros(self.config.num_actions, dtype=np.float32)
                legal_actions_mask = fge.get_legal_actions(state_t)

                for a in range(self.config.num_actions):
                    if legal_actions_mask[a]:
                        # Valor Contrafactual: ¿qué habría pasado si hubiéramos tomado la acción 'a'?
                        # Simplificación: si 'a' es la acción que tomamos, el valor es el que obtuvimos.
                        # Si 'a' es otra acción legal, asumimos un resultado neutral (cero).
                        if a == action_t:
                            cf_value = state_value
                        else:
                            cf_value = 0.0
                        
                        # Regret = (resultado de la acción alternativa) - (resultado de la acción real)
                        action_regrets[a] = cf_value - state_value
                    else:
                        action_regrets[a] = -1e9 # Regret muy bajo para acciones ilegales

                all_indices.append(info_set_index)
                all_regrets.append(action_regrets)

        if not all_indices:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32).reshape(0, self.config.num_actions)

        indices = np.array(all_indices, dtype=np.int32)
        regrets = np.stack(all_regrets)

        return indices, regrets

    def train_step(self, key: Array, regrets_table: Array, strategies_table: Array):
        # 1. Simulación
        # La política (logits) que guía la simulación es la estrategia actual.
        policy_logits = jnp.log(strategies_table + 1e-9)
        
        # El motor de juego ahora devuelve también los estados iniciales
        final_states, payoffs, histories, initial_states = fge.batch_play_game(
            batch_size=self.config.batch_size,
            policy_logits=policy_logits,
            key=key
        )

        # 2. Backtracking para obtener regrets
        indices, regrets_from_game = self._cfr_backtracking(payoffs, histories, initial_states)
        
        # 3. Acumular regrets (si hay alguno)
        if indices.shape[0] > 0:
            new_regrets_table = regrets_table.at[indices].add(regrets_from_game)
        else:
            new_regrets_table = regrets_table

        # 4. Actualizar estrategias con Regret Matching
        if indices.shape[0] > 0:
            current_regrets = new_regrets_table[indices]
            positive_regrets = jnp.maximum(current_regrets, 0)
            sum_pos_regrets = jnp.sum(positive_regrets, axis=1, keepdims=True)
            sum_pos_regrets = jnp.where(sum_pos_regrets > 0, sum_pos_regrets, 1)
            new_strategy_subset = positive_regrets / sum_pos_regrets
            new_strategies_table = strategies_table.at[indices].set(new_strategy_subset)
        else:
            new_strategies_table = strategies_table
        
        return new_regrets_table, new_strategies_table

    def save_model(self, path: str):
        model_data = {
            'regrets': np.array(self.regrets),
            'strategies': np.array(self.strategies),
            'iteration': self.iteration,
            'config': self.config
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"💾 Modelo guardado en: {path}")

    def load_model(self, path: str):
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.regrets = jnp.array(model_data['regrets'])
        self.strategies = jnp.array(model_data['strategies'])
        self.iteration = model_data['iteration']
        self.config = model_data['config']
        logger.info(f"📂 Modelo cargado desde: {path}")