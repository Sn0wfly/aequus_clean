import jax
import jax.numpy as jnp
import numpy as np
import logging
import pickle
from dataclasses import dataclass
from . import full_game_engine as fge
from .full_game_engine import GameState
from jax import Array

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
        logger.info("PokerTrainer inicializado con el nuevo motor CFR (modo híbrido).")

    def train(self, num_iterations: int, save_path: str, save_interval: int):
        logger.info(f"🚀 Iniciando entrenamiento por {num_iterations} iteraciones...")
        
        for it in range(num_iterations):
            key = jax.random.PRNGKey(self.iteration)
            self.train_step(key)
            self.iteration += 1
            if (self.iteration % save_interval == 0):
                self.save_model(f"{save_path}_iter_{self.iteration}.pkl")
            logger.info(f"Iteración {self.iteration}/{num_iterations} completada.")
        
        logger.info("🎉 Entrenamiento finalizado.")

    def _cfr_backtracking(self, payoffs: np.ndarray, histories: np.ndarray, initial_states):
        all_indices, all_regrets = [], []

        for i in range(self.config.batch_size):
            history = histories[i]
            payoff_vector = payoffs[i]
            
            states_trajectory = []
            current_state = jax.tree_util.tree_map(lambda x: x[i], initial_states)
            
            for t in range(fge.MAX_GAME_LENGTH):
                action = int(history[t])
                if action == -1: break
                states_trajectory.append(current_state)
                # La función step SÍ está compilada, por lo que esta reconstrucción es rápida
                current_state = fge.step(current_state, action)

            if not states_trajectory: continue
            
            for t in reversed(range(len(states_trajectory))):
                state_t = states_trajectory[t]
                action_t = int(history[t])
                player_t = int(state_t.current_player_idx[0])
                state_value = payoff_vector[player_t]
                info_set_index = player_t
                
                action_regrets = np.zeros(self.config.num_actions, dtype=np.float32)
                legal_actions_mask = fge.get_legal_actions(state_t)

                for a in range(self.config.num_actions):
                    if legal_actions_mask[a]:
                        cf_value = state_value if a == action_t else 0.0
                        action_regrets[a] = cf_value - state_value
                    else:
                        action_regrets[a] = -1e9
                
                all_indices.append(info_set_index)
                all_regrets.append(action_regrets)

        if not all_indices:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32).reshape(0, self.config.num_actions)

        return np.array(all_indices, dtype=np.int32), np.stack(all_regrets)

    def train_step(self, key: Array):
        # La política se basa en la estrategia actual
        # Usamos los regrets como logits, es una técnica válida en CFR.
        policy_logits = self.regrets
        
        final_states, payoffs, histories, initial_states = fge.batch_play_game(
            batch_size=self.config.batch_size, policy_logits=policy_logits, key=key
        )

        # El backtracking se ejecuta en Python puro
        indices, regrets_from_game = self._cfr_backtracking(
            np.array(payoffs), np.array(histories), initial_states
        )
        
        # La actualización de JAX es rápida y se hace en batch
        if indices.shape[0] > 0:
            self.regrets = self.regrets.at[indices].add(jnp.array(regrets_from_game))
            
            # Actualizamos solo las estrategias que han cambiado
            updated_regrets = self.regrets[indices]
            positive_regrets = jnp.maximum(updated_regrets, 0)
            sum_pos_regrets = jnp.sum(positive_regrets, axis=1, keepdims=True)
            sum_pos_regrets = jnp.where(sum_pos_regrets > 0, sum_pos_regrets, 1)
            new_strategy_subset = positive_regrets / sum_pos_regrets
            self.strategies = self.strategies.at[indices].set(new_strategy_subset)

    def save_model(self, path: str):
        model_data = {'regrets': np.array(self.regrets), 'strategies': np.array(self.strategies), 'iteration': self.iteration, 'config': self.config}
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
        logger.info(f"📂 Modelo cargado desde: {path}")import jax
import jax.numpy as jnp
import numpy as np
import logging
import pickle
from dataclasses import dataclass
from . import full_game_engine as fge
from .full_game_engine import GameState
from jax import Array

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
        logger.info("PokerTrainer inicializado con el nuevo motor CFR (Arquitectura JIT Final).")

    def train(self, num_iterations: int, save_path: str, save_interval: int):
        logger.info(f"🚀 Iniciando entrenamiento por {num_iterations} iteraciones...")
        
        for it in range(num_iterations):
            key = jax.random.PRNGKey(self.iteration)
            self.train_step(key)
            self.iteration += 1
            if (self.iteration % save_interval == 0):
                self.save_model(f"{save_path}_iter_{self.iteration}.pkl")
            logger.info(f"Iteración {self.iteration}/{num_iterations} completada.")
        
        logger.info("🎉 Entrenamiento finalizado.")

    def _cfr_backtracking(self, payoffs: np.ndarray, histories: np.ndarray, initial_states):
        all_indices, all_regrets = [], []
        # Este bucle ahora es mucho más simple porque el historial lo da la simulación
        for i in range(self.config.batch_size):
            history = histories[i]
            payoff_vector = payoffs[i]
            
            states_trajectory = []
            current_state = jax.tree_util.tree_map(lambda x: x[i], initial_states)
            
            for t in range(fge.MAX_GAME_LENGTH):
                action = int(history[t])
                if action == -1: break
                states_trajectory.append(current_state)
                current_state = fge.step(current_state, action)

            if not states_trajectory: continue
            
            for t in reversed(range(len(states_trajectory))):
                state_t = states_trajectory[t]
                action_t = int(history[t])
                player_t = int(state_t.current_player_idx[0])
                state_value = payoff_vector[player_t]
                info_set_index = player_t # Placeholder
                
                action_regrets = np.zeros(self.config.num_actions, dtype=np.float32)
                legal_actions_mask = fge.get_legal_actions(state_t)

                for a in range(self.config.num_actions):
                    if legal_actions_mask[a]:
                        cf_value = state_value if a == action_t else 0.0
                        action_regrets[a] = cf_value - state_value
                    else:
                        action_regrets[a] = -1e9
                
                all_indices.append(info_set_index)
                all_regrets.append(action_regrets)

        if not all_indices:
            return jnp.array([], dtype=jnp.int32), jnp.array([], dtype=jnp.float32).reshape(0, self.config.num_actions)

        return jnp.array(all_indices, dtype=jnp.int32), jnp.stack(all_regrets)

    def train_step(self, key: Array):
        # La política (logits) se basa en la estrategia actual.
        policy_logits = self.strategies
        
        # La simulación se ejecuta en un único bloque JIT
        final_states, payoffs, histories, initial_states = fge.batch_play_game(
            batch_size=self.config.batch_size, policy_logits=policy_logits, key=key
        )

        # El backtracking se ejecuta en Python, lo que es aceptable por ahora.
        indices, regrets_from_game = self._cfr_backtracking(
            np.array(payoffs), np.array(histories), initial_states
        )
        
        # La actualización de arrays se hace en JAX para mayor velocidad
        if indices.shape[0] > 0:
            self.regrets = self.regrets.at[indices].add(jnp.array(regrets_from_game))
            
            updated_regrets = self.regrets[indices]
            positive_regrets = jnp.maximum(updated_regrets, 0)
            sum_pos_regrets = jnp.sum(positive_regrets, axis=1, keepdims=True)
            sum_pos_regrets = jnp.where(sum_pos_regrets > 0, sum_pos_regrets, 1)
            new_strategy_subset = positive_regrets / sum_pos_regrets
            self.strategies = self.strategies.at[indices].set(new_strategy_subset)

    def save_model(self, path: str):
        model_data = {'regrets': np.array(self.regrets), 'strategies': np.array(self.strategies), 'iteration': self.iteration, 'config': self.config}
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