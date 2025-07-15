import jax
import jax.numpy as jnp
import numpy as np
import logging
import pickle
from dataclasses import dataclass
from . import full_game_engine as fge
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
        
        # Inicializamos los arrays de JAX
        self.regrets = jnp.zeros((config.max_info_sets, config.num_actions))
        
        # La estrategia inicial es uniforme.
        self.strategies = jnp.ones((config.max_info_sets, config.num_actions)) / config.num_actions
        
        logger.info("PokerTrainer inicializado con el nuevo motor CFR.")

    def train(self, num_iterations: int, save_path: str, save_interval: int):
        logger.info(f"🚀 Iniciando entrenamiento por {num_iterations} iteraciones...")
        
        # Compilamos el train_step una sola vez para un rendimiento óptimo
        jit_train_step = jax.jit(self.train_step)

        for it in range(num_iterations):
            key = jax.random.PRNGKey(self.iteration)
            
            # Pasamos los arrays como argumentos para que JIT funcione correctamente
            self.regrets, self.strategies = jit_train_step(key, self.regrets, self.strategies)
            
            self.iteration += 1
            if (self.iteration % save_interval == 0):
                self.save_model(f"{save_path}_iter_{self.iteration}.pkl")
            logger.info(f"Iteración {self.iteration}/{num_iterations} completada.")
        
        logger.info("🎉 Entrenamiento finalizado.")

    def _cfr_backtracking(self, payoffs: jnp.ndarray, histories: jnp.ndarray):
        # --- Placeholder para el Backtracking ---
        num_updates = 1000
        key = jax.random.PRNGKey(self.iteration)
        dummy_indices = jax.random.randint(key, (num_updates,), 0, self.config.max_info_sets)
        dummy_regrets = jax.random.normal(key, (num_updates, self.config.num_actions))
        return dummy_indices, dummy_regrets

    # Esta función ahora es estática y pura para ser compilable con JIT
    @staticmethod
    def train_step(key: Array, regrets_table: Array, strategies_table: Array, config: TrainerConfig):
        # 1. Simulación
        # La política (logits) que guía la simulación es la estrategia actual.
        # Un truco común es usar log(probabilidades) como logits.
        # Para evitar log(0), añadimos un valor pequeño (epsilon).
        policy_logits = jnp.log(strategies_table + 1e-9)
        
        sim_key, key = jax.random.split(key)
        final_states, payoffs, histories = fge.batch_play_game(
            batch_size=config.batch_size,
            policy_logits=policy_logits,
            key=sim_key
        )

        # 2. Backtracking para obtener regrets (usando el método estático _cfr_backtracking)
        # Nota: El backtracking real necesitaría más información. El placeholder actual es suficiente.
        # En una implementación real, esta lógica sería mucho más compleja.
        # Por ahora, generamos regrets de prueba.
        num_updates = 1000
        dummy_indices = jax.random.randint(key, (num_updates,), 0, config.max_info_sets)
        dummy_regrets = jax.random.normal(key, (num_updates, config.num_actions))
        indices, regrets_from_game = dummy_indices, dummy_regrets
        
        # 3. Acumular regrets
        new_regrets_table = regrets_table.at[indices].add(regrets_from_game)

        # 4. Actualizar estrategias con Regret Matching
        current_regrets = new_regrets_table[indices]
        positive_regrets = jnp.maximum(current_regrets, 0)
        sum_pos_regrets = jnp.sum(positive_regrets, axis=1, keepdims=True)
        
        sum_pos_regrets = jnp.where(sum_pos_regrets > 0, sum_pos_regrets, 1)
        new_strategy_subset = positive_regrets / sum_pos_regrets

        new_strategies_table = strategies_table.at[indices].set(new_strategy_subset)
        
        # Devuelve los nuevos arrays
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