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
        self.regrets = jnp.zeros((config.max_info_sets, config.num_actions))
        self.strategies = jnp.ones((config.max_info_sets, config.num_actions)) / config.num_actions
        logger.info("PokerTrainer inicializado con el nuevo motor CFR.")

    def train(self, num_iterations: int, save_path: str, save_interval: int):
        logger.info(f"🚀 Iniciando entrenamiento por {num_iterations} iteraciones...")
        
        # Envolvemos el train_step estático con la config para que JIT funcione
        jit_train_step = jax.jit(
            lambda key, regrets, strategies: PokerTrainer.train_step(key, regrets, strategies, self.config)
        )

        for it in range(num_iterations):
            key = jax.random.PRNGKey(self.iteration)
            
            self.regrets, self.strategies = jit_train_step(key, self.regrets, self.strategies)
            
            self.iteration += 1
            if (self.iteration % save_interval == 0):
                self.save_model(f"{save_path}_iter_{self.iteration}.pkl")
            logger.info(f"Iteración {self.iteration}/{num_iterations} completada.")
        
        logger.info("🎉 Entrenamiento finalizado.")

    @staticmethod
    def train_step(key: Array, regrets_table: Array, strategies_table: Array, config: TrainerConfig):
        policy_logits = jnp.log(strategies_table + 1e-9)
        
        final_states, payoffs, histories = fge.batch_play_game(
            batch_size=config.batch_size,
            policy_logits=policy_logits,
            key=key
        )

        # Placeholder para el Backtracking
        num_updates = 1000
        backtrack_key = jax.random.PRNGKey(0)
        dummy_indices = jax.random.randint(backtrack_key, (num_updates,), 0, config.max_info_sets)
        dummy_regrets = jax.random.normal(backtrack_key, (num_updates, config.num_actions))
        indices, regrets_from_game = dummy_indices, dummy_regrets
        
        new_regrets_table = regrets_table.at[indices].add(regrets_from_game)

        current_regrets = new_regrets_table[indices]
        positive_regrets = jnp.maximum(current_regrets, 0)
        sum_pos_regrets = jnp.sum(positive_regrets, axis=1, keepdims=True)
        sum_pos_regrets = jnp.where(sum_pos_regrets > 0, sum_pos_regrets, 1)
        new_strategy_subset = positive_regrets / sum_pos_regrets

        new_strategies_table = strategies_table.at[indices].set(new_strategy_subset)
        
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