import jax
import jax.numpy as jnp
import numpy as np
import logging
import pickle
from dataclasses import dataclass
from . import full_game_engine as fge
from jax import Array

logger = logging.getLogger(__name__)

@dataclass
class TrainerConfig:
    batch_size: int = 128
    num_actions: int = 14
    max_info_sets: int = 50000
    # (Puedes añadir más parámetros de config si los necesitas)

class PokerTrainer:

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.iteration = 0
        # Inicializamos los arrays de JAX
        key = jax.random.PRNGKey(0)
        self.regrets = jnp.zeros((config.max_info_sets, config.num_actions))
        # Inicializamos la estrategia a una política uniforme
        self.strategies = jnp.ones((config.max_info_sets, config.num_actions)) / config.num_actions
        # Puedes añadir aquí otros arrays que necesites (ej. q_values)
        logger.info("PokerTrainer inicializado con el nuevo motor CFR.")

    def train(self, num_iterations: int, save_path: str, save_interval: int):
        logger.info(f"🚀 Iniciando entrenamiento por {num_iterations} iteraciones...")
        for it in range(num_iterations):
            # Creamos una nueva clave para cada iteración para asegurar aleatoriedad
            key = jax.random.PRNGKey(self.iteration)
            self.train_step(key)
            self.iteration += 1
            if (self.iteration % save_interval == 0):
                self.save_model(f"{save_path}_iter_{self.iteration}.pkl")
            logger.info(f"Iteración {self.iteration}/{num_iterations} completada.")
        logger.info("🎉 Entrenamiento finalizado.")

    def _cfr_backtracking(self, payoffs: jnp.ndarray, histories: jnp.ndarray):
        # --- Placeholder para el Backtracking ---
        # Esta función será mucho más compleja en el futuro.
        # Por ahora, genera regrets aleatorios para info_sets de prueba.

        # Bucketing de prueba: usar el índice del jugador como info_set.
        # En un sistema real, aquí se llamaría a un bucketing complejo.
        # Vamos a simular que encontramos 1000 puntos de decisión.
        num_updates = 1000
        key = jax.random.PRNGKey(self.iteration)
        dummy_indices = jax.random.randint(key, (num_updates,), 0, self.config.max_info_sets)
        
        # Regret de prueba: simplemente un valor aleatorio.
        dummy_regrets = jax.random.normal(key, (num_updates, self.config.num_actions))
        
        return dummy_indices, dummy_regrets

    def train_step(self, key: Array):
        # 1. Simulación
        # La política se toma de las estrategias actuales.
        # Para batch_play_game, necesitamos logits, no probabilidades. Usaremos los regrets como logits.
        policy_logits = self.regrets
        
        sim_key, key = jax.random.split(key)
        final_states, payoffs, histories = fge.batch_play_game(
            batch_size=self.config.batch_size,
            policy_logits=policy_logits,
            key=sim_key
        )

        # 2. Backtracking para obtener regrets
        indices, regrets = self._cfr_backtracking(payoffs, histories)

        # 3. Acumular regrets
        self.regrets = self.regrets.at[indices].add(regrets)

        # 4. Actualizar estrategias con Regret Matching
        current_regrets = self.regrets[indices]
        positive_regrets = jnp.maximum(current_regrets, 0)
        sum_pos_regrets = jnp.sum(positive_regrets, axis=1, keepdims=True)
        
        # Evitar división por cero
        sum_pos_regrets = jnp.where(sum_pos_regrets > 0, sum_pos_regrets, 1)
        new_strategy_subset = positive_regrets / sum_pos_regrets

        self.strategies = self.strategies.at[indices].set(new_strategy_subset)

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