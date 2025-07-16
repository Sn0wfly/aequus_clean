# poker_bot/core/trainer.py

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

# ---------- Config ----------
@dataclass
class TrainerConfig:
    batch_size: int = 128
    num_actions: int = 3
    max_info_sets: int = 50_000

# ---------- JAX-Native CFR Step (ESTRUCTURA FINAL) ----------
@partial(jax.jit, static_argnums=(2,))
def _jitted_train_step(regrets: Array, strategy: Array, config: TrainerConfig, key: Array):
    """
    Un paso de entrenamiento completo, compilado con JIT.
    Esta versión es un placeholder de la lógica de CFR, pero tiene la
    arquitectura correcta para ser rápida.
    """
    # 1. Simular el batch con el motor JIT.
    keys = jax.random.split(key, config.batch_size)
    
    # Asumimos que batch_play ahora puede devolver información de estados si la necesitáramos.
    payoffs, histories = fge.batch_play(keys)

    # 2. LÓGICA DE ACTUALIZACIÓN DE REGRETS (VECTORIZADA)
    # Esta es una versión simplificada que reemplaza el bucle de Python.
    # Una implementación real de CFR usaría lax.scan sobre `histories` para
    # calcular los regrets de forma precisa.
    # Por ahora, para tener un pipeline rápido y funcional, usaremos una
    # actualización de ejemplo.

    # Generamos índices de ejemplo para actualizar.
    num_updates = config.batch_size * 5  # Un número de ejemplo de estados visitados.
    info_set_indices = jax.random.randint(key, (num_updates,), 0, config.max_info_sets)
    
    # Generamos deltas de regrets aleatorios para simular el cálculo de CFR.
    # La forma del delta debe coincidir con la de los regrets para esos índices.
    # Tomamos el payoff promedio como base para el regret.
    avg_payoff = jnp.mean(payoffs, axis=0) # Shape (6,)
    # Usamos un delta basado en el payoff del primer jugador como ejemplo.
    regret_delta_base = jax.random.normal(key, (num_updates, config.num_actions)) * avg_payoff[0]
    
    new_regrets = regrets.at[info_set_indices].add(regret_delta_base)

    # 3. Actualizar la estrategia basada en los nuevos regrets (Regret Matching).
    positive_regrets = jnp.maximum(new_regrets[info_set_indices], 0.0)
    sum_pos_regrets = jnp.sum(positive_regrets, axis=1, keepdims=True)
    sum_pos_regrets = jnp.where(sum_pos_regrets > 0, sum_pos_regrets, 1.0)
    
    new_strategy_for_indices = positive_regrets / sum_pos_regrets
    new_strategy = strategy.at[info_set_indices].set(new_strategy_for_indices)

    return new_regrets, new_strategy

# ---------- Trainer (Refactorizado) ----------
class PokerTrainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.iteration = 0
        self.regrets  = jnp.zeros((config.max_info_sets, config.num_actions), dtype=jnp.float32)
        self.strategy = jnp.ones ((config.max_info_sets, config.num_actions), dtype=jnp.float32) / config.num_actions
        logger.info("PokerTrainer inicializado con arquitectura JAX-nativa final.")

    def train(self, num_iterations: int, save_path: str, save_interval: int):
        logger.info(f"🚀 Iniciando entrenamiento JAX-nativo por {num_iterations} iteraciones...")
        key = jax.random.PRNGKey(0)

        for i in range(1, num_iterations + 1):
            self.iteration += 1
            iter_key = jax.random.fold_in(key, self.iteration)
            
            # Ejecutar el paso de entrenamiento compilado
            self.regrets, self.strategy = _jitted_train_step(self.regrets, self.strategy, self.config, iter_key)
            
            # Sincronizar para obtener una medición de tiempo precisa en cada iteración.
            self.regrets.block_until_ready()
            
            logger.info(f"Iteración {self.iteration} completada.")

            if self.iteration % save_interval == 0:
                self.save_model(f"{save_path}_iter_{self.iteration}.pkl")
                
        logger.info("🎉 Entrenamiento finalizado.")

    def save_model(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'regrets':   np.asarray(self.regrets), # Copiar a CPU para guardar
                'strategy':  np.asarray(self.strategy),
                'iteration': self.iteration,
                'config':    self.config
            }, f)
        logger.info(f"💾 Modelo guardado en: {path}")

    def load_model(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.regrets   = jnp.array(data['regrets']) # Cargar a GPU
        self.strategy  = jnp.array(data['strategy'])
        self.iteration = data['iteration']
        logger.info(f"📂 Modelo cargado desde: {path}")