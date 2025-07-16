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
from jax import lax

logger = logging.getLogger(__name__)

# ---------- Config ----------
@dataclass
class TrainerConfig:
    batch_size: int = 128
    num_actions: int = 3  # Cambiado de 14 a 3 para coincidir con el motor
    max_info_sets: int = 50_000

# ---------- JAX-Native CFR Step ----------
@jax.jit
def _jitted_train_step(regrets, strategy, key):
    """
    Un paso de CFR completamente vectorizado en JAX
    """
    cfg = TrainerConfig()
    keys = jax.random.split(key, cfg.batch_size)
    payoffs, histories = fge.batch_play(keys)

    def game_step(carry, step_idx):
        regrets, strategy = carry
        
        # Procesamos cada juego del batch para este paso
        def process_single_game(batch_idx):
            payoff = payoffs[batch_idx]
            history = histories[batch_idx]
            action = history[step_idx]
            valid = action != -1
            
            # Si no es válido, retornamos valores neutros
            def compute_deltas():
                # Reconstruimos el estado hasta este punto
                state = fge.initial_state_for_idx(batch_idx)
                
                # Aplicamos las acciones hasta el paso actual
                def apply_action_at_step(i, s):
                    return lax.cond(
                        history[i] != -1,
                        lambda: fge.step(s, history[i]),
                        lambda: s
                    )
                
                state = lax.fori_loop(0, step_idx + 1, apply_action_at_step, state)
                
                # Información del estado
                player_idx = state.cur_player[0]
                legal = fge.get_legal_actions(state)
                
                # Valores contrafactuales
                cfv_values = jnp.where(legal, payoff[player_idx], 0.0)
                cfv_taken = cfv_values[action]
                
                # Delta de regret para esta acción
                regret_delta = cfv_values - cfv_taken
                
                # Índice del information set
                info_set_idx = jnp.mod(player_idx, cfg.max_info_sets)
                
                return info_set_idx, regret_delta
            
            # Computamos los deltas solo si la acción es válida
            info_idx, reg_delta = lax.cond(
                valid,
                compute_deltas,
                lambda: (0, jnp.zeros(cfg.num_actions))
            )
            
            # Retornamos el índice, delta y máscara de validez
            return info_idx, reg_delta, valid
        
        # Procesamos todo el batch
        indices, deltas, valids = jax.vmap(process_single_game)(jnp.arange(cfg.batch_size))
        
        # Ahora necesitamos acumular los deltas en los regrets
        # Usamos un scatter-add pattern
        # Para cada índice único, sumamos todos los deltas correspondientes
        
        # Creamos una función para acumular un solo info set
        def accumulate_for_info_set(info_idx):
            # Máscara para este info set
            mask = (indices == info_idx) & valids
            # Sumamos todos los deltas para este info set
            accumulated_delta = jnp.sum(
                jnp.where(mask[:, None], deltas, 0.0),
                axis=0
            )
            return accumulated_delta
        
        # Aplicamos la acumulación para todos los info sets
        all_info_indices = jnp.arange(cfg.max_info_sets)
        accumulated_deltas = jax.vmap(accumulate_for_info_set)(all_info_indices)
        
        # Actualizamos los regrets
        new_regrets = regrets + accumulated_deltas
        
        # Actualizamos la estrategia con regret matching
        positive_regrets = jnp.maximum(new_regrets, 0.0)
        regret_sums = jnp.sum(positive_regrets, axis=1, keepdims=True)
        
        # Nueva estrategia
        new_strategy = jnp.where(
            regret_sums > 0,
            positive_regrets / regret_sums,
            jnp.ones((cfg.max_info_sets, cfg.num_actions)) / cfg.num_actions
        )
        
        # Retornamos el carry actualizado (manteniendo las formas)
        return (new_regrets, new_strategy), None

    # Ejecutamos el scan sobre todos los pasos del juego
    (final_regrets, final_strategy), _ = lax.scan(
        game_step,
        (regrets, strategy),
        jnp.arange(fge.MAX_GAME_LENGTH)
    )

    return final_regrets, final_strategy

# ---------- Trainer ----------
class PokerTrainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.iteration = 0
        self.regrets  = jnp.zeros((config.max_info_sets, config.num_actions), dtype=jnp.float32)
        self.strategy = jnp.ones((config.max_info_sets, config.num_actions), dtype=jnp.float32) / config.num_actions
        logger.info("PokerTrainer CFR-JIT listo.")
        logger.info(f"Configuración: batch_size={config.batch_size}, "
                   f"num_actions={config.num_actions}, "
                   f"max_info_sets={config.max_info_sets}")

    def train(self, num_iterations: int, save_path: str, save_interval: int):
        key = jax.random.PRNGKey(0)
        
        logger.info(f"🚀 Iniciando entrenamiento CFR")
        logger.info(f"   Iteraciones: {num_iterations}")
        logger.info(f"   Guardar cada: {save_interval} iteraciones")
        
        for i in range(1, num_iterations + 1):
            self.iteration += 1
            iter_key = jax.random.fold_in(key, self.iteration)
            
            try:
                # Un paso de entrenamiento
                self.regrets, self.strategy = _jitted_train_step(
                    self.regrets,
                    self.strategy,
                    iter_key
                )
                
                # Aseguramos que la computación termine
                self.regrets.block_until_ready()
                
                # Métricas de progreso
                if self.iteration % max(1, num_iterations // 10) == 0:
                    # Calculamos algunas estadísticas
                    avg_regret = float(jnp.mean(jnp.abs(self.regrets)))
                    max_regret = float(jnp.max(jnp.abs(self.regrets)))
                    non_zero_regrets = int(jnp.sum(jnp.any(self.regrets != 0, axis=1)))
                    
                    logger.info(f"📊 Iteración {self.iteration}/{num_iterations}")
                    logger.info(f"   Regret promedio: {avg_regret:.4f}")
                    logger.info(f"   Regret máximo: {max_regret:.4f}")
                    logger.info(f"   Info sets activos: {non_zero_regrets}/{self.config.max_info_sets}")
                
            except Exception as e:
                logger.error(f"❌ Error en iteración {self.iteration}: {str(e)}")
                logger.error(f"   Shapes - regrets: {self.regrets.shape}, strategy: {self.strategy.shape}")
                raise
                
            # Guardamos checkpoints
            if self.iteration % save_interval == 0:
                checkpoint_path = f"{save_path}_iter_{self.iteration}.pkl"
                self.save_model(checkpoint_path)
        
        # Guardamos el modelo final
        final_path = f"{save_path}_final.pkl"
        self.save_model(final_path)
        
        logger.info("🎉 Entrenamiento completado exitosamente!")
        logger.info(f"   Modelo final guardado en: {final_path}")

    def save_model(self, path: str):
        """Guarda el modelo actual a disco"""
        model_data = {
            'regrets':   np.asarray(self.regrets),
            'strategy':  np.asarray(self.strategy),
            'iteration': self.iteration,
            'config':    self.config
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"💾 Checkpoint guardado: {path}")

    def load_model(self, path: str):
        """Carga un modelo desde disco"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.regrets   = jnp.array(data['regrets'])
        self.strategy  = jnp.array(data['strategy'])
        self.iteration = data['iteration']
        
        if 'config' in data:
            self.config = data['config']
        
        logger.info(f"📂 Modelo cargado: {path}")
        logger.info(f"   Iteración: {self.iteration}")
        logger.info(f"   Shape regrets: {self.regrets.shape}")
        logger.info(f"   Shape strategy: {self.strategy.shape}")