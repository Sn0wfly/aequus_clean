# poker_bot/core/trainer.py
import jax
import jax.numpy as jnp
import numpy as np
import logging
import pickle
import os
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
    Un paso de CFR que mantiene las dimensiones correctas
    """
    cfg = TrainerConfig()
    keys = jax.random.split(key, cfg.batch_size)
    payoffs, histories = fge.batch_play(keys)

    def game_step(carry, step_idx):
        regrets, strategy = carry
        
        def one_game(batch_idx):
            payoff = payoffs[batch_idx]
            history = histories[batch_idx]
            action = history[step_idx]
            valid = action != -1
            
            def do_update():
                state = fge.initial_state_for_idx(batch_idx)
                state = lax.fori_loop(
                    0, step_idx + 1,
                    lambda i, s: lax.cond(
                        history[i] != -1,
                        lambda st: fge.step(st, history[i]),
                        lambda st: st,
                        s
                    ),
                    state
                )
                
                player_idx = state.cur_player[0]
                legal = fge.get_legal_actions(state)
                
                def cfv(a):
                    return lax.cond(legal[a], lambda: payoff[player_idx], lambda: 0.0)
                cfv_all = jax.vmap(cfv)(jnp.arange(cfg.num_actions))
                regret_delta = cfv_all - cfv_all[action]
                
                info_set_idx = jnp.mod(player_idx, cfg.max_info_sets).astype(jnp.int32)
                
                return info_set_idx, regret_delta
            
            info_idx, delta = lax.cond(
                valid, 
                do_update, 
                lambda: (0, jnp.zeros(cfg.num_actions))
            )
            
            masked_delta = jnp.where(valid, delta, 0.0)
            
            return info_idx, masked_delta

        info_indices, deltas = jax.vmap(one_game)(jnp.arange(cfg.batch_size))
        
        def accumulate_deltas(i, acc_regrets):
            idx = info_indices[i]
            delta = deltas[i]
            return acc_regrets.at[idx].add(delta)
        
        new_regrets = lax.fori_loop(
            0, cfg.batch_size,
            accumulate_deltas,
            regrets
        )
        
        positive_regrets = jnp.maximum(new_regrets, 0.0)
        regret_sums = jnp.sum(positive_regrets, axis=1, keepdims=True)
        new_strategy = jnp.where(
            regret_sums > 0,
            positive_regrets / regret_sums,
            jnp.ones((cfg.max_info_sets, cfg.num_actions)) / cfg.num_actions
        )
        
        return (new_regrets, new_strategy), None

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
        
        logger.info("=" * 60)
        logger.info("🎯 PokerTrainer CFR-JIT inicializado")
        logger.info("=" * 60)
        logger.info(f"📊 Configuración:")
        logger.info(f"   - Batch size: {config.batch_size}")
        logger.info(f"   - Num actions: {config.num_actions}")
        logger.info(f"   - Max info sets: {config.max_info_sets:,}")
        logger.info(f"   - Shape regrets: {self.regrets.shape}")
        logger.info(f"   - Shape strategy: {self.strategy.shape}")
        logger.info("=" * 60)

    def train(self, num_iterations: int, save_path: str, save_interval: int):
        key = jax.random.PRNGKey(42)  # Semilla fija para reproducibilidad
        
        logger.info("\n🚀 INICIANDO ENTRENAMIENTO CFR")
        logger.info(f"   Total iteraciones: {num_iterations}")
        logger.info(f"   Guardar cada: {save_interval} iteraciones")
        logger.info(f"   Path base: {save_path}")
        logger.info("\n⏳ Compilando función JIT (primera iteración será más lenta)...\n")
        
        import time
        start_time = time.time()
        
        for i in range(1, num_iterations + 1):
            self.iteration += 1
            iter_key = jax.random.fold_in(key, self.iteration)
            
            iter_start = time.time()
            
            try:
                # Un paso de entrenamiento
                self.regrets, self.strategy = _jitted_train_step(
                    self.regrets,
                    self.strategy,
                    iter_key
                )
                
                # Esperamos a que termine la computación
                self.regrets.block_until_ready()
                
                iter_time = time.time() - iter_start
                
                # Log simple cada iteración
                logger.info(f"✓ Iteración {self.iteration} completada ({iter_time:.2f}s)")
                
                # Métricas detalladas periódicamente
                if self.iteration % max(1, num_iterations // 10) == 0:
                    self._log_detailed_metrics(num_iterations, start_time)
                
            except Exception as e:
                logger.error(f"\n❌ ERROR en iteración {self.iteration}")
                logger.error(f"   Tipo: {type(e).__name__}")
                logger.error(f"   Mensaje: {str(e)}")
                logger.error(f"   Shapes - regrets: {self.regrets.shape}, strategy: {self.strategy.shape}")
                
                import traceback
                logger.error("\nTraceback completo:")
                logger.error(traceback.format_exc())
                
                raise
                
            # Guardamos checkpoints
            if self.iteration % save_interval == 0:
                checkpoint_path = f"{save_path}_iter_{self.iteration}.pkl"
                self.save_model(checkpoint_path)
        
        # Resumen final
        total_time = time.time() - start_time
        
        # Guardamos el modelo final
        final_path = f"{save_path}_final.pkl"
        self.save_model(final_path)
        
        logger.info("\n" + "="*60)
        logger.info("🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE! 🎉")
        logger.info("="*60)
        logger.info(f"⏱️  Tiempo total: {total_time:.1f}s ({total_time/60:.1f} min)")
        logger.info(f"📊 Iteraciones completadas: {self.iteration}")
        logger.info(f"⚡ Velocidad promedio: {self.iteration/total_time:.1f} iter/s")
        logger.info(f"💾 Modelo final guardado: {final_path}")
        logger.info("="*60 + "\n")

    def _log_detailed_metrics(self, total_iterations, start_time):
        """Log métricas detalladas del entrenamiento"""
        elapsed = time.time() - start_time
        
        # Métricas de regret
        avg_regret = float(jnp.mean(jnp.abs(self.regrets)))
        max_regret = float(jnp.max(jnp.abs(self.regrets)))
        min_regret = float(jnp.min(self.regrets))
        non_zero_regrets = int(jnp.sum(jnp.any(self.regrets != 0, axis=1)))
        
        # Métricas de estrategia
        eps = 1e-8
        strategy_entropy = -float(jnp.mean(
            jnp.sum(self.strategy * jnp.log(self.strategy + eps), axis=1)
        ))
        max_action_prob = float(jnp.max(self.strategy))
        min_action_prob = float(jnp.min(self.strategy))
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 REPORTE DE PROGRESO - Iteración {self.iteration}/{total_iterations}")
        logger.info(f"{'='*60}")
        logger.info(f"⏱️  Tiempo transcurrido: {elapsed:.1f}s")
        logger.info(f"⚡ Velocidad: {self.iteration/elapsed:.1f} iter/s")
        logger.info(f"⏳ ETA: {(total_iterations-self.iteration)/(self.iteration/elapsed):.1f}s")
        logger.info(f"\n📈 MÉTRICAS DE REGRET:")
        logger.info(f"   - Promedio: {avg_regret:.6f}")
        logger.info(f"   - Máximo: {max_regret:.6f}")
        logger.info(f"   - Mínimo: {min_regret:.6f}")
        logger.info(f"   - Info sets activos: {non_zero_regrets:,}/{self.config.max_info_sets:,} ({100*non_zero_regrets/self.config.max_info_sets:.1f}%)")
        logger.info(f"\n🎲 MÉTRICAS DE ESTRATEGIA:")
        logger.info(f"   - Entropía: {strategy_entropy:.4f}")
        logger.info(f"   - Prob máxima: {max_action_prob:.4f}")
        logger.info(f"   - Prob mínima: {min_action_prob:.6f}")
        logger.info(f"{'='*60}\n")

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
        
        size_mb = os.path.getsize(path) / 1024 / 1024
        logger.info(f"💾 Checkpoint guardado: {path} ({size_mb:.1f} MB)")

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

# Importamos time si no está importado
import time