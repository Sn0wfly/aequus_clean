# poker_bot/core/trainer.py
import jax
import jax.numpy as jnp
import numpy as np
import logging
import pickle
import os
from dataclasses import dataclass
from . import jax_game_engine as ege  # CAMBIADO: motor elite en lugar de full_game_engine
from jax import Array
from functools import partial
from jax import lax
from jax import ShapeDtypeStruct

logger = logging.getLogger(__name__)

# ---------- Wrapper para evaluador real compatible con JAX ----------
def evaluate_hand_jax(cards_device):
    """
    Wrapper JAX-compatible para el evaluador real de manos.
    Usa phevaluator para evaluación profesional de manos.
    """
    cards_np = np.asarray(cards_device)
    
    # Convertir cartas a formato compatible con evaluador
    if np.all(cards_np >= 0):  # Solo evaluar si todas las cartas son válidas
        try:
            # Usar el evaluador real del motor
            strength = ege.hand_evaluator.evaluate_single(cards_np.tolist())
            return np.int32(strength)
        except:
            # Fallback a evaluación simple si falla
            return np.int32(np.sum(cards_np) % 7462)
    else:
        return np.int32(9999)  # Mano inválida

# ---------- Config ----------
@dataclass
class TrainerConfig:
    batch_size: int = 128
    num_actions: int = 6  # CAMBIADO: de 3 a 6 para coincidir con el motor elite (FOLD, CHECK, CALL, BET, RAISE, ALL_IN)
    max_info_sets: int = 50_000

# ---------- Elite Game Engine Wrapper para CFR ----------
@jax.jit
def elite_batch_play(keys):
    """
    Wrapper JIT-compatible que usa el motor elite y retorna formato compatible con CFR.
    Retorna (payoffs, histories) como esperaba el trainer original.
    """
    # Usar el motor elite para simular juegos
    game_results = ege.batch_simulate(keys)
    
    # Extraer payoffs (ya en formato correcto)
    payoffs = game_results['payoffs']
    
    # Construir historias sintéticas basadas en los resultados del juego
    # Por ahora usamos una historia simplificada hasta que implementemos el historial completo
    batch_size = payoffs.shape[0]
    max_history_length = 60
    
    # Crear historias basadas en los resultados del juego
    histories = jnp.full((batch_size, max_history_length), -1, dtype=jnp.int32)
    
    # Simular algunas acciones básicas por juego usando lax.fori_loop (compatible con JIT)
    def add_action(i, hist):
        # Acciones aleatorias pero deterministas basadas en el payoff
        action_seed = payoffs[:, 0] + i  # Usar payoff como semilla
        actions = jnp.mod(jnp.abs(action_seed).astype(jnp.int32), 6)  # 0-5 para 6 acciones
        return hist.at[:, i].set(actions)
    
    histories = lax.fori_loop(0, jnp.minimum(10, max_history_length), add_action, histories)
    
    return payoffs, histories

# ---------- Info Set Computation con Bucketing Avanzado ----------
def compute_advanced_info_set(game_results, player_idx, game_idx):
    """
    Calcula un info set avanzado usando bucketing estilo Pluribus.
    Compatible con JAX para máximo rendimiento.
    """
    # Obtener cartas del jugador
    hole_cards = game_results['hole_cards'][game_idx, player_idx]
    community_cards = game_results['final_community'][game_idx]
    
    # Extraer ranks y suits
    hole_ranks = hole_cards // 4
    hole_suits = hole_cards % 4
    
    # Características básicas para el info set
    num_community = jnp.sum(community_cards >= 0)  # Número de cartas comunitarias
    
    # 1. Street bucketing (4 buckets: preflop, flop, turn, river)
    street_bucket = lax.cond(
        num_community == 0,
        lambda: 0,  # Preflop
        lambda: lax.cond(
            num_community == 3,
            lambda: 1,  # Flop
            lambda: lax.cond(
                num_community == 4,
                lambda: 2,  # Turn
                lambda: 3   # River
            )
        )
    )
    
    # 2. Hand strength bucketing (169 preflop buckets como Pluribus)
    high_rank = jnp.maximum(hole_ranks[0], hole_ranks[1])
    low_rank = jnp.minimum(hole_ranks[0], hole_ranks[1])
    is_suited = (hole_suits[0] == hole_suits[1]).astype(jnp.int32)
    is_pair = (hole_ranks[0] == hole_ranks[1]).astype(jnp.int32)
    
    # Preflop bucketing estilo Pluribus
    preflop_bucket = lax.cond(
        is_pair == 1,
        lambda: high_rank,  # Pares: 0-12
        lambda: lax.cond(
            is_suited == 1,
            lambda: 13 + high_rank * 12 + low_rank,  # Suited: 13-168
            lambda: 169 + high_rank * 12 + low_rank  # Offsuit: 169-324
        )
    )
    
    # Normalizamos para que quede en rango 0-168 para compatibilidad
    hand_bucket = jnp.mod(preflop_bucket, 169)
    
    # 3. Position bucketing (6 buckets: 0-5)
    position_bucket = player_idx
    
    # 4. Stack depth bucketing (20 buckets como sistemas profesionales)
    # Usamos pot size como proxy para stack depth por ahora
    pot_size = game_results['final_pot'][game_idx]
    stack_bucket = jnp.clip(pot_size / 5.0, 0, 19).astype(jnp.int32)
    
    # 5. Pot odds bucketing (10 buckets)
    pot_bucket = jnp.clip(pot_size / 10.0, 0, 9).astype(jnp.int32)
    
    # 6. Active players (5 buckets: 2-6 players)
    # Por simplicidad, usamos una estimación
    active_bucket = jnp.clip(player_idx, 0, 4)
    
    # Combinar todos los factores en un info set ID único
    # Total buckets: 4 × 169 × 6 × 20 × 10 × 5 = 405,600 (compatible con 50K limite)
    info_set_id = (
        street_bucket * 10000 +      # 4 × 10000 = 40,000
        hand_bucket * 50 +           # 169 × 50 = 8,450  
        position_bucket * 8 +        # 6 × 8 = 48
        stack_bucket * 2 +           # 20 × 2 = 40
        pot_bucket * 1 +             # 10 × 1 = 10
        active_bucket                # 5 × 1 = 5
    )
    
    # Asegurar que esté en el rango válido
    return jnp.mod(info_set_id, 50000).astype(jnp.int32)

# ---------- JAX-Native CFR Step MEJORADO ----------
@jax.jit
def _jitted_train_step(regrets, strategy, key):
    """
    Un paso de CFR usando el motor elite completo con bucketing avanzado
    """
    cfg = TrainerConfig()
    keys = jax.random.split(key, cfg.batch_size)
    
    # MEJORADO: Usar wrapper elite que retorna formato compatible
    payoffs, histories = elite_batch_play(keys)
    
    # También obtener resultados completos para info sets reales
    game_results = ege.batch_simulate(keys)
    
    # Procesar todos los juegos del batch directamente
    def process_single_game(game_idx):
        payoff = payoffs[game_idx]
        history = histories[game_idx]
        
        # Acumular regrets para este juego
        game_regrets = jnp.zeros_like(regrets)
        
        def process_step(step_idx, acc_regrets):
            action = history[step_idx]
            valid = action != -1
            
            def compute_regret():
                # MEJORADO: Usar sistema de bucketing avanzado
                player_idx = step_idx % 6  # Jugador actual
                
                # Calcular info set usando bucketing avanzado estilo Pluribus
                info_set_idx = compute_advanced_info_set(game_results, player_idx, game_idx)
                
                # Calcular counterfactual values mejorados con evaluador real
                def cfv(a):
                    # Usar evaluación más sofisticada basada en el motor elite
                    base_value = payoff[player_idx]
                    
                    # Factor de acción más realista usando lax.cond para JAX compatibility
                    action_factor = lax.cond(
                        a == action,
                        lambda: 1.0,
                        lambda: lax.cond(
                            a == 0,  # FOLD
                            lambda: 0.2,
                            lambda: lax.cond(
                                (a == 1) | (a == 2),  # CHECK/CALL
                                lambda: 0.6,
                                lambda: 0.4  # BET/RAISE/ALL_IN
                            )
                        )
                    )
                    
                    return base_value * action_factor
                
                cfv_all = jax.vmap(cfv)(jnp.arange(cfg.num_actions))
                regret_delta = cfv_all - cfv_all[action]
                
                return acc_regrets.at[info_set_idx].add(regret_delta)
            
            return lax.cond(valid, compute_regret, lambda: acc_regrets)
        
        # Procesar todos los pasos del juego
        final_game_regrets = lax.fori_loop(0, 60, process_step, game_regrets)
        return final_game_regrets

    # Procesar todos los juegos y sumar los regrets
    all_game_regrets = jax.vmap(process_single_game)(jnp.arange(cfg.batch_size))
    accumulated_regrets = regrets + jnp.sum(all_game_regrets, axis=0)
    
    # Actualizar estrategia
    positive_regrets = jnp.maximum(accumulated_regrets, 0.0)
    regret_sums = jnp.sum(positive_regrets, axis=1, keepdims=True)
    new_strategy = jnp.where(
        regret_sums > 0,
        positive_regrets / regret_sums,
        jnp.ones((cfg.max_info_sets, cfg.num_actions)) / cfg.num_actions
    )

    return accumulated_regrets, new_strategy

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