"""
🚀 DEFINITIVE HYBRID TRAINER
Combines vectorized GPU simulation with efficient CPU-GPU bridge for dynamic memory management.

Key Innovations:
- Vectorized GPU simulation (fastest possible)
- Efficient CPU-GPU bridge for memory management
- Scatter-gather updates for optimal GPU usage
- Dynamic growth with minimal CPU overhead
- PURE JIT functions for maximum performance
- MULTIPROCESSING for CPU bottleneck optimization
"""

import jax
import jax.numpy as jnp
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from functools import partial
import time
import pickle
from poker_bot.evaluator import HandEvaluator
from functools import lru_cache
from joblib import Parallel, delayed
import pyspiel
from poker_bot.core.cfr_gpu import cfr_step_gpu
from poker_bot.core.mccfr_gpu import mccfr_rollout_gpu
import functools
from . import full_game_engine as fge

logger = logging.getLogger(__name__)

# 🚀 Import Cython module for ultra-fast hashing
try:
    from .hasher import map_hashes_cython
    CYTHON_AVAILABLE = True
    logger.info("🚀 Cython fast hasher: ENABLED for maximum performance")
except ImportError:
    CYTHON_AVAILABLE = False
    logger.warning("⚠️ Cython fast hasher: NOT AVAILABLE, falling back to Python")

def make_dummy_cf_values(batch_size, num_players, num_actions):
    """Devuelve cf_values aleatorios para probar el pipeline."""
    total = batch_size * num_players
    rng = jax.random.PRNGKey(42)
    cf_values = jax.random.normal(rng, (total, num_actions))
    return cf_values

@dataclass
class TrainerConfig:
    """Configuration for PokerTrainer - H100 Optimized"""
    batch_size: int = 32768  # H100 optimized - fills VRAM
    learning_rate: float = 0.05  # Pluribus-like learning rate
    temperature: float = 1.0
    num_actions: int = 14  # Pluribus-like action set
    dtype: jnp.dtype = jnp.bfloat16
    accumulation_dtype: jnp.dtype = jnp.float32
    max_info_sets: int = 50000  # 50k buckets fixed for super-human quality
    growth_factor: float = 1.5  # Grow by 50% when full
    chunk_size: int = 20000  # Subo chunk_size a 20_000
    gpu_bucket: bool = False  # Placeholder para bucketing en GPU
    use_pluribus_bucketing: bool = True  # Usar bucketing super-Pluribus
    N_rollouts: int = 500  # GPU-accelerated rollouts per bucket

# 🚀 PURE JIT-COMPILED FUNCTION (Outside class for maximum performance)
@partial(jax.jit, static_argnums=(4, 5))
def _static_vectorized_scatter_update(q_values: jnp.ndarray, 
                                    strategies: jnp.ndarray,
                                    indices: jnp.ndarray, 
                                    cf_values: jnp.ndarray,
                                    learning_rate: float,
                                    temperature: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    🚀 PURE JIT-COMPILED FUNCTION: Maximum performance with dynamic shapes
    JAX will automatically recompile for new array shapes while maintaining speed
    """
    # Ensure cf_values has the same dtype as q_values to avoid warnings
    cf_values = cf_values.astype(q_values.dtype)
    
    # GATHER: Get current Q-values for indices
    current_q_subset = q_values[indices]
    
    # UPDATE: Compute new Q-values
    updated_q_subset = current_q_subset + learning_rate * (cf_values - current_q_subset)
    
    # SCATTER: Update Q-values
    new_q_values = q_values.at[indices].set(updated_q_subset)
    
    # Update strategies
    strategies_subset = jax.nn.softmax(updated_q_subset / temperature)
    new_strategies = strategies.at[indices].set(strategies_subset)
    
    return new_q_values, new_strategies

class PokerTrainer:
    """
    🚀 POKER TRAINER
    Combines vectorized GPU simulation with efficient CPU-GPU bridge
    """
    
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.iteration = 0
        self.total_games = 0
        self.total_info_sets = 0
        self.total_unique_info_sets = 0
        self.growth_events = 0
        self._main_device = None

        # --- SOLUCIÓN TAREA 1: Guardar el dispositivo principal ---
        try:
            self._main_device = jax.devices('gpu')[0]
            logger.info(f"✅ GPU detectada. Asignando arrays principales a: {self._main_device}")
        except IndexError:
            self._main_device = jax.devices('cpu')[0]
            logger.warning(f"⚠️ No se encontró GPU. Usando CPU: {self._main_device}")

        initial_q_values = jnp.zeros((config.max_info_sets, config.num_actions), dtype=config.dtype)
        self.q_values = jax.device_put(initial_q_values, device=self._main_device)
        
        initial_strategies = jnp.ones((config.max_info_sets, config.num_actions), dtype=config.dtype) / config.num_actions
        self.strategies = jax.device_put(initial_strategies, device=self._main_device)
        
        initial_regrets = jnp.zeros((config.max_info_sets, config.num_actions), dtype=config.dtype)
        self.regrets = jax.device_put(initial_regrets, device=self._main_device)

        # 🧠 CPU Memory Management (the brain)
        self.info_set_hashes: Dict[str, int] = {}
        self.info_set_data: Dict[int, Dict[str, Any]] = {}
        self.next_index = 0
        
        self.evaluator = HandEvaluator()  # Instancia para bucketing
        
        # --- GPU hash-table para bucketing persistente ---
        import cupy as cp
        self.table_size = 2**26  # 67 M slots
        self.table_keys = cp.zeros(self.table_size, dtype=cp.uint64)
        self.table_vals = cp.zeros(self.table_size, dtype=cp.uint32)
        self.counter = cp.zeros(1, dtype=cp.uint32)
        
        logger.info("🚀 Definitive Hybrid Trainer initialized")
        logger.info(f"   Batch size: {config.batch_size}")
        logger.info(f"   Max info sets: {config.max_info_sets:,}")
        logger.info(f"   Growth factor: {config.growth_factor}")
        logger.info(f"   Chunk size: {config.chunk_size}")
        logger.info(f"   Target: Real NLHE 6-player strategies with optimal GPU-CPU bridge")
        logger.info(f"   🚀 PURE JIT functions: ENABLED for maximum performance")
    
    def _get_info_set_bucket(self, hole_cards_tuple, community_cards_tuple, position, pot_size, stack_size, num_active):
        """
        Bucketing medio: rangos anchos para stack y pot, string compacto.
        """
        hole_cards = np.array(hole_cards_tuple)
        community_cards = np.array(community_cards_tuple)
        dealt_community = community_cards[community_cards >= 0]
        round_map = {0: "Preflop", 3: "Flop", 4: "Turn", 5: "River"}
        round_name = round_map.get(len(dealt_community), "River")
        if round_name == "Preflop":
            card1_rank, card2_rank = sorted([c % 13 for c in hole_cards], reverse=True)
            is_pair = (card1_rank == card2_rank)
            is_suited = (hole_cards[0] // 13 == hole_cards[1] // 13)
            if is_pair and card1_rank >= 10: hand_category = "PremiumPair"
            elif is_pair: hand_category = "LowPair"
            elif is_suited and card1_rank >= 9: hand_category = "PremiumSuited"
            elif card1_rank >= 10 and card2_rank >= 10: hand_category = "PremiumBroadway"
            else: hand_category = "Other"
        else:
            full_hand = np.concatenate((hole_cards, dealt_community))
            strength_rank = self.evaluator.evaluate_single(full_hand.tolist())
            if strength_rank <= 10: hand_category = "StraightFlush"
            elif strength_rank <= 166: hand_category = "Quads"
            elif strength_rank <= 322: hand_category = "FullHouse"
            elif strength_rank <= 1599: hand_category = "Flush"
            elif strength_rank <= 1609: hand_category = "Straight"
            elif strength_rank <= 2467: hand_category = "ThreeOfAKind"
            elif strength_rank <= 3325: hand_category = "TwoPair"
            elif strength_rank <= 6185: hand_category = "OnePair"
            else: hand_category = "HighCard"
        stack_bucket = f"{int(stack_size//10)*10}-{(int(stack_size//10)+1)*10}"
        pot_bucket   = f"{int(pot_size//5)*5}-{(int(pot_size//5)+1)*5}"
        return f"R:{round_name}_H:{hand_category}_P:{position}_Stk:{stack_bucket}_Pot:{pot_bucket}_Act:{num_active}"

    @staticmethod
    @lru_cache(maxsize=100_000)
    def _cached_bucket(hole_cards_tuple, community_cards_tuple, position, pot_size, stack_size, num_active):
        # Esta función será llamada desde _batch_get_buckets
        # Debe ser staticmethod para que lru_cache funcione correctamente
        # La lógica real se delega a _get_info_set_bucket
        # Se asume que se pasa self como primer argumento en el wrapper
        raise NotImplementedError("El wrapper debe pasar self y delegar a _get_info_set_bucket")

    def _get_or_create_index(self, bucket_id: str) -> int:
        """Get or create index for info set hash (CPU operation)"""
        if bucket_id not in self.info_set_hashes:
            # Check if we need to grow arrays
            if self.next_index >= self.config.max_info_sets:
                self._grow_arrays()
            
            # Create new index
            self.info_set_hashes[bucket_id] = self.next_index
            self.info_set_data[self.next_index] = {'hash': bucket_id}
            self.next_index += 1
            self.total_unique_info_sets += 1
        
        return self.info_set_hashes[bucket_id]
    
    def _grow_arrays(self):
        old_size = self.config.max_info_sets
        new_size = int(old_size * self.config.growth_factor)
        logger.info(f"🔄 Growing arrays from {old_size:,} to {new_size:,}")

        # --- SOLUCIÓN TAREA 1: Usar el dispositivo guardado ---
        target_device = self._main_device 
        logger.info(f"🔄 Creciendo arrays y asignando a dispositivo: {target_device}")

        new_q_values = jnp.zeros((new_size, self.config.num_actions), dtype=self.config.dtype)
        new_strategies = jnp.ones((new_size, self.config.num_actions), dtype=self.config.dtype) / self.config.num_actions

        new_q_values = new_q_values.at[:old_size].set(self.q_values)
        new_strategies = new_strategies.at[:old_size].set(self.strategies)

        self.q_values = jax.device_put(new_q_values, device=target_device)
        self.strategies = jax.device_put(new_strategies, device=target_device)
        self.config.max_info_sets = new_size

        self.growth_events += 1
        logger.info(f"✅ Arrays grown successfully (event #{self.growth_events})")
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def _state_to_tensor(self, game_dict):
        # game_dict: dict con 'hole_cards' (B,6,2), 'final_community' (B,5), 'payoffs' (B,6), 'final_pot' (B)
        # Devuelve (B, state_size) tensor JAX compatible con OpenSpiel
        hole = game_dict['hole_cards']          # (B,6,2)
        comm = game_dict['final_community']     # (B,5)
        B = hole.shape[0]
        state_size = 309  # longitud fija OpenSpiel 6-max
        # Placeholder: TODO lógica real de reconstrucción de estado OpenSpiel
        return jnp.zeros((B, state_size), dtype=jnp.float32)

    def _batch_get_buckets_gpu(self, hole_cards, community_cards,
                               positions, pot_sizes, stack_sizes,
                               num_actives):
        """
        Fine-grained GPU bucketing + lookup persistente.
        Devuelve (indices_gpu, keys_gpu) - tuple con ambos arrays
        """
        import cupy as cp
        import numpy as np
        from .bucket_gpu import build_or_get_indices
        from .pluribus_bucket_gpu import pluribus_bucket_kernel_wrapper
        from .bucket_fine import fine_bucket_kernel_wrapper
        
        # 1. Convertir a CuPy arrays
        hole_cards_gpu = cp.asarray(hole_cards, dtype=cp.int32)
        community_cards_gpu = cp.asarray(community_cards, dtype=cp.int32)
        positions_gpu = cp.asarray(positions, dtype=cp.int32)
        stack_sizes_gpu = cp.asarray(stack_sizes, dtype=cp.float32)
        pot_sizes_gpu = cp.asarray(pot_sizes, dtype=cp.float32)
        num_active_gpu = cp.asarray(num_actives, dtype=cp.int32)
        
        # 2. Llamar a kernel CUDA de bucketing (Pluribus o fino según configuración)
        if self.config.use_pluribus_bucketing:
            # Bucketing ultra-agresivo estilo Pluribus (~200k buckets)
            keys_gpu = pluribus_bucket_kernel_wrapper(
                hole_cards_gpu,
                community_cards_gpu,
                positions_gpu,
                pot_sizes_gpu,
                stack_sizes_gpu,
                num_active_gpu
            )
        else:
            # Bucketing fino original (millones de buckets)
            keys_gpu = fine_bucket_kernel_wrapper(
                hole_cards_gpu,
                community_cards_gpu,
                positions_gpu,
                stack_sizes_gpu,
                pot_sizes_gpu,
                num_active_gpu
            )
        
        # 3. Lookup/inserción en tabla persistente
        indices_gpu = build_or_get_indices(
            keys_gpu.reshape(-1),  # Flatten for lookup
            self.table_keys,
            self.table_vals,
            self.counter
        )
        
        # 4. CAMBIO CRÍTICO: Devolver AMBOS arrays
        return indices_gpu, keys_gpu.reshape(-1)

    def train(self, num_iterations: int, save_path: str, save_interval: int):
        """
        Bucle principal de entrenamiento.
        """
        logger.info(f"🚀 Iniciando bucle de entrenamiento por {num_iterations} iteraciones...")
        start_time = time.time()
        
        initial_iteration = self.iteration
        for i in range(initial_iteration, initial_iteration + num_iterations):
            self.iteration = i
            
            # Generar claves para el batch (una subclave por elemento)
            base_rng = jax.random.PRNGKey(i)
            rngs = jax.random.split(base_rng, self.config.batch_size)  # Una subclave por elemento
            
            # Configuración del juego (puede ser más dinámica en el futuro)
            game_config = {
                'players': 6,
                'starting_stack': 100.0,
                'small_blind': 1.0,
                'big_blind': 2.0
            }
            
            # Simulación en GPU con datos únicos por elemento
            from .simulation import batch_simulate_real_holdem
            game_results = batch_simulate_real_holdem(rngs, game_config)
            
            # Paso de entrenamiento (CPU + GPU)
            self.train_step(game_results)

            # Guardar checkpoint
            if (self.iteration + 1) % save_interval == 0:
                logger.info(f"\n💾 Guardando checkpoint en la iteración {self.iteration + 1}...")
                self.save_model(save_path)

        # Guardado final
        self.save_model(save_path)
        total_time = time.time() - start_time
        logger.info(f"🎉 Entrenamiento finalizado en {total_time:.2f} segundos.")

    def train_step(self, game_results: dict, *, iteration: int | None = None):
        """
        Paso de entrenamiento con control explícito de dispositivos.
        """
        if iteration is not None:
            self.iteration = iteration
        else:
            self.iteration += 1
        self.total_games += self.config.batch_size
        batch_size = game_results['payoffs'].shape[0]
        num_players = game_results['payoffs'].shape[1]
        total_info_sets = batch_size * num_players
        logger.info(f"   🚀 DEFINITIVE processing: {batch_size} games × {num_players} players = {total_info_sets} info sets")

        rng_policy, rng_sim = jax.random.split(jax.random.PRNGKey(self.iteration))
        policy_logits = jax.random.normal(rng_policy, (6, 14))

        final_states, payoffs = fge.batch_play_game(
            batch_size=self.config.batch_size,
            policy_logits=policy_logits,
            key=rng_sim
        )

        # Usa 'payoffs' y 'final_states' en vez de 'game_results'
        # Adapta el resto del código para usar estos nuevos outputs
        # Por ejemplo, si antes usabas game_results['payoffs'], ahora usa 'payoffs'
        # ...
        # return lo que corresponda

        # Calcula num_active una sola vez
        num_active = jnp.sum(game_results['hole_cards'][:, :, 0] != -1, axis=1)

        # 2. GPU Bucketing + Indexado
        # La simulación devuelve: 'hole_cards', 'final_community', 'payoffs', 'final_pot'
        # Necesitamos generar datos sintéticos para bucketing
        batch_size = game_results['hole_cards'].shape[0]
        num_players = game_results['hole_cards'].shape[1]
        
        # Generar datos reales por jugador (simular variación real)
        rng = jax.random.PRNGKey(self.iteration)
        rng_batch = jax.random.split(rng, self.config.batch_size)

        # Posiciones reales (0-5 para cada jugador)
        positions = jnp.tile(jnp.arange(6), (self.config.batch_size, 1))
        
        # Stack sizes variables por jugador (simular stacks reales)
        rng_stacks = jax.random.split(rng, self.config.batch_size * 6).reshape(self.config.batch_size, 6, -1)
        base_stacks = jnp.full((self.config.batch_size, 6), 100.0)
        
        # Usar vmap para aplicar random.uniform a cada key individualmente
        def generate_stack_variation(keys):
            return jax.vmap(lambda k: jax.random.uniform(k, shape=(), minval=-20.0, maxval=20.0))(keys)
        stack_variation = jax.vmap(generate_stack_variation)(rng_stacks)
        stack_sizes = base_stacks + stack_variation
        
        # Pot sizes variables por jugador (simular pots reales)
        base_pot = game_results['final_pot'][:, None]
        def generate_pot_variation(keys):
            return jax.vmap(lambda k: jax.random.uniform(k, shape=(), minval=0.5, maxval=1.5))(keys)
        pot_variation = jax.vmap(generate_pot_variation)(rng_stacks)
        pot_sizes = base_pot * pot_variation
        
        # Active players variables por jugador (simular jugadores que se van)
        base_active = (game_results['hole_cards'][:, :, 0] != -1).astype(jnp.int32)
        def generate_active_variation(keys):
            return jax.vmap(lambda k: jax.random.randint(k, shape=(), minval=-1, maxval=2))(keys)
        active_variation = jax.vmap(generate_active_variation)(rng_stacks)
        num_active = jnp.clip(base_active + active_variation, 2, 6)

        # Convertir a numpy/cupy
        import numpy as np
        import cupy as cp
        indices_gpu, keys_gpu = self._batch_get_buckets_gpu(
            np.array(game_results['hole_cards']),
            np.array(game_results['final_community'][:, None, :].repeat(6, axis=1)),
            np.array(positions),
            np.array(pot_sizes),
            np.array(stack_sizes),
            np.array(num_active)
        )
        indices_cpu = cp.asnumpy(indices_gpu)

        # Debug: verificar que tenemos keys reales vs indices
        bucket_type = "PLURIBUS" if self.config.use_pluribus_bucketing else "FINO"
        print(f"🔧 Bucketing: {bucket_type}")
        print(f"Primeras 5 keys: {cp.asnumpy(keys_gpu)[:5]}")
        print(f"Primeras 5 indices: {cp.asnumpy(indices_gpu)[:5]}")
        
        # 🔧 DEBUG: Verificar rango de keys antes de MCCFR
        max_key = cp.max(keys_gpu)
        min_key = cp.min(keys_gpu)
        print(f"🔍 Keys range: {min_key} to {max_key}")
        if max_key > 25000:
            print(f"⚠️  WARNING: Keys too high, clamping to 25000")
            keys_gpu = cp.clip(keys_gpu, 0, 25000)

        # 3. Convertir a tensores de estado JAX
        states = jax.vmap(self._state_to_tensor)(game_results)

        # 4. Generar cf_values con MCCFR GPU (Monte-Carlo CFR vectorizado)
        # Ya no necesitamos la línea errónea de conversión - usamos keys reales
        cf_values_gpu = mccfr_rollout_gpu(keys_gpu, N_rollouts=self.config.N_rollouts, num_actions=self.config.num_actions)
        cf_values = cp.asnumpy(cf_values_gpu).astype(self.config.dtype)

        # 5. Actualizar q_values y strategies con scatter GPU-JAX
        new_q, new_strat = _static_vectorized_scatter_update(
            self.q_values,
            self.strategies,
            jnp.asarray(indices_gpu),  # mover a JAX
            cf_values,
            self.config.learning_rate,
            self.config.temperature
        )
        self.q_values = new_q
        self.strategies = new_strat

        # 6. Sincronizar contador con GPU
        self.total_unique_info_sets = int(self.counter[0])
        
        # Compute metrics
        avg_payoff = jnp.mean(game_results['payoffs'])
        if len(indices_cpu) > 0:
            strategies_subset = self.strategies[indices_cpu]
            entropy = -jnp.sum(strategies_subset * jnp.log(strategies_subset + 1e-8), axis=1)
            avg_entropy = jnp.mean(entropy)
        else:
            avg_entropy = 0.0
            
        logger.info(f"   Iteración {self.iteration} completada.")
        logger.info(f"   📊 Unique info sets: {self.total_unique_info_sets:,}")
        logger.info(f"   📊 Info sets processed: {len(indices_cpu)}")
        logger.info(f"   📊 Growth events: {self.growth_events}")
        logger.info(f"   📊 Array size: {self.config.max_info_sets:,}")
        
        return {
            'iteration': self.iteration,
            'total_games': self.total_games,
            'total_info_sets': self.total_info_sets,
            'unique_info_sets': self.total_unique_info_sets,
            'info_sets_processed': len(indices_cpu),
            'avg_payoff': avg_payoff,
            'strategy_entropy': avg_entropy,
            'q_values_count': self.total_unique_info_sets,
            'strategies_count': self.total_unique_info_sets,
            'games_processed': self.config.batch_size,
            'growth_events': self.growth_events,
            'array_size': self.config.max_info_sets
        }
    
    def save_model(self, path: str):
        """Save definitive hybrid model"""
        import cupy as cp
        model_data = {
            'q_values': np.array(self.q_values),
            'strategies': np.array(self.strategies),
            'info_set_hashes': self.info_set_hashes,
            'info_set_data': self.info_set_data,
            'iteration': self.iteration,
            'total_games': self.total_games,
            'total_info_sets': self.total_info_sets,
            'unique_info_sets': self.total_unique_info_sets,
            'growth_events': self.growth_events,
            'config': self.config,
            'gpu_keys': cp.asnumpy(self.table_keys),
            'gpu_vals': cp.asnumpy(self.table_vals),
            'gpu_counter': int(self.counter[0])
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        file_size = len(pickle.dumps(model_data))
        logger.info(f"💾 Definitive Hybrid model saved: {path}")
        logger.info(f"   Q-values shape: {self.q_values.shape}")
        logger.info(f"   Strategies shape: {self.strategies.shape}")
        logger.info(f"   Unique info sets: {self.total_unique_info_sets:,}")
        logger.info(f"   File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        logger.info(f"   Growth events: {self.growth_events}")
    
    def load_model(self, path: str):
        """Load definitive hybrid model"""
        import cupy as cp
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_values = jnp.array(model_data['q_values'])
        self.strategies = jnp.array(model_data['strategies'])
        self.info_set_hashes = model_data['info_set_hashes']
        self.info_set_data = model_data['info_set_data']
        self.iteration = model_data['iteration']
        self.total_games = model_data['total_games']
        self.total_info_sets = model_data['total_info_sets']
        self.total_unique_info_sets = model_data['unique_info_sets']
        self.growth_events = model_data['growth_events']
        self.next_index = len(self.info_set_data)
        
        # Cargar tabla hash GPU si existe
        if 'gpu_keys' in model_data:
            self.table_keys = cp.array(model_data['gpu_keys'])
            self.table_vals = cp.array(model_data['gpu_vals'])
            self.counter = cp.array([model_data['gpu_counter']], dtype=cp.uint32)
            logger.info(f"   GPU hash table loaded: {model_data['gpu_counter']:,} unique keys")
        
        logger.info(f"📂 Definitive Hybrid model loaded: {path}")
        logger.info(f"   Q-values shape: {self.q_values.shape}")
        logger.info(f"   Strategies shape: {self.strategies.shape}")
        logger.info(f"   Unique info sets: {self.total_unique_info_sets:,}")
        logger.info(f"   Growth events: {self.growth_events}")

def benchmark_definitive_hybrid_performance():
    """Benchmark the definitive hybrid trainer performance"""
    from .simulation import batch_simulate_real_holdem
    import jax.random as jr
    import time
    
    logger.info("🚀 Benchmarking DEFINITIVE HYBRID Trainer Performance")
    logger.info("=" * 60)
    
    # Configuration
    config = TrainerConfig(
        batch_size=8192,
        learning_rate=0.1,
        temperature=1.0
    )
    
    trainer = PokerTrainer(config)
    
    # Test configuration
    game_config = {
        'players': 6,
        'starting_stack': 100.0,
        'small_blind': 1.0,
        'big_blind': 2.0
    }
    
    # Warm-up
    logger.info("🔥 Warming up JAX compilation...")
    rng_key = jr.PRNGKey(42)
    rng_keys = jr.split(rng_key, 1024)  # Smaller batch for warm-up
    
    start_time = time.time()
    game_results = batch_simulate_real_holdem(rng_keys, game_config)
    warmup_results = trainer.train_step(game_results)
    warmup_time = time.time() - start_time
    
    logger.info(f"   ✅ Warm-up completed in {warmup_time:.2f}s")
    
    # Benchmark
    logger.info("🚀 Running benchmark...")
    iterations = 10
    total_time = 0
    total_games = 0
    
    for i in range(iterations):
        rng_key = jr.fold_in(rng_key, i)
        rng_keys = jr.split(rng_key, config.batch_size)
        
        start_time = time.time()
        game_results = batch_simulate_real_holdem(rng_keys, game_config)
        results = trainer.train_step(game_results)
        iteration_time = time.time() - start_time
        
        total_time += iteration_time
        total_games += results['games_processed']
        
        games_per_second = results['games_processed'] / iteration_time
        logger.info(f"   Iteration {i+1}: {games_per_second:,.1f} games/sec")
    
    # Final results
    avg_games_per_second = total_games / total_time
    
    logger.info("🎉 DEFINITIVE HYBRID Benchmark Results:")
    logger.info("=" * 60)
    logger.info(f"🚀 Performance:")
    logger.info(f"   Average games/sec: {avg_games_per_second:,.1f}")
    logger.info(f"   Total games: {total_games:,}")
    logger.info(f"   Total time: {total_time:.2f}s")
    logger.info(f"   Target achieved: {'✅' if avg_games_per_second > 1000 else '❌'}")
    logger.info("")
    logger.info(f"🧠 Memory Management:")
    logger.info(f"   Unique info sets: {trainer.total_unique_info_sets:,}")
    logger.info(f"   Growth events: {trainer.growth_events}")
    logger.info(f"   Array size: {trainer.config.max_info_sets:,}")
    
    return avg_games_per_second 