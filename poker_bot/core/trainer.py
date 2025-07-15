# poker_bot/core/trainer.py
import jax
import jax.numpy as jnp
import numpy as np
import logging
import pickle
from dataclasses import dataclass
from . import full_game_engine as fge            # expone fge.batch_play(...)
from jax import Array

logger = logging.getLogger(__name__)

# ---------- Config ----------
@dataclass
class TrainerConfig:
    batch_size: int = 128
    num_actions: int = 3          # ahora usas 3 acciones (fold, check/call, bet/raise)
    max_info_sets: int = 50_000

# ---------- Trainer ----------
class PokerTrainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.iteration = 0
        self.regrets  = jnp.zeros((config.max_info_sets, config.num_actions), dtype=jnp.float32)
        self.strategy = jnp.ones ((config.max_info_sets, config.num_actions), dtype=jnp.float32) / config.num_actions
        logger.info("PokerTrainer inicializado con motor CFR-JIT puro.")

    # ------------------------------------------------------------------ #
    def train(self, num_iterations: int, save_path: str, save_interval: int):
        logger.info(f"🚀 Iniciando entrenamiento por {num_iterations} iteraciones...")
        for _ in range(num_iterations):
            key = jax.random.PRNGKey(self.iteration)
            self.train_step(key)
            self.iteration += 1
            if self.iteration % save_interval == 0:
                self.save_model(f"{save_path}_iter_{self.iteration}.pkl")
            logger.info(f"Iteración {self.iteration} completada.")
        logger.info("🎉 Entrenamiento finalizado.")

    # ------------------------------------------------------------------ #
    def _cfr_backtracking(self, payoffs: np.ndarray, histories: np.ndarray):
        """
        payoffs:  (batch, 6)   – payoff final de cada jugador
        histories:(batch, MAX_GAME_LENGTH) – acciones por juego
        Devuelve indices y regrets para actualizar matrices grandes.
        """
        all_indices, all_regrets = [], []

        for b in range(self.config.batch_size):
            hist   = histories[b]
            payoff = payoffs[b]

            traj_states = []
            state = fge.initial_state_for_idx(b)          # helper JIT que devuelve estado inicial del batch
            for t in range(fge.MAX_GAME_LENGTH):
                a = int(hist[t])
                if a == -1:
                    break
                traj_states.append((state, a))
                state = fge.step(state, a)

            # Back-prop de regrets
            for st, act in reversed(traj_states):
                p   = int(st.cur_player[0])
                val = float(payoff[p])
                idx = p                                # bucket trivial

                mask = fge.get_legal_actions(st)
                regret = np.full(self.config.num_actions, -1e9, dtype=np.float32)
                for i in range(self.config.num_actions):
                    if mask[i]:
                        regret[i] = (val if i == act else 0.0) - val

                all_indices.append(idx)
                all_regrets.append(regret)

        if not all_indices:
            return (np.empty(0, np.int32),
                    np.empty((0, self.config.num_actions), np.float32))

        return (np.array(all_indices, np.int32),
                np.stack(all_regrets))

    # ------------------------------------------------------------------ #
    def train_step(self, key: Array):
        # 1. Simulamos el batch con el motor JIT
        keys = jax.random.split(key, self.config.batch_size)
        payoffs, histories = fge.batch_play(keys)

        # 2. Backtracking en Python puro
        indices, regrets = self._cfr_backtracking(
            np.asarray(payoffs), np.asarray(histories))

        # 3. Actualización vectorizada de regrets y estrategia
        if indices.size == 0:
            return

        self.regrets = self.regrets.at[indices].add(jnp.asarray(regrets))

        pos  = jnp.maximum(self.regrets[indices], 0.0)
        norm = pos.sum(axis=1, keepdims=True)
        norm = jnp.where(norm == 0, 1, norm)
        self.strategy = self.strategy.at[indices].set(pos / norm)

    # ------------------------------------------------------------------ #
    def save_model(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'regrets':   np.array(self.regrets),
                'strategy':  np.array(self.strategy),
                'iteration': self.iteration,
                'config':    self.config
            }, f)
        logger.info(f"💾 Modelo guardado en: {path}")

    def load_model(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.regrets   = jnp.array(data['regrets'])
        self.strategy  = jnp.array(data['strategy'])
        self.iteration = data['iteration']
        logger.info(f"📂 Modelo cargado desde: {path}")