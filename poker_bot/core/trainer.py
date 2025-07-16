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
    num_actions: int = 3
    max_info_sets: int = 50_000

# ---------- JAX-Native CFR Step ----------
@jax.jit
def _jitted_train_step(regrets, strategy, key):
    """
    CFR real sin bucles Python.
    1. Simula batch
    2. Reconstruye trayectorias
    3. Actualiza regrets y estrategia
    """
    cfg = TrainerConfig()  # valores constantes dentro del jit
    keys = jax.random.split(key, cfg.batch_size)
    payoffs, histories = fge.batch_play(keys)

    def game_step(carry, step_idx):
        regrets, strategy = carry
        # Por cada juego y paso
        def one_game(batch_idx):
            payoff = payoffs[batch_idx]
            history = histories[batch_idx]
            action = history[step_idx]
            valid = action != -1

            def do_update():
                state = fge.initial_state_for_idx(batch_idx)
                # Reconstruir hasta step_idx
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

                info_set_idx = player_idx % cfg.max_info_sets
                new_regrets = regrets.at[info_set_idx].add(regret_delta)
                pos = jnp.maximum(new_regrets[info_set_idx], 0.0)
                norm = jnp.sum(pos)
                new_strat = jnp.where(norm > 0, pos / norm, jnp.ones(cfg.num_actions) / cfg.num_actions)
                new_strategy = strategy.at[info_set_idx].set(new_strat)
                return new_regrets, new_strategy

            return lax.cond(valid, do_update, lambda: (regrets, strategy))

        return jax.vmap(one_game)(jnp.arange(cfg.batch_size)), None

    final_regrets, final_strategy = lax.scan(
        game_step,
        (regrets, strategy),
        jnp.arange(fge.MAX_GAME_LENGTH)
    )[0]

    return final_regrets, final_strategy

# ---------- Trainer ----------
class PokerTrainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.iteration = 0
        self.regrets  = jnp.zeros((config.max_info_sets, config.num_actions), dtype=jnp.float32)
        self.strategy = jnp.ones((config.max_info_sets, config.num_actions), dtype=jnp.float32) / config.num_actions
        logger.info("PokerTrainer CFR-JIT listo.")

    def train(self, num_iterations: int, save_path: str, save_interval: int):
        key = jax.random.PRNGKey(0)
        for i in range(1, num_iterations + 1):
            self.iteration += 1
            iter_key = jax.random.fold_in(key, self.iteration)
            self.regrets, self.strategy = _jitted_train_step(self.regrets, self.strategy, iter_key)
            self.regrets.block_until_ready()
            logger.info(f"Iteración {self.iteration} completada.")
            if self.iteration % save_interval == 0:
                self.save_model(f"{save_path}_iter_{self.iteration}.pkl")
        logger.info("🎉 Entrenamiento finalizado.")

    def save_model(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'regrets':   np.asarray(self.regrets),
                'strategy':  np.asarray(self.strategy),
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