import jax
import jax.numpy as jnp
from poker_bot.core.full_game_engine import *

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    batch_size = 1
    # Política aleatoria placeholder
    policy_logits = jax.random.normal(key, (6, 14))
    final_states, payoffs = batch_play_game(batch_size, policy_logits, key)
    print("Final state:", final_states)
    print("Payoffs:", payoffs) 