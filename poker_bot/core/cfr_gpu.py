import jax
import jax.numpy as jnp

def cfr_step_gpu(batch_states, regrets_prev, strategy_prev, lr=0.1):
    """
    CFR regret-matching step puramente en JAX.
    Entradas:
        batch_states: (B, state_size) – no usado aquí, se recibe por compatibilidad
        regrets_prev: (B, num_actions) – regrets acumulados
        strategy_prev: (B, num_actions) – estrategia anterior (no se usa)
        lr: float – learning rate
    Salidas:
        new_regrets, new_strategy ambos en GPU
    """
    # Regret-Matching
    positive_regrets = jnp.maximum(regrets_prev, 0.0)
    sum_pos = jnp.sum(positive_regrets, axis=1, keepdims=True)
    # Evitar división por cero
    sum_pos = jnp.where(sum_pos == 0.0, 1.0, sum_pos)
    new_strategy = positive_regrets / sum_pos

    # Actualizar regrets (placeholder: en tu pipeline real vendrán los cf_values)
    new_regrets = regrets_prev  # se actualizarán luego con scatter_update

    return new_regrets, new_strategy 