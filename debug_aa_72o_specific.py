#!/usr/bin/env python3
"""
DEBUG ESPECÍFICO: ¿Se están entrenando los info sets de AA y 72o?
"""

import jax
import jax.numpy as jnp
from poker_bot.core.trainer import (
    PokerTrainer, TrainerConfig, _jitted_train_step,
    unified_batch_simulation, compute_advanced_info_set, compute_mock_info_set
)

def debug_specific_info_sets():
    """Verificar si AA y 72o específicos están siendo entrenados"""
    print("🔍 DEBUG: Info sets específicos de AA y 72o")
    print("="*50)
    
    # Calcular info sets objetivo
    aa_info = compute_mock_info_set([12, 12], False, 2)  # AA middle position
    trash_info = compute_mock_info_set([5, 0], False, 2)  # 72o middle position
    
    print(f"🎯 Target info sets:")
    print(f"   - AA info set: {aa_info}")
    print(f"   - 72o info set: {trash_info}")
    
    # Configuración
    config = TrainerConfig()
    config.batch_size = 256  # GRANDE para mayor probabilidad
    
    # Estado inicial
    regrets = jnp.zeros((config.max_info_sets, config.num_actions), dtype=jnp.float32)
    strategy = jnp.ones((config.max_info_sets, config.num_actions), dtype=jnp.float32) / config.num_actions
    
    # Rastrear estos info sets específicos a través de múltiples pasos
    aa_trained_steps = []
    trash_trained_steps = []
    
    print(f"\n🔄 Entrenando por 50 pasos y rastreando info sets específicos...")
    
    for step in range(50):
        key = jax.random.PRNGKey(step + 1000)
        
        # Guardar regrets previos para estos info sets
        prev_aa_regrets = regrets[aa_info] if aa_info < config.max_info_sets else jnp.zeros(6)
        prev_trash_regrets = regrets[trash_info] if trash_info < config.max_info_sets else jnp.zeros(6)
        
        # Ejecutar paso de entrenamiento
        regrets, strategy = _jitted_train_step(regrets, strategy, key)
        
        # Verificar si estos info sets fueron modificados
        if aa_info < config.max_info_sets:
            aa_regrets_changed = not jnp.allclose(prev_aa_regrets, regrets[aa_info], atol=1e-6)
            if aa_regrets_changed:
                aa_trained_steps.append(step)
        
        if trash_info < config.max_info_sets:
            trash_regrets_changed = not jnp.allclose(prev_trash_regrets, regrets[trash_info], atol=1e-6)
            if trash_regrets_changed:
                trash_trained_steps.append(step)
        
        # Log progreso cada 10 pasos
        if (step + 1) % 10 == 0:
            positive_regrets = jnp.maximum(regrets, 0.0)
            regret_sums = jnp.sum(positive_regrets, axis=1, keepdims=True)
            total_trained = jnp.sum(regret_sums > 1e-6)
            print(f"   Paso {step + 1}: {total_trained} info sets con regrets positivos")
    
    print(f"\n📊 RESULTADOS:")
    print(f"   - AA entrenado en pasos: {aa_trained_steps}")
    print(f"   - 72o entrenado en pasos: {trash_trained_steps}")
    print(f"   - AA entrenado {len(aa_trained_steps)} veces de 50")
    print(f"   - 72o entrenado {len(trash_trained_steps)} veces de 50")
    
    # Verificar estrategias finales
    if aa_info < config.max_info_sets and trash_info < config.max_info_sets:
        final_aa_strategy = strategy[aa_info]
        final_trash_strategy = strategy[trash_info]
        
        print(f"\n🎯 ESTRATEGIAS FINALES:")
        print(f"   AA strategy: {[f'{x:.3f}' for x in final_aa_strategy]}")
        print(f"   72o strategy: {[f'{x:.3f}' for x in final_trash_strategy]}")
        
        strategies_identical = jnp.allclose(final_aa_strategy, final_trash_strategy, atol=1e-6)
        print(f"   ¿Idénticas? {strategies_identical}")
        
        # Verificar si son uniformes
        uniform_strategy = jnp.ones(6) / 6
        aa_is_uniform = jnp.allclose(final_aa_strategy, uniform_strategy, atol=1e-6)
        trash_is_uniform = jnp.allclose(final_trash_strategy, uniform_strategy, atol=1e-6)
        
        print(f"   AA es uniforme? {aa_is_uniform}")
        print(f"   72o es uniforme? {trash_is_uniform}")
        
        return len(aa_trained_steps), len(trash_trained_steps), strategies_identical
    else:
        print(f"❌ Info sets fuera de rango")
        return 0, 0, True

def debug_info_set_generation_during_training():
    """Verificar qué info sets se generan durante el entrenamiento real"""
    print(f"\n🎮 DEBUG: Info sets generados durante simulación")
    print("="*50)
    
    # Hacer varias simulaciones y ver qué info sets aparecen
    info_sets_seen = set()
    
    for trial in range(10):  # 10 trials
        key = jax.random.PRNGKey(trial + 2000)
        keys = jax.random.split(key, 128)  # 128 juegos por trial
        
        payoffs, histories, game_results = unified_batch_simulation(keys)
        
        # Extraer todos los info sets de esta simulación
        for game_idx in range(min(10, payoffs.shape[0])):  # Solo primeros 10 juegos por velocidad
            for player_idx in range(6):
                try:
                    info_set_idx = compute_advanced_info_set(game_results, player_idx, game_idx)
                    info_sets_seen.add(int(info_set_idx))
                except Exception as e:
                    print(f"   Error en info set {game_idx}-{player_idx}: {e}")
    
    print(f"📊 Simulación completada:")
    print(f"   - Info sets únicos vistos: {len(info_sets_seen)}")
    print(f"   - Rango: [{min(info_sets_seen)}, {max(info_sets_seen)}]")
    
    # Verificar si nuestros targets están en los vistos
    aa_info = compute_mock_info_set([12, 12], False, 2)
    trash_info = compute_mock_info_set([5, 0], False, 2)
    
    aa_seen = aa_info in info_sets_seen
    trash_seen = trash_info in info_sets_seen
    
    print(f"\n🎯 TARGET INFO SETS:")
    print(f"   - AA ({aa_info}) visto durante simulación: {aa_seen}")
    print(f"   - 72o ({trash_info}) visto durante simulación: {trash_seen}")
    
    if not aa_seen and not trash_seen:
        print(f"\n❌ PROBLEMA DETECTADO:")
        print(f"   ¡Los info sets de AA y 72o nunca aparecen durante la simulación!")
        print(f"   Esto explica por qué nunca se entrenan.")
        print(f"   Posible causa: Diferencia entre compute_mock_info_set y compute_advanced_info_set")
    
    # Mostrar muestra de info sets vistos
    sample_info_sets = sorted(list(info_sets_seen))[:20]
    print(f"\n📋 Muestra de info sets vistos: {sample_info_sets}")
    
    return aa_seen, trash_seen, info_sets_seen

def main():
    """Ejecutar debugging completo"""
    print("🚨 DEBUG ESPECÍFICO: AA vs 72o")
    print("="*60)
    
    # Test 1: ¿Se entrenan estos info sets específicos?
    try:
        aa_count, trash_count, identical = debug_specific_info_sets()
    except Exception as e:
        print(f"❌ Error en debug_specific_info_sets: {e}")
        aa_count, trash_count, identical = 0, 0, True
    
    # Test 2: ¿Aparecen estos info sets durante la simulación?
    try:
        aa_seen, trash_seen, all_info_sets = debug_info_set_generation_during_training()
    except Exception as e:
        print(f"❌ Error en debug_info_set_generation: {e}")
        aa_seen, trash_seen = False, False
    
    print(f"\n🎯 DIAGNÓSTICO FINAL:")
    print(f"   - AA entrenado: {aa_count}/50 pasos")
    print(f"   - 72o entrenado: {trash_count}/50 pasos")
    print(f"   - AA visto en simulación: {aa_seen}")
    print(f"   - 72o visto en simulación: {trash_seen}")
    print(f"   - Estrategias idénticas: {identical}")
    
    if not aa_seen or not trash_seen:
        print(f"\n🔍 CAUSA RAÍZ IDENTIFICADA:")
        print(f"   Los info sets objetivo nunca aparecen durante la simulación.")
        print(f"   Solución: Verificar consistency entre compute_mock_info_set y compute_advanced_info_set")
    elif aa_count == 0 and trash_count == 0:
        print(f"\n🔍 CAUSA RAÍZ IDENTIFICADA:")
        print(f"   Los info sets aparecen pero nunca se entrenan.")
        print(f"   Solución: Verificar lógica de regret update en _jitted_train_step")
    else:
        print(f"\n🤔 ESTADO INTERMEDIO:")
        print(f"   Los info sets se ven y entrenan ocasionalmente.")
        print(f"   Necesita más iteraciones para convergencia.")

if __name__ == "__main__":
    main() 