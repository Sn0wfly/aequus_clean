#!/usr/bin/env python3
"""
DEBUG SCRIPT: Investigar por qué CFR no está aprendiendo
Problema: Todas las estrategias permanecen en 0.500 (uniforme)
"""

import jax
import jax.numpy as jnp
import numpy as np
from poker_bot.core.trainer import (
    PokerTrainer, TrainerConfig, _jitted_train_step,
    unified_batch_simulation, compute_advanced_info_set
)

def debug_regret_accumulation():
    """Debuggear paso a paso por qué los regrets no se acumulan"""
    print("🔍 DEBUGGING: Acumulación de Regrets")
    print("="*50)
    
    config = TrainerConfig()
    config.batch_size = 8  # Pequeño para debugging
    
    # Estado inicial
    regrets = jnp.zeros((config.max_info_sets, config.num_actions), dtype=jnp.float32)
    strategy = jnp.ones((config.max_info_sets, config.num_actions), dtype=jnp.float32) / config.num_actions
    
    key = jax.random.PRNGKey(42)
    
    print(f"📊 Estado inicial:")
    print(f"   - Regrets shape: {regrets.shape}")
    print(f"   - Strategy shape: {strategy.shape}")
    print(f"   - Regrets iniciales (muestra): {regrets[1000:1005, :3]}")
    print(f"   - Strategy inicial (muestra): {strategy[1000:1005, :3]}")
    
    # Ejecutar UN paso y ver qué pasa
    print(f"\n🔄 Ejecutando UN paso de CFR...")
    
    new_regrets, new_strategy = _jitted_train_step(regrets, strategy, key)
    
    print(f"\n📈 Después de UN paso:")
    print(f"   - Regrets cambiaron: {not jnp.allclose(regrets, new_regrets)}")
    print(f"   - Strategy cambió: {not jnp.allclose(strategy, new_strategy)}")
    
    if not jnp.allclose(regrets, new_regrets):
        regret_diff = jnp.abs(new_regrets - regrets)
        max_change = jnp.max(regret_diff)
        print(f"   - Máximo cambio en regrets: {max_change}")
        
        # Buscar los info sets que más cambiaron
        max_change_idx = jnp.unravel_index(jnp.argmax(regret_diff), regret_diff.shape)
        print(f"   - Info set con más cambio: {max_change_idx[0]}, acción {max_change_idx[1]}")
        print(f"   - Regret antes: {regrets[max_change_idx]}")
        print(f"   - Regret después: {new_regrets[max_change_idx]}")
    else:
        print("   ❌ ¡LOS REGRETS NO CAMBIARON!")
    
    if not jnp.allclose(strategy, new_strategy):
        strategy_diff = jnp.abs(new_strategy - strategy)
        max_strategy_change = jnp.max(strategy_diff)
        print(f"   - Máximo cambio en strategy: {max_strategy_change}")
    else:
        print("   ❌ ¡LA ESTRATEGIA NO CAMBIÓ!")
    
    # Análisis detallado del problema
    print(f"\n🔍 ANÁLISIS DETALLADO:")
    
    # Verificar si hay regrets positivos
    positive_regrets = jnp.maximum(new_regrets, 0.0)
    regret_sums = jnp.sum(positive_regrets, axis=1, keepdims=True)
    
    non_zero_regrets = jnp.sum(regret_sums > 1e-6)
    print(f"   - Info sets con regrets > 1e-6: {non_zero_regrets} de {config.max_info_sets}")
    print(f"   - Máximo regret sum: {jnp.max(regret_sums)}")
    print(f"   - Mínimo regret sum: {jnp.min(regret_sums)}")
    
    # Muestras de regret sums
    sample_regret_sums = regret_sums[1000:1010].flatten()
    print(f"   - Muestra de regret sums: {sample_regret_sums}")
    
    return new_regrets, new_strategy

def debug_game_simulation():
    """Debuggear si la simulación genera datos válidos"""
    print("\n🎮 DEBUGGING: Simulación de Juegos")
    print("="*50)
    
    key = jax.random.PRNGKey(123)
    keys = jax.random.split(key, 8)  # 8 juegos
    
    payoffs, histories, game_results = unified_batch_simulation(keys)
    
    print(f"📊 Resultados de simulación:")
    print(f"   - Payoffs shape: {payoffs.shape}")
    print(f"   - Histories shape: {histories.shape}")
    print(f"   - Game results keys: {list(game_results.keys())}")
    
    # Analizar payoffs
    print(f"\n💰 Análisis de Payoffs:")
    print(f"   - Payoffs únicos: {len(jnp.unique(payoffs))}")
    print(f"   - Rango payoffs: [{jnp.min(payoffs):.2f}, {jnp.max(payoffs):.2f}]")
    print(f"   - Std payoffs: {jnp.std(payoffs):.4f}")
    
    # Muestra de payoffs
    print(f"   - Muestra payoffs juego 0: {payoffs[0]}")
    print(f"   - Muestra payoffs juego 1: {payoffs[1]}")
    
    # Analizar historiales
    print(f"\n📜 Análisis de Historiales:")
    valid_actions = histories[histories >= 0]
    print(f"   - Acciones válidas: {len(valid_actions)}")
    print(f"   - Acciones únicas: {jnp.unique(valid_actions)}")
    
    if len(valid_actions) > 0:
        print(f"   - Distribución de acciones:")
        for action in range(6):
            count = jnp.sum(valid_actions == action)
            percentage = count / len(valid_actions) * 100
            print(f"     Acción {action}: {count} ({percentage:.1f}%)")
    else:
        print("   ❌ ¡NO HAY ACCIONES VÁLIDAS!")
    
    # Analizar info sets
    print(f"\n🎯 Análisis de Info Sets:")
    info_sets_found = []
    
    for game_idx in range(min(3, payoffs.shape[0])):  # Solo 3 juegos
        for player_idx in range(6):
            try:
                info_set_idx = compute_advanced_info_set(game_results, player_idx, game_idx)
                info_sets_found.append(int(info_set_idx))
            except Exception as e:
                print(f"   ❌ Error computing info set {game_idx}-{player_idx}: {e}")
    
    if info_sets_found:
        unique_info_sets = len(set(info_sets_found))
        print(f"   - Info sets únicos generados: {unique_info_sets}")
        print(f"   - Total info sets: {len(info_sets_found)}")
        print(f"   - Rango info sets: [{min(info_sets_found)}, {max(info_sets_found)}]")
        print(f"   - Muestra info sets: {info_sets_found[:10]}")
    else:
        print("   ❌ ¡NO SE GENERARON INFO SETS!")
    
    return payoffs, histories, game_results

def debug_cfr_math():
    """Debuggear la matemática del CFR paso a paso"""
    print("\n📐 DEBUGGING: Matemática CFR")
    print("="*50)
    
    # Simular un caso simple con payoffs conocidos
    print("🧮 Simulando caso CFR simple...")
    
    # Crear un batch pequeño con payoffs controlados
    payoffs = jnp.array([
        [10.0, -2.0, -2.0, -2.0, -2.0, -2.0],  # Jugador 0 gana
        [-2.0, 15.0, -2.0, -2.0, -2.0, -2.0],  # Jugador 1 gana
        [-2.0, -2.0, 8.0, -2.0, -2.0, -2.0],   # Jugador 2 gana
    ])
    
    print(f"   - Payoffs de prueba: {payoffs.shape}")
    print(f"   - Juego 0: {payoffs[0]}")
    print(f"   - Juego 1: {payoffs[1]}")
    print(f"   - Juego 2: {payoffs[2]}")
    
    # Simular que todos los jugadores tienen el mismo info set para simplificar
    test_info_set = 1000
    
    # Calcular regrets manualmente para el jugador 0 en cada juego
    print(f"\n🎯 Calculando regrets para info set {test_info_set}:")
    
    for game_idx in range(3):
        for player_idx in range(3):  # Solo primeros 3 jugadores
            player_payoff = payoffs[game_idx, player_idx]
            
            print(f"\n   Juego {game_idx}, Jugador {player_idx}:")
            print(f"   - Payoff real: {player_payoff}")
            
            # En CFR, regret = valor_acción - valor_esperado
            # Valor esperado = payoff actual (estrategia seguida)
            expected_value = player_payoff
            
            # Simular valores de acciones
            for action in range(6):
                # El valor de una acción depende de si habríamos ganado/perdido más
                if action == 0:  # FOLD
                    action_value = -1.0 if player_payoff > 0 else 0.0  # Evita pérdidas pero pierde ganancias
                elif action <= 2:  # CHECK/CALL (pasivo)
                    action_value = player_payoff * 0.8  # Conservador
                else:  # BET/RAISE/ALLIN (agresivo)
                    action_value = player_payoff * 1.2 if player_payoff > 0 else player_payoff * 1.5
                
                regret = action_value - expected_value
                print(f"     Acción {action}: valor={action_value:.2f}, regret={regret:.2f}")
    
    return payoffs

def main():
    """Ejecutar todos los debugging tests"""
    print("🚨 DEBUGGING CFR - ESTRATEGIAS NO APRENDEN")
    print("="*60)
    
    # Test 1: Verificar acumulación de regrets
    try:
        debug_regret_accumulation()
    except Exception as e:
        print(f"❌ Error en debug_regret_accumulation: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Verificar simulación de juegos
    try:
        debug_game_simulation()
    except Exception as e:
        print(f"❌ Error en debug_game_simulation: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Verificar matemática CFR
    try:
        debug_cfr_math()
    except Exception as e:
        print(f"❌ Error en debug_cfr_math: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎯 CONCLUSIONES:")
    print("   1. ¿Los regrets cambian después de un paso? (Ver output arriba)")
    print("   2. ¿La simulación genera payoffs diversos? (Ver output arriba)")
    print("   3. ¿Los info sets se mapean correctamente? (Ver output arriba)")
    print("   4. Si todo parece correcto, el problema está en la lógica CFR interna.")

if __name__ == "__main__":
    main() 