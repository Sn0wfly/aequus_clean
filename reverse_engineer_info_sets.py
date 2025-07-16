#!/usr/bin/env python3
"""
REVERSE ENGINEERING: Descubrir qué combinaciones de buckets generan
los info sets realmente entrenados (1, 3, 6, 11, 13, etc.)
"""

import logging
import jax
import jax.numpy as jnp
import numpy as np
from poker_bot.core.trainer import (
    compute_mock_info_set, compute_advanced_info_set,
    unified_batch_simulation
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def reverse_engineer_info_sets():
    """
    Analiza los info sets realmente entrenados para descubrir
    qué combinaciones de buckets los generan.
    """
    logger.info("🔍 REVERSE ENGINEERING: Analizando info sets entrenados")
    logger.info("="*60)
    
    # 1. Obtener info sets realmente entrenados
    logger.info("🎲 Generando datos de entrenamiento...")
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 128)
    payoffs, histories, game_results = unified_batch_simulation(keys)
    
    logger.info("🔍 Extrayendo info sets y sus datos originales...")
    training_data = []
    
    # Extraer primeros 32 juegos para análisis detallado
    for game_idx in range(32):
        for player_idx in range(6):
            try:
                info_set = int(compute_advanced_info_set(game_results, player_idx, game_idx))
                
                # Extraer datos originales del juego
                hole_cards = game_results['hole_cards'][game_idx, player_idx]
                community_cards = game_results['final_community'][game_idx]
                pot_size = float(game_results['final_pot'][game_idx])
                
                # Calcular buckets manualmente (misma lógica que compute_advanced_info_set)
                hole_ranks = hole_cards // 4
                hole_suits = hole_cards % 4
                
                high_rank = int(max(hole_ranks))
                low_rank = int(min(hole_ranks))
                is_suited = bool(hole_suits[0] == hole_suits[1])
                is_pair = bool(hole_ranks[0] == hole_ranks[1])
                
                # Hand bucket
                if is_pair:
                    preflop_bucket = high_rank
                elif is_suited:
                    preflop_bucket = 13 + high_rank * 12 + low_rank
                else:
                    preflop_bucket = 169 + high_rank * 12 + low_rank
                hand_bucket = preflop_bucket % 169
                
                # Otros buckets
                street_bucket = 0  # Preflop
                position_bucket = player_idx
                stack_bucket = int(min(pot_size / 5.0, 19))
                pot_bucket = int(min(pot_size / 10.0, 9))
                active_bucket = min(player_idx, 4)
                
                training_data.append({
                    'info_set': info_set,
                    'hand_bucket': hand_bucket,
                    'position_bucket': position_bucket,
                    'stack_bucket': stack_bucket,
                    'pot_bucket': pot_bucket,
                    'active_bucket': active_bucket,
                    'pot_size': pot_size,
                    'high_rank': high_rank,
                    'low_rank': low_rank,
                    'is_suited': is_suited,
                    'is_pair': is_pair
                })
                
            except Exception as e:
                logger.warning(f"Error en game={game_idx}, player={player_idx}: {e}")
    
    logger.info(f"📊 Datos extraídos: {len(training_data)} muestras")
    
    # 2. Análizar los info sets más bajos (más comunes durante entrenamiento)
    sorted_data = sorted(training_data, key=lambda x: x['info_set'])
    
    logger.info("\n🔍 ANÁLISIS DE INFO SETS MÁS BAJOS:")
    logger.info("-" * 80)
    logger.info("Info Set | Hand | Pos | Stack | Pot | Active | Pot Size | Hand Type")
    logger.info("-" * 80)
    
    for i, data in enumerate(sorted_data[:20]):  # Primeros 20
        hand_type = "PAIR" if data['is_pair'] else ("SUITED" if data['is_suited'] else "OFFSUIT")
        hand_desc = f"{data['high_rank']:2d},{data['low_rank']:2d}"
        
        logger.info(f"{data['info_set']:8d} | {hand_desc:4s} | {data['position_bucket']:3d} | "
                   f"{data['stack_bucket']:5d} | {data['pot_bucket']:3d} | {data['active_bucket']:6d} | "
                   f"{data['pot_size']:8.1f} | {hand_type}")
    
    # 3. Estadísticas de buckets para los info sets bajos
    low_info_sets = [d for d in sorted_data if d['info_set'] < 100]
    
    if low_info_sets:
        logger.info(f"\n📊 ESTADÍSTICAS DE INFO SETS BAJOS (< 100):")
        logger.info(f"   Cantidad: {len(low_info_sets)}")
        
        # Analizar rangos de buckets
        stack_buckets = [d['stack_bucket'] for d in low_info_sets]
        pot_buckets = [d['pot_bucket'] for d in low_info_sets]
        active_buckets = [d['active_bucket'] for d in low_info_sets]
        hand_buckets = [d['hand_bucket'] for d in low_info_sets]
        
        logger.info(f"   Stack buckets: min={min(stack_buckets)}, max={max(stack_buckets)}, avg={np.mean(stack_buckets):.1f}")
        logger.info(f"   Pot buckets: min={min(pot_buckets)}, max={max(pot_buckets)}, avg={np.mean(pot_buckets):.1f}")
        logger.info(f"   Active buckets: min={min(active_buckets)}, max={max(active_buckets)}, avg={np.mean(active_buckets):.1f}")
        logger.info(f"   Hand buckets: min={min(hand_buckets)}, max={max(hand_buckets)}, avg={np.mean(hand_buckets):.1f}")
    
    # 4. Intentar recrear info sets específicos
    logger.info(f"\n🧪 RECREANDO INFO SETS ESPECÍFICOS:")
    logger.info("-" * 50)
    
    target_info_sets = [1, 3, 6, 11, 13, 14, 17, 19, 22, 26]
    
    for target in target_info_sets:
        matches = [d for d in training_data if d['info_set'] == target]
        if matches:
            match = matches[0]
            logger.info(f"Info set {target:2d}: hand={match['hand_bucket']:3d}, pos={match['position_bucket']:1d}, "
                       f"stack={match['stack_bucket']:1d}, pot={match['pot_bucket']:1d}, active={match['active_bucket']:1d}")
            
            # Verificar la fórmula
            calculated = (
                0 * 10000 +  # street_bucket
                match['hand_bucket'] * 50 +
                match['position_bucket'] * 8 +
                match['stack_bucket'] * 2 +
                match['pot_bucket'] * 1 +
                match['active_bucket']
            ) % 50000
            
            if calculated == target:
                logger.info(f"           ✅ Fórmula CORRECTA: {calculated}")
            else:
                logger.info(f"           ❌ Fórmula ERROR: esperado {target}, calculado {calculated}")
    
    # 5. Encontrar valores típicos para compute_mock_info_set
    if low_info_sets:
        typical_stack = int(np.median([d['stack_bucket'] for d in low_info_sets]))
        typical_pot = int(np.median([d['pot_bucket'] for d in low_info_sets]))
        typical_active = int(np.median([d['active_bucket'] for d in low_info_sets]))
        
        logger.info(f"\n🎯 VALORES TÍPICOS RECOMENDADOS:")
        logger.info(f"   stack_bucket = {typical_stack}")
        logger.info(f"   pot_bucket = {typical_pot}")
        logger.info(f"   active_bucket = {typical_active}")
        
        return {
            'recommended_stack': typical_stack,
            'recommended_pot': typical_pot,
            'recommended_active': typical_active,
            'low_info_sets_count': len(low_info_sets)
        }
    
    return None

if __name__ == "__main__":
    results = reverse_engineer_info_sets()
    
    if results:
        logger.info(f"\n🎉 RECOMENDACIÓN FINAL:")
        logger.info(f"   Usar stack_bucket = {results['recommended_stack']}")
        logger.info(f"   Usar pot_bucket = {results['recommended_pot']}")
        logger.info(f"   Usar active_bucket = {results['recommended_active']}")
        logger.info(f"   Basado en {results['low_info_sets_count']} info sets bajos")
    else:
        logger.info("\n❌ No se encontraron suficientes info sets bajos para análisis") 