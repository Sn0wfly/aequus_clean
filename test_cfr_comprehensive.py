#!/usr/bin/env python3
"""
Test comprehensivo del CFR corregido con más iteraciones y debug de evaluación.
"""

import logging
import sys
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import global para evitar problemas de scope
from poker_bot.core.trainer import PokerTrainer, TrainerConfig
from poker_bot.core.trainer import evaluate_poker_intelligence, compute_mock_info_set
import jax.numpy as jnp

def main():
    try:
        logger.info("🔧 TESTING CFR COMPREHENSIVO - MÁS ITERACIONES")
        logger.info("="*60)
        
        logger.info("✅ Módulos importados correctamente")
        
        # Configuración más robusta
        config = TrainerConfig()
        config.batch_size = 64
        
        # Crear trainer
        trainer = PokerTrainer(config)
        
        # TEST 1: Entrenamiento más largo (50 iteraciones)
        logger.info("🚀 Entrenando 50 iteraciones (más tiempo para aprender conceptos)...")
        trainer.train(
            num_iterations=50,  # MÁS ITERACIONES
            save_path="models/test_cfr_comprehensive",
            save_interval=25,
            snapshot_iterations=[10, 25, 50]  # Más snapshots
        )
        
        # TEST 2: Debug detallado de las funciones de evaluación
        logger.info("\n🔍 DEBUG DETALLADO DE EVALUACIÓN DE POKER IQ")
        logger.info("="*60)
        
        # Debug cada función individualmente
        debug_poker_iq_functions(trainer.strategy, config)
        
        # TEST 3: Análisis de estrategias específicas
        logger.info("\n🎯 ANÁLISIS DE ESTRATEGIAS ESPECÍFICAS")
        logger.info("="*50)
        
        analyze_specific_strategies(trainer.strategy, config)
        
        logger.info("\n✅ Test comprehensivo completado")
        
    except Exception as e:
        logger.error(f"❌ Error en test comprehensivo: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

def debug_poker_iq_functions(strategy, config):
    """
    Debug detallado de cada función de evaluación para ver por qué no detectan mejoras.
    """
    
    # TEST 1: Hand Strength Awareness (debug detallado)
    logger.info("🧪 DEBUG: Hand Strength Awareness")
    
    # Info sets específicos
    aa_info_set = compute_mock_info_set(hole_ranks=[12, 12], is_suited=False, position=2)
    trash_info_set = compute_mock_info_set(hole_ranks=[5, 0], is_suited=False, position=2)
    
    logger.info(f"   - AA info set: {aa_info_set}")
    logger.info(f"   - 72o info set: {trash_info_set}")
    
    if aa_info_set < config.max_info_sets and trash_info_set < config.max_info_sets:
        aa_strategy = strategy[aa_info_set]
        trash_strategy = strategy[trash_info_set]
        
        aa_aggression = float(jnp.sum(aa_strategy[3:6]))
        trash_aggression = float(jnp.sum(trash_strategy[3:6]))
        
        logger.info(f"   - AA estrategia: {[float(x) for x in aa_strategy]}")
        logger.info(f"   - 72o estrategia: {[float(x) for x in trash_strategy]}")
        logger.info(f"   - AA aggression: {aa_aggression:.3f}")
        logger.info(f"   - 72o aggression: {trash_aggression:.3f}")
        logger.info(f"   - Diferencia: {aa_aggression - trash_aggression:.3f}")
        
        # Análisis del threshold
        threshold_needed = 0.1
        if aa_aggression > trash_aggression + threshold_needed:
            logger.info(f"   ✅ AA es más agresivo que 72o (+{aa_aggression - trash_aggression:.3f} > {threshold_needed})")
            hand_strength_score = 25.0
        elif aa_aggression > trash_aggression:
            logger.info(f"   🟡 AA ligeramente más agresivo (+{aa_aggression - trash_aggression:.3f} < {threshold_needed})")
            hand_strength_score = 15.0
        else:
            logger.info(f"   ❌ AA NO es más agresivo que 72o ({aa_aggression:.3f} vs {trash_aggression:.3f})")
            hand_strength_score = 0.0
    else:
        logger.error(f"   ❌ Info sets fuera de rango: AA={aa_info_set}, 72o={trash_info_set}")
        hand_strength_score = 0.0
    
    # TEST 2: Position Awareness (debug detallado)
    logger.info("\n🧪 DEBUG: Position Awareness")
    
    # J-T suited en diferentes posiciones
    marginal_hand = [10, 9]
    early_pos_info = compute_mock_info_set(hole_ranks=marginal_hand, is_suited=True, position=0)
    late_pos_info = compute_mock_info_set(hole_ranks=marginal_hand, is_suited=True, position=5)
    
    logger.info(f"   - JTs early pos info set: {early_pos_info}")
    logger.info(f"   - JTs late pos info set: {late_pos_info}")
    
    if early_pos_info < config.max_info_sets and late_pos_info < config.max_info_sets:
        early_strategy = strategy[early_pos_info]
        late_strategy = strategy[late_pos_info]
        
        early_aggression = float(jnp.sum(early_strategy[3:6]))
        late_aggression = float(jnp.sum(late_strategy[3:6]))
        
        early_fold = float(early_strategy[0])
        late_fold = float(late_strategy[0])
        
        logger.info(f"   - Early estrategia: {[float(x) for x in early_strategy]}")
        logger.info(f"   - Late estrategia: {[float(x) for x in late_strategy]}")
        logger.info(f"   - Early aggression: {early_aggression:.3f}")
        logger.info(f"   - Late aggression: {late_aggression:.3f}")
        logger.info(f"   - Early fold: {early_fold:.3f}")
        logger.info(f"   - Late fold: {late_fold:.3f}")
        
        # Análisis de position awareness
        aggression_diff = late_aggression - early_aggression
        fold_diff = early_fold - late_fold
        
        logger.info(f"   - Aggression diff (late-early): {aggression_diff:.3f}")
        logger.info(f"   - Fold diff (early-late): {fold_diff:.3f}")
        
        position_score = 0.0
        if late_aggression > early_aggression + 0.1:
            position_score += 15.0
            logger.info("   ✅ Más agresivo en late position (+15)")
        elif late_aggression > early_aggression + 0.05:
            position_score += 10.0
            logger.info("   🟡 Ligeramente más agresivo en late position (+10)")
            
        if early_fold > late_fold + 0.05:
            position_score += 10.0
            logger.info("   ✅ Más fold en early position (+10)")
        elif early_fold > late_fold:
            position_score += 5.0
            logger.info("   🟡 Ligeramente más fold en early position (+5)")
            
        logger.info(f"   📊 Position score total: {position_score:.1f}/25")
    else:
        logger.error(f"   ❌ Info sets fuera de rango: Early={early_pos_info}, Late={late_pos_info}")
        position_score = 0.0
    
    # TEST 3: Suited vs Offsuit (debug detallado)
    logger.info("\n🧪 DEBUG: Suited Awareness")
    
    # AJ suited vs offsuit
    test_hand = [12, 10]  # A-J
    suited_info = compute_mock_info_set(hole_ranks=test_hand, is_suited=True, position=3)
    offsuit_info = compute_mock_info_set(hole_ranks=test_hand, is_suited=False, position=3)
    
    logger.info(f"   - AJs info set: {suited_info}")
    logger.info(f"   - AJo info set: {offsuit_info}")
    
    if suited_info < config.max_info_sets and offsuit_info < config.max_info_sets:
        suited_strategy = strategy[suited_info]
        offsuit_strategy = strategy[offsuit_info]
        
        suited_aggression = float(jnp.sum(suited_strategy[3:6]))
        offsuit_aggression = float(jnp.sum(offsuit_strategy[3:6]))
        
        suited_fold = float(suited_strategy[0])
        offsuit_fold = float(offsuit_strategy[0])
        
        logger.info(f"   - AJs estrategia: {[float(x) for x in suited_strategy]}")
        logger.info(f"   - AJo estrategia: {[float(x) for x in offsuit_strategy]}")
        logger.info(f"   - AJs aggression: {suited_aggression:.3f}")
        logger.info(f"   - AJo aggression: {offsuit_aggression:.3f}")
        logger.info(f"   - AJs fold: {suited_fold:.3f}")
        logger.info(f"   - AJo fold: {offsuit_fold:.3f}")
        
        aggression_diff = suited_aggression - offsuit_aggression
        fold_diff = offsuit_fold - suited_fold
        
        logger.info(f"   - Aggression diff (suited-offsuit): {aggression_diff:.3f}")
        logger.info(f"   - Fold diff (offsuit-suited): {fold_diff:.3f}")
        
        suited_score = 0.0
        if suited_aggression > offsuit_aggression + 0.05:
            suited_score += 7.0
            logger.info("   ✅ Suited más agresivo (+7)")
        elif suited_aggression > offsuit_aggression:
            suited_score += 3.0
            logger.info("   🟡 Suited ligeramente más agresivo (+3)")
            
        if suited_fold < offsuit_fold - 0.03:
            suited_score += 6.0
            logger.info("   ✅ Suited fold menos (+6)")
        elif suited_fold < offsuit_fold:
            suited_score += 2.0
            logger.info("   🟡 Suited fold ligeramente menos (+2)")
            
        logger.info(f"   📊 Suited score: {suited_score:.1f}/20")
    else:
        logger.error(f"   ❌ Info sets fuera de rango: Suited={suited_info}, Offsuit={offsuit_info}")
    
    # RESUMEN
    logger.info(f"\n📊 RESUMEN DEBUG:")
    logger.info(f"   - Hand Strength: {hand_strength_score:.1f}/25")
    logger.info(f"   - Position: {position_score:.1f}/25") 
    logger.info(f"   - Suited: {suited_score:.1f}/20")

def analyze_specific_strategies(strategy, config):
    """
    Analiza estrategias de manos específicas para ver cómo evolucionaron.
    """
    
    # Manos de test
    test_hands = [
        ([12, 12], False, 2, "AA (Pocket Aces)"),
        ([11, 11], False, 2, "KK (Pocket Kings)"),
        ([10, 9], True, 2, "JTs (Jack-Ten suited)"),
        ([10, 9], False, 2, "JTo (Jack-Ten offsuit)"),
        ([5, 0], False, 2, "72o (Seven-Deuce trash)"),
        ([3, 1], False, 2, "52o (Five-Deuce trash)"),
    ]
    
    logger.info("🎯 ESTRATEGIAS ESPECÍFICAS ENTRENADAS:")
    logger.info("-" * 60)
    
    for hole_ranks, is_suited, position, name in test_hands:
        info_set = compute_mock_info_set(hole_ranks, is_suited, position)
        
        if info_set < config.max_info_sets:
            hand_strategy = strategy[info_set]
            
            # Calcular métricas
            fold_prob = float(hand_strategy[0])
            check_call = float(jnp.sum(hand_strategy[1:3]))
            aggression = float(jnp.sum(hand_strategy[3:6]))
            
            logger.info(f"📋 {name:20s} (info_set: {info_set:5d})")
            logger.info(f"   Estrategia: {[f'{float(x):.3f}' for x in hand_strategy]}")
            logger.info(f"   Fold: {fold_prob:.3f}, Check/Call: {check_call:.3f}, Aggression: {aggression:.3f}")
            
            # Clasificar comportamiento
            if aggression > 0.5:
                behavior = "🔥 MUY AGRESIVO"
            elif aggression > 0.3:
                behavior = "⚡ AGRESIVO"
            elif fold_prob > 0.6:
                behavior = "🛡️ MUY PASIVO"
            elif fold_prob > 0.4:
                behavior = "😐 PASIVO"
            else:
                behavior = "⚖️ BALANCEADO"
                
            logger.info(f"   Comportamiento: {behavior}")
            logger.info("")
        else:
            logger.warning(f"⚠️ {name}: Info set {info_set} fuera de rango")

if __name__ == "__main__":
    main() 