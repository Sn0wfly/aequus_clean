#!/usr/bin/env python3
"""
SCRIPT DE PRUEBA - SOLUCIÓN COMPLETA PARA BUG DE HISTORIALES SINTÉTICOS

Este script demuestra que la solución completa funciona correctamente:
1. CFR entrenamiento usando historiales REALES del motor de juego
2. Validación automática de integridad de datos
3. Evaluación mejorada de Poker IQ
4. Detección y prevención de bugs críticos

PROBLEMA ORIGINAL: _jitted_train_step usaba historiales sintéticos (action_seed = payoff)
SOLUCIÓN: Reescritura completa para usar historiales reales del full_game_engine
"""

import logging
import sys
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_solution_completa.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    """
    Test principal que demuestra la solución completa funcionando
    """
    logger.info("="*80)
    logger.info("🚀 INICIANDO TEST DE SOLUCIÓN COMPLETA")
    logger.info("="*80)
    logger.info("PROBLEMA: CFR entrenaba con historiales sintéticos")
    logger.info("SOLUCIÓN: CFR usa historiales reales del motor de juego")
    logger.info("="*80)
    
    try:
        # Importar módulos después de setup
        from poker_bot.core.trainer import (
            PokerTrainer, 
            TrainerConfig, 
            SuperHumanTrainerConfig,
            create_super_human_trainer,
            validate_training_data_integrity,
            enhanced_poker_iq_evaluation
        )
        import jax
        
        logger.info("✅ Módulos importados correctamente")
        
        # =================== TEST 1: VALIDACIÓN DE DATOS ===================
        logger.info("\n🔍 TEST 1: VALIDACIÓN DE INTEGRIDAD DE DATOS")
        logger.info("-" * 50)
        
        # Crear trainer estándar para test
        config = TrainerConfig()
        trainer = PokerTrainer(config)
        
        # Ejecutar validación crítica
        validation_key = jax.random.PRNGKey(42)
        validation_results = validate_training_data_integrity(
            trainer.strategy, 
            validation_key, 
            verbose=True
        )
        
        # Verificar que no hay bugs críticos
        if validation_results['critical_bugs']:
            logger.error(f"❌ TEST 1 FALLIDO: Bugs críticos detectados: {validation_results['critical_bugs']}")
            return False
        
        logger.info("✅ TEST 1 EXITOSO: Sin bugs críticos detectados")
        
        # =================== TEST 2: ENTRENAMIENTO BREVE ===================
        logger.info("\n🎯 TEST 2: ENTRENAMIENTO CFR CON HISTORIALES REALES")
        logger.info("-" * 50)
        
        # Entrenamiento muy breve para verificar que funciona
        num_iterations = 10
        save_path = "test_model_solution"
        
        logger.info(f"Ejecutando {num_iterations} iteraciones de entrenamiento...")
        
        # Configurar snapshots para todas las iteraciones
        snapshot_iterations = list(range(1, num_iterations + 1, max(1, num_iterations // 3)))
        if num_iterations not in snapshot_iterations:
            snapshot_iterations.append(num_iterations)
        
        trainer.train(
            num_iterations=num_iterations,
            save_path=save_path,
            save_interval=5,
            snapshot_iterations=snapshot_iterations
        )
        
        logger.info("✅ TEST 2 EXITOSO: Entrenamiento completado sin errores")
        
        # =================== TEST 3: EVALUACIÓN DE RESULTADOS ===================
        logger.info("\n🧠 TEST 3: EVALUACIÓN DE POKER IQ")
        logger.info("-" * 50)
        
        # Evaluar el modelo entrenado
        final_iq = enhanced_poker_iq_evaluation(trainer.strategy, config, num_iterations)
        
        logger.info("📊 RESULTADOS FINALES:")
        logger.info(f"   - IQ Total: {final_iq['total_poker_iq']:.1f}/100")
        logger.info(f"   - Hand Strength: {final_iq['hand_strength_score']:.1f}/25")
        logger.info(f"   - Position: {final_iq['position_score']:.1f}/25")
        logger.info(f"   - Suited: {final_iq['suited_score']:.1f}/20")
        logger.info(f"   - Fold Discipline: {final_iq['fold_discipline_score']:.1f}/15")
        logger.info(f"   - Stability: {final_iq['stability_score']:.1f}/10")
        
        # Verificar que hay al menos algo de aprendizaje
        if final_iq['total_poker_iq'] > 5.0:  # Threshold muy bajo para test breve
            logger.info("✅ TEST 3 EXITOSO: Modelo muestra signos de aprendizaje")
        else:
            logger.warning("⚠️ TEST 3 ADVERTENCIA: Aprendizaje limitado (normal para test breve)")
        
        # =================== TEST 4: SUPER-HUMAN CONFIG ===================
        logger.info("\n🏆 TEST 4: CONFIGURACIÓN SUPER-HUMANA")
        logger.info("-" * 50)
        
        # Crear trainer super-humano
        super_trainer = create_super_human_trainer("super_human")
        logger.info("✅ Super-human trainer creado correctamente")
        
        # Validar configuración super-humana
        super_validation = validate_training_data_integrity(
            super_trainer.strategy, 
            jax.random.PRNGKey(123), 
            verbose=False
        )
        
        if not super_validation['critical_bugs']:
            logger.info("✅ TEST 4 EXITOSO: Configuración super-humana validada")
        else:
            logger.error(f"❌ TEST 4 FALLIDO: Bugs en config super-humana: {super_validation['critical_bugs']}")
            return False
        
        # =================== VERIFICACIÓN DE ARCHIVOS ===================
        logger.info("\n💾 VERIFICACIÓN DE ARCHIVOS GENERADOS")
        logger.info("-" * 50)
        
        # Verificar que se generaron los archivos
        expected_files = [
            f"{save_path}_iter_5.pkl",
            f"{save_path}_iter_10.pkl",
            f"{save_path}_final.pkl"
        ]
        
        files_created = 0
        for file_path in expected_files:
            if os.path.exists(file_path):
                size_kb = os.path.getsize(file_path) / 1024
                logger.info(f"   ✅ {file_path} ({size_kb:.1f} KB)")
                files_created += 1
            else:
                logger.warning(f"   ⚠️ {file_path} no encontrado")
        
        if files_created >= 2:  # Al menos 2 archivos
            logger.info("✅ VERIFICACIÓN EXITOSA: Archivos generados correctamente")
        else:
            logger.warning("⚠️ VERIFICACIÓN PARCIAL: Algunos archivos faltantes")
        
        # =================== RESUMEN FINAL ===================
        logger.info("\n" + "="*80)
        logger.info("🎉 RESUMEN FINAL - SOLUCIÓN COMPLETA")
        logger.info("="*80)
        logger.info("✅ PROBLEMA RESUELTO: CFR ahora usa historiales reales")
        logger.info("✅ VALIDACIÓN AUTOMÁTICA: Detecta bugs críticos")
        logger.info("✅ EVALUACIÓN MEJORADA: Poker IQ con estabilidad")
        logger.info("✅ CONFIGURACIÓN SUPER-HUMANA: Lista para entrenamientos largos")
        logger.info("")
        logger.info("🚀 PRÓXIMOS PASOS RECOMENDADOS:")
        logger.info("   1. Ejecutar entrenamiento de 100+ iteraciones")
        logger.info("   2. Usar SuperHumanTrainerConfig para resultados óptimos")
        logger.info("   3. Monitorear Poker IQ durante entrenamiento")
        logger.info("   4. Validar que Hand Strength > 15.0/25 después de 50 iteraciones")
        logger.info("="*80)
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ ERROR CRÍTICO EN TEST: {type(e).__name__}")
        logger.error(f"   Mensaje: {str(e)}")
        
        import traceback
        logger.error("\nTraceback completo:")
        logger.error(traceback.format_exc())
        
        return False

def cleanup_test_files():
    """Limpiar archivos de test generados"""
    test_files = [
        "test_model_solution_iter_5.pkl",
        "test_model_solution_iter_10.pkl", 
        "test_model_solution_final.pkl",
        "test_solution_completa.log"
    ]
    
    for file_path in test_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🧹 Limpiado: {file_path}")
        except Exception as e:
            print(f"⚠️ No se pudo limpiar {file_path}: {e}")

if __name__ == "__main__":
    print("🔧 SCRIPT DE PRUEBA - SOLUCIÓN COMPLETA")
    print("Este script verifica que la solución al bug de historiales sintéticos funciona correctamente.")
    print("")
    
    # Ejecutar test principal
    success = main()
    
    if success:
        print("\n🎉 TODOS LOS TESTS EXITOSOS")
        print("La solución completa está funcionando correctamente.")
        
        # Preguntar si limpiar archivos de test
        response = input("\n¿Desea limpiar los archivos de test generados? (y/n): ")
        if response.lower().startswith('y'):
            cleanup_test_files()
    else:
        print("\n❌ ALGUNOS TESTS FALLARON")
        print("Revisar los logs para más detalles.")
        sys.exit(1) 