#!/usr/bin/env python3
"""
🚀 ENTRENAMIENTO SUPER-HUMANO OPTIMIZADO
Script para entrenar el mejor modelo de poker CFR posible
"""

import sys
import os
import time
import logging

# Setup logging optimizado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_superhuman.log')
    ]
)
logger = logging.getLogger(__name__)

# Add path
sys.path.append('.')

try:
    from poker_bot.core.trainer import (
        SuperHumanTrainerConfig, 
        PokerTrainer,
        create_super_human_trainer
    )
    logger.info("✅ Módulos importados correctamente")
except ImportError as e:
    logger.error(f"❌ Error importando módulos: {e}")
    sys.exit(1)

def run_superhuman_training():
    """
    Ejecuta entrenamiento super-humano optimizado
    """
    logger.info("\n" + "="*80)
    logger.info("🚀 INICIANDO ENTRENAMIENTO SUPER-HUMANO CFR")
    logger.info("="*80)
    logger.info("🎯 OBJETIVO: Crear un bot de poker de nivel élite")
    logger.info("🧠 SISTEMA: CFR con historiales reales + JAX optimizado")
    logger.info("🏆 META: Poker IQ 60+ en 100 iteraciones")
    logger.info("="*80)
    
    # Configuración super-humana optimizada
    config = SuperHumanTrainerConfig()
    
    # CUSTOMIZACIÓN PARA MÁXIMO RENDIMIENTO
    config.batch_size = 256              # Más muestras por iteración
    config.max_iterations = 200          # Entrenamiento sólido 
    config.save_interval = 25            # Guardar frecuente
    config.learning_rate = 0.015         # Learning rate optimizado
    
    # Factores de awareness mejorados
    config.position_awareness_factor = 0.4   # Awareness fuerte
    config.suited_awareness_factor = 0.3     # Suited recognition
    config.pot_odds_factor = 0.25           # Pot odds consideration
    
    # Thresholds calibrados para poker avanzado
    config.strong_hand_threshold = 3800     # Solo verdaderas premium
    config.weak_hand_threshold = 1400       # Threshold estricto
    config.bluff_threshold = 700            # Bluffs selectivos
    
    logger.info("⚙️  CONFIGURACIÓN SUPER-HUMANA:")
    logger.info(f"   - Batch size: {config.batch_size}")
    logger.info(f"   - Max iterations: {config.max_iterations}")
    logger.info(f"   - Learning rate: {config.learning_rate}")
    logger.info(f"   - Position awareness: {config.position_awareness_factor}")
    logger.info(f"   - Suited awareness: {config.suited_awareness_factor}")
    logger.info(f"   - Strong hand threshold: {config.strong_hand_threshold}")
    
    # Crear trainer
    trainer = PokerTrainer(config)
    
    # PATHS para modelos
    base_path = "models/superhuman_cfr"
    os.makedirs("models", exist_ok=True)
    
    logger.info(f"\n💾 PATHS DE GUARDADO:")
    logger.info(f"   - Base path: {base_path}")
    logger.info(f"   - Checkpoints cada: {config.save_interval} iteraciones")
    
    # ENTRENAMIENTO
    start_time = time.time()
    
    try:
        logger.info("\n🚀 INICIANDO ENTRENAMIENTO...")
        trainer.train(
            num_iterations=config.max_iterations,
            save_path=base_path,
            save_interval=config.save_interval,
            snapshot_iterations=[50, 100, 150, 200]  # Snapshots frecuentes
        )
        
        total_time = time.time() - start_time
        
        logger.info("\n" + "="*80)
        logger.info("🎉 ENTRENAMIENTO SUPER-HUMANO COMPLETADO")
        logger.info("="*80)
        logger.info(f"⏱️  Tiempo total: {total_time:.1f}s ({total_time/60:.1f} min)")
        logger.info(f"🚀 Velocidad: {config.max_iterations/total_time:.2f} iter/s")
        logger.info(f"🎯 Hands procesadas: ~{config.max_iterations * config.batch_size * 30:,}")
        
        # Modelo final
        final_model = f"{base_path}_final.pkl"
        logger.info(f"\n🏆 MODELO FINAL GUARDADO: {final_model}")
        
        # Verificar tamaño
        if os.path.exists(final_model):
            size_mb = os.path.getsize(final_model) / 1024 / 1024
            logger.info(f"📊 Tamaño del modelo: {size_mb:.1f} MB")
            
            if size_mb > 2.5:
                logger.info("✅ Modelo grande = Mayor exploración de info sets")
            else:
                logger.info("ℹ️  Modelo compacto = Eficiente para esta fase")
        
        logger.info("\n🎯 PRÓXIMOS PASOS:")
        logger.info("   1. Revisar logs para Poker IQ progression")
        logger.info("   2. Testear modelo contra oponentes")
        logger.info("   3. Si IQ>60: Listo para producción")
        logger.info("   4. Si IQ<60: Aumentar iteraciones o ajustar config")
        logger.info("="*80)
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  ENTRENAMIENTO INTERRUMPIDO POR USUARIO")
        logger.info("💾 Checkpoints guardados están disponibles")
        
    except Exception as e:
        logger.error(f"\n💥 ERROR DURANTE ENTRENAMIENTO: {e}")
        import traceback
        logger.error(f"Traceback completo:\n{traceback.format_exc()}")
        raise

def quick_validation_test():
    """
    Test rápido para verificar que todo funciona antes del entrenamiento largo
    """
    logger.info("\n🔍 EJECUTANDO VALIDACIÓN PRE-ENTRENAMIENTO...")
    
    # Test config simple
    config = SuperHumanTrainerConfig()
    config.batch_size = 64
    config.max_iterations = 5  # Solo 5 iteraciones para test
    config.save_interval = 5
    
    trainer = PokerTrainer(config)
    
    # Test path
    test_path = "models/test_validation"
    os.makedirs("models", exist_ok=True)
    
    try:
        trainer.train(
            num_iterations=5,
            save_path=test_path,
            save_interval=5
        )
        
        logger.info("✅ VALIDACIÓN EXITOSA - Sistema listo para entrenamiento largo")
        return True
        
    except Exception as e:
        logger.error(f"❌ VALIDACIÓN FALLIDA: {e}")
        return False

if __name__ == "__main__":
    logger.info("🎮 POKER CFR SUPER-HUMAN TRAINER")
    
    # Verificar argumentos
    import argparse
    parser = argparse.ArgumentParser(description='Entrenamiento super-humano de poker CFR')
    parser.add_argument('--quick-test', action='store_true', 
                       help='Ejecutar test rápido de validación solamente')
    parser.add_argument('--full-train', action='store_true',
                       help='Ejecutar entrenamiento completo')
    
    args = parser.parse_args()
    
    if args.quick_test:
        logger.info("🧪 MODO: Test rápido de validación")
        success = quick_validation_test()
        if success:
            logger.info("\n✅ SISTEMA VALIDADO - Ejecutar con --full-train para entrenamiento completo")
        else:
            logger.error("\n❌ VALIDACIÓN FALLIDA - Revisar errores antes de continuar")
            sys.exit(1)
            
    elif args.full_train:
        logger.info("🚀 MODO: Entrenamiento completo super-humano")
        run_superhuman_training()
        
    else:
        logger.info("📋 USO:")
        logger.info("   python train_super_human_optimized.py --quick-test    # Test rápido")
        logger.info("   python train_super_human_optimized.py --full-train    # Entrenamiento completo")
        logger.info("\n💡 RECOMENDACIÓN: Ejecutar --quick-test primero") 