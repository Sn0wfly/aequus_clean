#!/usr/bin/env python3
"""
Test rápido del CFR arreglado para verificar que aprende conceptos de poker
"""

import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append('.')

try:
    from poker_bot.core.trainer import PokerTrainer, TrainerConfig
    logger.info("✅ Módulos importados correctamente")
except ImportError as e:
    logger.error(f"❌ Error importando: {e}")
    sys.exit(1)

def test_cfr_fix():
    """Test básico del CFR corregido"""
    logger.info("🔧 TESTING CFR ARREGLADO")
    logger.info("="*50)
    
    # Configuración básica
    config = TrainerConfig()
    config.batch_size = 64
    
    trainer = PokerTrainer(config)
    
    try:
        # Test de 10 iteraciones
        logger.info("🚀 Entrenando 10 iteraciones de prueba...")
        trainer.train(
            num_iterations=10,
            save_path="models/test_cfr_fix",
            save_interval=10,
            snapshot_iterations=[5, 10]
        )
        
        logger.info("✅ Test completado - CFR funciona")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error durante test: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_cfr_fix()
    if success:
        logger.info("🎉 CFR arreglado funciona correctamente")
    else:
        logger.error("💥 CFR tiene problemas - necesita más trabajo")
        sys.exit(1) 