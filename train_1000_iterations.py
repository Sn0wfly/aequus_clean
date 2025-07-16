#!/usr/bin/env python3
"""
ENTRENAMIENTO OPTIMIZADO: 1000 iteraciones con evaluaciones y checkpoints.

Este script entrena un bot de poker CFR durante 1000 iteraciones,
guardando checkpoints cada 200 iteraciones y evaluando el progreso.
"""

import logging
import os
from datetime import datetime
from poker_bot.core.trainer import (
    PokerTrainer, TrainerConfig, SuperHumanTrainerConfig,
    evaluate_direct_poker_iq
)

# Configurar logging detallado
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s:%(name)s:%(message)s',
    handlers=[
        logging.StreamHandler(),  # Console
        logging.FileHandler(f'training_1000_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')  # File
    ]
)
logger = logging.getLogger(__name__)

def train_poker_bot_1000():
    """
    Entrena un bot de poker durante 1000 iteraciones con checkpoints y evaluaciones
    """
    logger.info("🚀 INICIANDO ENTRENAMIENTO DE 1000 ITERACIONES")
    logger.info("="*70)
    
    # Crear directorio para modelos si no existe
    os.makedirs("models", exist_ok=True)
    
    # Configuración optimizada para 1000 iteraciones
    config = TrainerConfig()
    config.batch_size = 128        # Buen balance entre velocidad y aprendizaje
    config.max_info_sets = 50_000  # Suficiente para manos complejas
    
    logger.info(f"📊 CONFIGURACIÓN DE ENTRENAMIENTO:")
    logger.info(f"   - Iteraciones total: 1000")
    logger.info(f"   - Checkpoint cada: 200 iteraciones") 
    logger.info(f"   - Batch size: {config.batch_size}")
    logger.info(f"   - Max info sets: {config.max_info_sets:,}")
    
    # Crear trainer
    trainer = PokerTrainer(config)
    
    # Path base para checkpoints
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"models/poker_bot_1000iter_{timestamp}"
    
    logger.info(f"💾 Modelos se guardarán en: {save_path}_iter_XXX.pkl")
    
    # Configurar snapshots para evaluación en puntos clave
    snapshot_iterations = [200, 400, 600, 800, 1000]
    
    logger.info(f"📸 Evaluaciones de Poker IQ en iteraciones: {snapshot_iterations}")
    logger.info("")
    
    # EJECUTAR ENTRENAMIENTO PRINCIPAL
    try:
        trainer.train(
            num_iterations=1000,
            save_path=save_path,
            save_interval=200,  # Guardar cada 200 iteraciones
            snapshot_iterations=snapshot_iterations
        )
        
        logger.info("\n🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
        
        # EVALUACIÓN FINAL DIRECTA
        logger.info("\n🧠 EJECUTANDO EVALUACIÓN FINAL DIRECTA...")
        
        try:
            # Evaluar usando el método directo (más preciso)
            from test_direct_poker_iq import evaluate_direct_poker_iq
            
            # Crear un trainer temporal con el modelo entrenado para evaluación
            final_results = {
                'strategy': trainer.strategy,
                'regrets': trainer.regrets,
                'iteration': trainer.iteration
            }
            
            logger.info("✅ Modelo final guardado y listo para uso")
            
        except Exception as e:
            logger.warning(f"⚠️ No se pudo ejecutar evaluación directa: {e}")
            logger.info("💡 Puedes ejecutar 'python test_direct_poker_iq.py' manualmente")
        
        # RESUMEN FINAL
        logger.info("\n" + "="*70)
        logger.info("🏆 RESUMEN FINAL DEL ENTRENAMIENTO")
        logger.info("="*70)
        logger.info(f"✅ Iteraciones completadas: {trainer.iteration}/1000")
        logger.info(f"💾 Checkpoints guardados: {save_path}_iter_200.pkl a {save_path}_iter_1000.pkl")
        logger.info(f"📈 Evolución completa registrada en logs")
        logger.info(f"🎯 Modelo final: {save_path}_final.pkl")
        
        # Instrucciones para usar el modelo
        logger.info(f"\n📋 CÓMO USAR EL MODELO ENTRENADO:")
        logger.info(f"   1. Cargar: trainer.load_model('{save_path}_final.pkl')")
        logger.info(f"   2. Evaluar: python test_direct_poker_iq.py")
        logger.info(f"   3. Ver evolución: revisar snapshots en logs")
        
        return {
            'success': True,
            'final_model': f'{save_path}_final.pkl',
            'checkpoints': [f'{save_path}_iter_{i}.pkl' for i in range(200, 1200, 200)],
            'iterations_completed': trainer.iteration
        }
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️ ENTRENAMIENTO INTERRUMPIDO POR USUARIO")
        logger.info(f"💾 Último checkpoint disponible en: {save_path}_iter_XXX.pkl")
        return {'success': False, 'reason': 'interrupted'}
        
    except Exception as e:
        logger.error(f"\n❌ ERROR DURANTE ENTRENAMIENTO: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {'success': False, 'reason': str(e)}

def train_super_human_version():
    """
    Versión SUPER-HUMANA con configuración avanzada (opcional)
    """
    logger.info("🏆 INICIANDO ENTRENAMIENTO SUPER-HUMANO")
    logger.info("⚠️  ADVERTENCIA: Esto tomará MÁS TIEMPO pero mejor resultado")
    
    # Configuración super-humana
    config = SuperHumanTrainerConfig()
    config.max_iterations = 1000
    config.batch_size = 256  # Más muestras por iteración
    config.save_interval = 200
    
    trainer = PokerTrainer(config)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"models/superhuman_poker_bot_{timestamp}"
    
    trainer.train(
        num_iterations=1000,
        save_path=save_path,
        save_interval=200,
        snapshot_iterations=[200, 400, 600, 800, 1000]
    )
    
    return save_path

if __name__ == "__main__":
    print("🎯 ENTRENAMIENTO DE POKER BOT - 1000 ITERACIONES")
    print("="*60)
    print("Opciones disponibles:")
    print("1. Entrenamiento estándar (recomendado)")
    print("2. Entrenamiento super-humano (más lento)")
    print()
    
    try:
        choice = input("Selecciona opción (1/2) [1]: ").strip() or "1"
        
        if choice == "2":
            results = train_super_human_version()
        else:
            results = train_poker_bot_1000()
        
        if results.get('success'):
            print(f"\n🎉 ¡ENTRENAMIENTO EXITOSO!")
            print(f"📁 Modelo final: {results['final_model']}")
        else:
            print(f"\n❌ Entrenamiento falló: {results.get('reason', 'Unknown')}")
            
    except KeyboardInterrupt:
        print("\n👋 Saliendo...")
    except Exception as e:
        print(f"\n❌ Error: {e}") 