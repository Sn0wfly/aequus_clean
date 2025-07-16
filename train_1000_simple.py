#!/usr/bin/env python3
"""
ENTRENAMIENTO SIMPLE: 1000 iteraciones enfocado en velocidad y eficiencia.

Solo entrena, guarda checkpoints, y muestra progreso. 
Evaluación se hace después con los .pkl guardados.
"""

import logging
import os
import time
from datetime import datetime
from poker_bot.core.trainer import PokerTrainer, TrainerConfig

# Logging simple y claro
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def train_simple_1000():
    """
    Entrenamiento simple y eficiente de 1000 iteraciones
    """
    print("🚀 ENTRENAMIENTO SIMPLE - 1000 ITERACIONES")
    print("="*50)
    
    # Crear directorio para modelos
    os.makedirs("models", exist_ok=True)
    
    # Configuración optimizada para velocidad
    config = TrainerConfig()
    config.batch_size = 128
    config.max_info_sets = 50_000
    
    print(f"📊 Configuración:")
    print(f"   - Iteraciones: 1000")
    print(f"   - Batch size: {config.batch_size}")
    print(f"   - Checkpoints cada: 200 iteraciones")
    print()
    
    # Crear trainer
    trainer = PokerTrainer(config)
    
    # Path para guardar modelos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"models/poker_bot_{timestamp}"
    
    print(f"💾 Modelos se guardarán como: {save_path}_iter_XXX.pkl")
    print()
    
    # ENTRENAMIENTO CON TIMING
    print("⏳ Iniciando entrenamiento...")
    start_time = time.time()
    
    try:
        # Sin snapshots - solo entrenamiento puro
        trainer.train(
            num_iterations=1000,
            save_path=save_path,
            save_interval=200,
            snapshot_iterations=None  # No evaluaciones durante entrenamiento
        )
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 ENTRENAMIENTO COMPLETADO!")
        print(f"⏱️  Tiempo total: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"🚀 Velocidad promedio: {1000/total_time:.2f} iter/s")
        print(f"💾 Modelo final: {save_path}_final.pkl")
        print()
        print(f"📁 Checkpoints guardados:")
        for i in range(200, 1200, 200):
            print(f"   - {save_path}_iter_{i}.pkl")
        
        return save_path
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Entrenamiento interrumpido")
        elapsed = time.time() - start_time
        completed = trainer.iteration if hasattr(trainer, 'iteration') else 0
        print(f"⏱️  Tiempo transcurrido: {elapsed:.1f}s")
        print(f"📊 Iteraciones completadas: {completed}")
        return None
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None

if __name__ == "__main__":
    print("🎯 Poker Bot Training - Versión Simple")
    print()
    
    model_path = train_simple_1000()
    
    if model_path:
        print(f"\n🔧 CÓMO EVALUAR EL MODELO:")
        print(f"   1. Evaluación directa:")
        print(f"      python test_direct_poker_iq.py")
        print()
        print(f"   2. Comparar checkpoints:")
        print(f"      python compare_checkpoints.py {model_path}")
        print()
        print(f"   3. Cargar modelo específico:")
        print(f"      trainer.load_model('{model_path}_final.pkl')")
        
    else:
        print("\n❌ Entrenamiento no completado") 