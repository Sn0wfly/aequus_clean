#!/usr/bin/env python3
"""
ENTRENAMIENTO CORREGIDO: Script que usa el CFR arreglado.

Este script entrena con el bug de regrets arreglado y monitorea
el progreso correctamente.
"""

import logging
from poker_bot.core.trainer import PokerTrainer, TrainerConfig, create_super_human_trainer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def train_basic_model():
    """Entrenamiento básico para verificar que el CFR funciona"""
    print("🚀 ENTRENAMIENTO BÁSICO - CFR Arreglado")
    print("="*60)
    
    config = TrainerConfig()
    config.batch_size = 128
    trainer = PokerTrainer(config)
    
    # Entrenamiento con validaciones
    trainer.train(
        num_iterations=100,
        save_path='model_basic_fixed',
        save_interval=50,
        snapshot_iterations=[25, 50, 75, 100]  # Snapshots para monitorear progreso
    )
    
    print("✅ Entrenamiento básico completado")
    return trainer

def train_intermediate_model():
    """Entrenamiento intermedio para desarrollar conceptos de poker"""
    print("🎯 ENTRENAMIENTO INTERMEDIO - Desarrollo de Conceptos")
    print("="*60)
    
    config = TrainerConfig()
    config.batch_size = 256  # Más coverage
    trainer = PokerTrainer(config)
    
    # Entrenamiento más largo para conceptos
    trainer.train(
        num_iterations=500,
        save_path='model_intermediate_fixed',
        save_interval=100,
        snapshot_iterations=[100, 200, 300, 400, 500]
    )
    
    print("✅ Entrenamiento intermedio completado")
    return trainer

def train_advanced_model():
    """Entrenamiento avanzado con configuración super-humana"""
    print("🏆 ENTRENAMIENTO AVANZADO - Nivel Super-Humano")
    print("="*60)
    
    # Usar configuración super-humana
    trainer = create_super_human_trainer("super_human")
    
    # Entrenamiento extenso
    trainer.train(
        num_iterations=1000,
        save_path='model_superhuman_fixed', 
        save_interval=200,
        snapshot_iterations=[200, 400, 600, 800, 1000]
    )
    
    print("✅ Entrenamiento avanzado completado")
    return trainer

def quick_verification_training():
    """Entrenamiento rápido solo para verificar que no hay bugs"""
    print("⚡ VERIFICACIÓN RÁPIDA - Solo para probar que funciona")
    print("="*60)
    
    config = TrainerConfig()
    config.batch_size = 64
    trainer = PokerTrainer(config)
    
    # Solo 20 iteraciones para verificación rápida
    trainer.train(
        num_iterations=20,
        save_path='model_verification',
        save_interval=20,
        snapshot_iterations=[]  # Sin snapshots para velocidad
    )
    
    print("✅ Verificación completada")
    
    # Quick analysis
    from poker_bot.core.trainer import validate_training_data_integrity
    import jax
    
    print("\n🔍 Análisis rápido post-entrenamiento:")
    validation_results = validate_training_data_integrity(
        trainer.strategy, 
        jax.random.PRNGKey(999), 
        verbose=False
    )
    
    print(f"   - Historiales reales: {'✅' if validation_results['real_histories_detected'] else '❌'}")
    print(f"   - Info sets consistentes: {'✅' if validation_results['info_set_consistency'] else '❌'}")
    print(f"   - Hand strength variable: {'✅' if validation_results['hand_strength_variation'] else '❌'}")
    print(f"   - Bugs críticos: {len(validation_results['critical_bugs'])}")
    
    return trainer

def main():
    """Script principal"""
    print("🎮 POKER AI - ENTRENAMIENTO CORREGIDO")
    print("="*80)
    print("Bugs arreglados:")
    print("  ✅ Acumulación de regrets para todos los jugadores")
    print("  ✅ Info sets usando datos reales del entrenamiento")
    print("  ✅ Validación completa del sistema")
    print("="*80)
    
    # Menú simple
    print("\nSelecciona tipo de entrenamiento:")
    print("1. ⚡ Verificación rápida (20 iter, ~1 min)")
    print("2. 🚀 Básico (100 iter, ~5 min)")
    print("3. 🎯 Intermedio (500 iter, ~20 min)")
    print("4. 🏆 Avanzado (1000 iter, ~45 min)")
    
    # Por defecto correr verificación rápida si se ejecuta directamente
    choice = "1"  # Cambiar este valor para diferentes entrenamientos
    
    try:
        if choice == "1":
            trainer = quick_verification_training()
        elif choice == "2":
            trainer = train_basic_model()
        elif choice == "3":
            trainer = train_intermediate_model()
        elif choice == "4":
            trainer = train_advanced_model()
        else:
            print("❌ Opción inválida")
            return
        
        print(f"\n🎉 ENTRENAMIENTO EXITOSO!")
        print(f"   - Iteraciones completadas: {trainer.iteration}")
        print(f"   - Modelo guardado con éxito")
        print(f"   - Listo para evaluación con test_poker_concepts_fixed.py")
        
    except Exception as e:
        print(f"\n❌ ERROR EN ENTRENAMIENTO: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n🔧 DEBUGGING:")
        print(f"   - Verificar que poker_bot/core/trainer.py tiene el fix de regrets")
        print(f"   - Verificar que no hay conflictos de imports")
        print(f"   - Ejecutar test_poker_concepts_fixed.py para más diagnósticos")

if __name__ == "__main__":
    main() 