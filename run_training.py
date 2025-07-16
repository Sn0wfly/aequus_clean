import logging
from poker_bot.core.trainer import PokerTrainer, TrainerConfig

# Configurar logging básico para ver el progreso
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    # 1. Crear una configuración de entrenamiento
    config = TrainerConfig(
        batch_size=128,
        num_actions=6,  # CAMBIADO: de 3 a 6 para coincidir con el motor elite (FOLD, CHECK, CALL, BET, RAISE, ALL_IN)
        max_info_sets=50000
    )

    # 2. Crear una instancia del entrenador
    trainer = PokerTrainer(config)

    # 3. Llamar al bucle de entrenamiento principal con snapshots específicos
    trainer.train(
        num_iterations=100,  # CAMBIADO: de 10 a 100 para ver evolución real
        save_path='intelligent_model',
        save_interval=20,
        snapshot_iterations=[33, 66, 100]  # Snapshots en 33%, 66% y 100%
    )

    print("\n" + "="*80)
    print("🎉 ENTRENAMIENTO CON EVALUACIÓN DE INTELIGENCIA COMPLETADO")
    print("="*80)
    print("📊 Revisa el resumen de evolución arriba para ver cómo mejoró el Poker IQ")
    print("💾 Modelos guardados: intelligent_model_iter_X.pkl e intelligent_model_final.pkl")
    print("🚀 ¡El bot ahora tiene una estrategia entrenada con motor elite + phevaluator!")
    print("="*80 + "\n")