import logging
from poker_bot.core.trainer import PokerTrainer, TrainerConfig

# Configurar logging básico para ver el progreso
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    # 1. Crear una configuración de prueba
    config = TrainerConfig(
        batch_size=128,
        num_actions=6,  # CAMBIADO: de 3 a 6 para coincidir con el motor elite (FOLD, CHECK, CALL, BET, RAISE, ALL_IN)
        max_info_sets=50000
    )

    # 2. Crear una instancia del entrenador
    trainer = PokerTrainer(config)

    # 3. Llamar al bucle de entrenamiento principal
    trainer.train(
        num_iterations=10,
        save_path='test_model',
        save_interval=10
    )

    print("\n✅✅✅ PRUEBA DE ENTRENAMIENTO COMPLETA Y EXITOSA ✅✅✅")