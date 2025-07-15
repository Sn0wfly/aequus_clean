# poker_bot/cli.py

"""
Interfaz de Línea de Comandos para JaxPoker.
"""

import logging
import os
import click
import jax

# Importa las clases y configuraciones principales desde la nueva estructura
from .core.trainer import PokerTrainer, TrainerConfig
from .bot import PokerBot

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@click.group()
@click.version_option(version="1.0.0")
def cli():
    """JaxPoker: Un entrenador de IA de Póker de alto rendimiento."""
    pass

@cli.command()
@click.option('--iterations', default=10000, help='Número de iteraciones de entrenamiento.')
@click.option('--batch-size', default=8192, help='Tamaño del batch para la simulación en GPU.')
@click.option('--save-interval', default=1000, help='Guardar checkpoint cada N iteraciones.')
@click.option('--model-path', default='models/gto_model.pkl', help='Ruta para guardar el modelo entrenado.')
def train(iterations: int, batch_size: int, save_interval: int, model_path: str):
    """Entrena el modelo de IA de Póker usando el entrenador GTO."""
    
    logger.info("🚀 Iniciando el entrenamiento del modelo GTO de JaxPoker...")
    logger.info(f"Dispositivos JAX detectados: {jax.devices()}")
    
    # Crear el directorio de modelos si no existe
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Configurar el entrenador
    config = TrainerConfig(
        batch_size=batch_size,
        learning_rate=0.1,
        temperature=1.0
    )
    
    trainer = PokerTrainer(config)
    
    try:
        # Iniciar entrenamiento
        trainer.train(
            num_iterations=iterations,
            save_path=model_path,
            save_interval=save_interval
        )
        
        logger.info("✅ Entrenamiento completado exitosamente!")
        logger.info(f"Modelo guardado en: {model_path}")
        
    except Exception as e:
        logger.error(f"❌ Error durante el entrenamiento: {e}")
        raise

@cli.command()
@click.option('--model', required=True, help='Ruta al modelo entrenado (.pkl)')
def play(model: str):
    """Carga y prueba el bot entrenado."""
    if not os.path.exists(model):
        logger.error(f"Modelo no encontrado en la ruta: {model}")
        return

    logger.info(f"🤖 Cargando bot con el modelo: {model}")
    
    try:
        bot = PokerBot(model_path=model)
        logger.info("✅ Bot cargado exitosamente!")
        
        # Prueba simple del bot
        test_state = {'player_id': 0, 'hole_cards': [0, 1], 'community_cards': [2, 3, 4]}
        action = bot.get_action(test_state)
        logger.info(f"Acción de prueba: {action}")
        
    except Exception as e:
        logger.error(f"❌ Error al cargar el bot: {e}")

@cli.command()
def evaluate():
    """Evalúa los componentes del sistema."""
    logger.info("🔍 Evaluando componentes del sistema...")
    
    try:
        # Test JAX
        logger.info(f"✅ JAX version: {jax.__version__}")
        logger.info(f"✅ JAX devices: {jax.devices()}")
        
        # Test trainer
        config = TrainerConfig(batch_size=1024)
        trainer = PokerTrainer(config)
        logger.info("✅ Trainer creado exitosamente")
        
        # Test bot (sin modelo)
        logger.info("✅ Componentes básicos funcionando")
        
    except Exception as e:
        logger.error(f"❌ Error en evaluación: {e}")

if __name__ == '__main__':
    cli()