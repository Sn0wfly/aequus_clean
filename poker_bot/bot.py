# poker_bot/bot.py

"""
🤖 Agente de Póker (Bot Jugable)

Este módulo contiene la clase PokerBot, que carga un modelo GTO entrenado
y lo utiliza para tomar decisiones de juego en tiempo real.
"""

import numpy as np
import pickle
import logging
import hashlib
from typing import Dict, Any

# Importamos desde la nueva estructura 'core'
# ¡OJO! Ya no importamos 'engine.py'
from .core.trainer import PokerTrainer, TrainerConfig 

logger = logging.getLogger(__name__)

class PokerBot:
    """
    Un agente de IA que juega al póker utilizando una estrategia GTO pre-entrenada.
    """
    def __init__(self, model_path: str):
        """
        Inicializa el bot cargando el modelo entrenado.

        Args:
            model_path: Ruta al archivo .pkl del modelo GTO.
        """
        self.model_path = model_path
        self.trainer_state: Dict[str, Any] = {}
        self.q_values: Dict[str, np.ndarray] = {}
        self.strategies: Dict[str, np.ndarray] = {}
        self.config: TrainerConfig = None

        logger.info(f"🤖 Cargando modelo GTO desde {model_path}...")
        self._load_model()

    def _load_model(self):
        """Carga los datos del modelo GTO desde el archivo .pkl."""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Extrae los componentes principales del modelo
            self.q_values = model_data.get('q_values', {})
            self.strategies = model_data.get('strategies', {})
            # Renombramos la config para que coincida con lo que guardamos
            trainer_config_data = model_data.get('config', {})
            self.config = TrainerConfig(**trainer_config_data)

            # Guardamos todos los hashes para poder buscar estrategias
            self.info_set_hashes = model_data.get('info_set_hashes', {})

            logger.info(f"✅ Modelo cargado con {len(self.info_set_hashes):,} estrategias únicas.")
            
        except FileNotFoundError:
            logger.error(f"❌ Error: Archivo de modelo no encontrado en '{self.model_path}'")
            raise
        except Exception as e:
            logger.error(f"❌ Error al cargar el modelo: {e}")
            raise

    def _get_info_set_hash(self, game_state_for_hashing: dict) -> str:
        """
        Calcula el hash para un estado de juego dado.
        Esta función DEBE ser idéntica a la usada durante el entrenamiento.
        """
        # Aquí crearías la tupla de componentes a partir de un estado de juego
        # y la hashearías, igual que en el trainer.
        # Por ahora, es un placeholder.
        components = (
            game_state_for_hashing.get('player_id'),
            game_state_for_hashing.get('hole_cards').tobytes(),
            game_state_for_hashing.get('community_cards').tobytes(),
            # ... etc.
        )
        return hashlib.md5(repr(components).encode()).hexdigest()

    def get_action(self, current_game_state: dict) -> str:
        """
        La función principal del bot. Dado un estado de juego, devuelve la mejor acción.
        """
        # 1. Convertir el estado del juego actual a un hash
        # info_hash = self._get_info_set_hash(current_game_state)
        # 
        # 2. Buscar el índice correspondiente en nuestro diccionario de hashes
        # index = self.info_set_hashes.get(info_hash)
        #
        # 3. Si se encuentra el estado, obtener la estrategia GTO
        # if index is not None:
        #    strategy_probabilities = self.strategies[index]
        #    # Elegir una acción basada en las probabilidades (ej. la más probable)
        #    action_index = np.argmax(strategy_probabilities)
        #    action = ["FOLD", "CALL", "BET", "RAISE"][action_index]
        #    return action
        # else:
        #    # Si el estado es desconocido (no se vio en el entrenamiento),
        #    # usar una acción por defecto.
        #    return "CHECK"
        
        # Por ahora, como la lógica de juego no está implementada, devolvemos un placeholder.
        logger.warning("Lógica de juego no implementada. Usando acción por defecto.")
        return "CHECK" 