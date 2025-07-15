#!/bin/bash
set -e
echo "🚀 Configurando PokerBot en Vast.ai..."

apt-get update -y && apt-get install -y python3-pip python3-venv build-essential

python3 -m venv /opt/poker_env
source /opt/poker_env/bin/activate

# Asume que el código ya está en /opt/PokerTrainer
cd /opt/PokerTrainer

# Instala el proyecto. Esto instalará JAX, Cython, compilará el .pyx,
# y hará que el comando 'poker-bot' esté disponible.
pip install -e .

echo "✅ ¡Configuración completada!"
echo "Para empezar a entrenar, ejecuta:"
echo "source /opt/poker_env/bin/activate"
echo "poker-bot train --iterations 10000" 