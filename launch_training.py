#!/usr/bin/env python3
"""
🚀 LAUNCH TRAINING - MCCFR Mercedes-Benz
Script para lanzar entrenamientos largos de manera fácil

Uso:
  python launch_training.py professional  # 1K iteraciones
  python launch_training.py elite         # 5K iteraciones  
  python launch_training.py superhuman    # 10K iteraciones
  python launch_training.py custom 2000   # Iteraciones custom
"""

import sys
import time
from trainer_mccfr_real import create_mccfr_trainer

def launch_training(training_type="professional", custom_iterations=None):
    """🚀 Lanza entrenamiento largo con configuración optimizada"""
    
    # Configuraciones predefinidas
    training_configs = {
        "professional": {
            "iterations": 1000,
            "config": "standard",
            "description": "Entrenamiento Profesional",
            "expected_level": "Competitivo vs humanos buenos"
        },
        "elite": {
            "iterations": 5000, 
            "config": "standard",
            "description": "Entrenamiento Elite",
            "expected_level": "Nivel profesional, competitivo vs bots comerciales"
        },
        "superhuman": {
            "iterations": 10000,
            "config": "large", 
            "description": "Entrenamiento Super-Humano",
            "expected_level": "Nivel expert, competitivo vs mejores bots públicos"
        },
        "custom": {
            "iterations": custom_iterations or 1000,
            "config": "standard",
            "description": f"Entrenamiento Custom ({custom_iterations or 1000} iter)",
            "expected_level": "Nivel según iteraciones"
        }
    }
    
    if training_type not in training_configs:
        print(f"❌ Tipo de entrenamiento inválido: {training_type}")
        print(f"✅ Opciones válidas: {list(training_configs.keys())}")
        return
    
    config = training_configs[training_type]
    
    print("🚀 LAUNCH TRAINING - MCCFR Mercedes-Benz")
    print("="*60)
    print(f"🎯 {config['description']}")
    print(f"📊 Configuración: {config['config']}")  
    print(f"🔢 Iteraciones: {config['iterations']:,}")
    print(f"🏆 Nivel esperado: {config['expected_level']}")
    print("="*60)
    
    # Confirmación
    print(f"\n⚠️  CONFIRMACIÓN:")
    print(f"   Este entrenamiento puede tardar varias horas.")
    print(f"   ¿Deseas continuar? (y/N): ", end="")
    
    try:
        response = input().strip().lower()
        if response not in ['y', 'yes', 'sí', 'si']:
            print("❌ Entrenamiento cancelado por el usuario")
            return
    except KeyboardInterrupt:
        print("\n❌ Entrenamiento cancelado")
        return
    
    # Crear trainer y comenzar
    print(f"\n🏗️  Creando trainer ({config['config']})...")
    trainer = create_mccfr_trainer(config['config'])
    
    # Generar nombre del modelo
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_name = f"mccfr_{training_type}_{timestamp}"
    
    print(f"💾 Modelo se guardará como: {model_name}")
    print(f"🚀 Iniciando entrenamiento...")
    print("="*60)
    
    # ¡ENTRENAR!
    try:
        start_time = time.time()
        
        trainer.train(
            num_iterations=config['iterations'],
            save_path=model_name,
            save_interval=max(50, config['iterations'] // 20)  # Guardar cada 5% del progreso
        )
        
        total_time = time.time() - start_time
        hours = total_time / 3600
        
        print("="*60)
        print("🎉 ¡ENTRENAMIENTO COMPLETADO!")
        print(f"⏰ Tiempo total: {hours:.1f} horas")
        print(f"💾 Modelo guardado: {model_name}_final.pkl")
        print(f"🏆 Nivel alcanzado: {config['expected_level']}")
        
        # Análisis final
        print(f"\n📊 Analizando modelo final...")
        final_results = trainer.analyze_training_progress()
        
        print(f"\n🏆 ESTADÍSTICAS FINALES:")
        print(f"   - Info sets entrenados: {final_results['trained_info_sets']:,}")
        print(f"   - Diferenciación rica: {final_results['rich_differentiation']:.1%}")
        print(f"   - Varianza estrategia: {final_results['strategy_variance']:.6f}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Entrenamiento interrumpido por el usuario")
        print(f"💾 Progreso guardado en checkpoints")
        
    except Exception as e:
        print(f"\n❌ Error durante entrenamiento: {e}")
        print(f"💾 Revisa los checkpoints para recuperar progreso")

def show_help():
    """📖 Muestra información de uso"""
    print("🚀 LAUNCH TRAINING - MCCFR Mercedes-Benz")
    print("="*60)
    print("📖 USO:")
    print("  python launch_training.py professional   # 1K iter (~1-2 horas)")
    print("  python launch_training.py elite          # 5K iter (~5-8 horas)")  
    print("  python launch_training.py superhuman     # 10K iter (~10-15 horas)")
    print("  python launch_training.py custom 2000    # 2K iter custom")
    print("")
    print("🎯 NIVELES:")
    print("  professional → Competitivo vs humanos buenos")
    print("  elite        → Nivel profesional")
    print("  superhuman   → Competitivo vs mejores bots")
    print("")
    print("💡 TIP: Ejecuta 'python benchmark_mccfr_speed.py quick' primero")
    print("       para estimar tiempos en tu hardware")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        show_help()
    elif len(sys.argv) == 2:
        training_type = sys.argv[1]
        if training_type in ["help", "-h", "--help"]:
            show_help()
        else:
            launch_training(training_type)
    elif len(sys.argv) == 3 and sys.argv[1] == "custom":
        try:
            custom_iters = int(sys.argv[2])
            launch_training("custom", custom_iters)
        except ValueError:
            print("❌ Número de iteraciones inválido")
            show_help()
    else:
        show_help() 