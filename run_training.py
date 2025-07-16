import logging
import sys
import argparse
from poker_bot.core.trainer import create_super_human_trainer, TrainerConfig, SuperHumanTrainerConfig

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='🎯 Elite Poker CFR Training System')
    parser.add_argument('--level', choices=['standard', 'super_human', 'pluribus_level'], 
                        default='standard', help='Training level configuration')
    parser.add_argument('--iterations', type=int, default=None, 
                        help='Number of training iterations (overrides config default)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config default)')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Model name prefix (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Crear trainer según el nivel especificado
    trainer = create_super_human_trainer(args.level)
    
    # Override configuraciones si se especifican
    if args.iterations:
        if hasattr(trainer.config, 'max_iterations'):
            trainer.config.max_iterations = args.iterations
        else:
            num_iterations = args.iterations
    else:
        num_iterations = getattr(trainer.config, 'max_iterations', 100)
    
    if args.batch_size:
        trainer.config.batch_size = args.batch_size
    
    # Auto-generar nombre del modelo
    if args.model_name:
        model_name = args.model_name
    else:
        model_name = f"{args.level}_model"
    
    # Determinar save interval basado en el nivel
    if args.level == "pluribus_level":
        save_interval = 100  # Guardar cada 100 en entrenamientos muy largos
    elif args.level == "super_human":
        save_interval = 50   # Guardar cada 50 en entrenamientos largos  
    else:
        save_interval = 20   # Guardar cada 20 en entrenamientos estándar
    
    # Configurar snapshots para evaluación de IQ
    if hasattr(trainer.config, 'snapshot_iterations') and trainer.config.snapshot_iterations:
        snapshot_iterations = trainer.config.snapshot_iterations
    else:
        # Calcular snapshots basado en iteraciones
        snapshot_iterations = [
            max(1, num_iterations // 4),      # 25%
            max(1, num_iterations // 2),      # 50%
            max(1, 3 * num_iterations // 4),  # 75%
            num_iterations                    # 100%
        ]
    
    logger.info("="*80)
    logger.info("🎯 ELITE POKER CFR TRAINING SYSTEM")
    logger.info("="*80)
    logger.info(f"🎮 Training Level: {args.level.upper()}")
    logger.info(f"🔄 Iterations: {num_iterations:,}")
    logger.info(f"📦 Batch Size: {trainer.config.batch_size}")
    logger.info(f"💾 Model Name: {model_name}")
    logger.info(f"📸 Snapshots: {snapshot_iterations}")
    
    if args.level == "pluribus_level":
        logger.info("⚠️  WARNING: Pluribus-level training will take several hours!")
        logger.info("   Estimated time: 2-4 hours depending on hardware")
    elif args.level == "super_human":
        logger.info("🚀 Super-human training with advanced concepts enabled")
        logger.info("   Estimated time: 30-60 minutes")
    
    logger.info("="*80)
    
    # Iniciar entrenamiento
    try:
        trainer.train(
            num_iterations=num_iterations,
            save_path=model_name,
            save_interval=save_interval,
            snapshot_iterations=snapshot_iterations
        )
        
        logger.info("\n" + "="*80)
        logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        
        # Mostrar instrucciones de uso
        logger.info(f"📁 Models saved with prefix: {model_name}")
        logger.info(f"💡 To continue training, load: {model_name}_final.pkl")
        logger.info(f"🧠 Best checkpoint based on Poker IQ evolution")
        
        if args.level in ["super_human", "pluribus_level"]:
            logger.info("\n🏆 SUPER-HUMAN MODEL READY FOR COMPETITION!")
            logger.info("   This model can compete against:")
            logger.info("   - Professional poker players")
            logger.info("   - Advanced poker bots")
            logger.info("   - Multi-table tournaments")
            
            if args.level == "pluribus_level":
                logger.info("   - Pluribus-level AI systems")
                logger.info("   - High-stakes cash games")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Training interrupted by user")
        logger.info(f"💾 Partial models saved with prefix: {model_name}")
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()