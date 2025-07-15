# 🧹 PROMPT DE LIMPIEZA DE CÓDIGO - Aequus Poker Bot

## 📋 CONTEXTO ACTUAL DEL PROYECTO

### **✅ Sistema Funcional Actual:**
- **Método**: Monte Carlo Counterfactual Regret Minimization (MCCFR)
- **Abstracción**: Pluribus bucketing con 200k+ buckets
- **Entrenamiento**: 83.2M info sets únicos, 65.5M juegos procesados
- **Arquitectura**: GPU-accelerated con CUDA kernels
- **Configuración**: 6-max NLHE, 100BB stacks, Phase 1 optimizado

### **📁 Archivos ACTIVOS y Funcionales:**
1. **main_phase1.py** - Script principal de entrenamiento
2. **test_phase1.py** - Suite de pruebas
3. **compare_models.py** - Comparación de checkpoints
4. **poker_bot/core/trainer.py** - Entrenador definitivo
5. **poker_bot/core/enhanced_eval.py** - Evaluación de manos mejorada
6. **poker_bot/core/icm_modeling.py** - Modelado ICM
7. **poker_bot/core/history_aware_bucketing.py** - Sistema de bucketing
8. **config/phase1_config.yaml** - Configuración optimizada
9. **scripts/deploy_phase1_vastai.sh** - Despliegue a Vast.ai

## 🎯 OBJETIVO DE LIMPIEZA

### **🔍 Tareas de Refactoring:**

#### **1. Eliminar Código Obsoleto:**
- **DEBUG prints** innecesarios en producción
- **Comentarios TODO** viejos o resueltos
- **Variables no utilizadas** o deprecadas
- **Imports sin usar**
- **Funciones legacy** de debugging

#### **2. Mantener Funcionalidad Esencial:**
- **MCCFR training loop** completo
- **GPU acceleration** con CUDA kernels
- **Pluribus bucketing** con 200k+ buckets
- **ICM modeling** para torneos
- **Enhanced evaluation** de manos
- **Checkpoint system** funcional

#### **3. Optimizar Estructura:**
- **Consolidar imports** redundantes
- **Eliminar código de fallback** obsoleto
- **Limpiar configuraciones** deprecadas
- **Remover referencias** a archivos eliminados

## 📝 INSTRUCCIONES ESPECÍFICAS POR ARCHIVO

### **📄 main_phase1.py**
**MANTENER:**
- Configuración de TrainerConfig
- Loop de entrenamiento principal
- Logging estructurado
- Sistema de checkpoints

**ELIMINAR:**
- Comentarios de debug antiguos
- Configuraciones "debug" obsoletas
- Referencias a archivos deprecados

### **📄 poker_bot/core/trainer.py**
**MANTENER:**
- MCCFR training step
- GPU acceleration con CUDA
- Pluribus bucketing system
- Memory management dinámico
- Checkpoint save/load

**ELIMINAR:**
- Código de fallback a CPU
- Comentarios de debugging
- Variables de configuración obsoletas
- Referencias a "super_bot"

### **📄 poker_bot/core/enhanced_eval.py**
**MANTENER:**
- Hand strength calculation
- ICM adjustments
- GPU acceleration

**ELIMINAR:**
- Comentarios de performance testing
- Código de benchmark obsoleto

### **📄 poker_bot/core/icm_modeling.py**
**MANTENER:**
- ICM calculations
- Stack depth adjustments
- Position-based modeling

**ELIMINAR:**
- Debug prints de performance
- Comentarios de implementación vieja

### **📄 config/phase1_config.yaml**
**MANTENER:**
- Batch size: 32768
- Learning rate: 0.05
- Max info sets: 50000
- Pluribus bucketing: True
- ICM modeling: True

**ELIMINAR:**
- Configuraciones de debug
- Parámetros obsoletos

## 🎯 CRITERIOS DE DECISIÓN

### **✅ MANTENER si:**
- Función está siendo usada activamente
- Código es parte del flujo de entrenamiento principal
- Feature está documentado en README
- Es necesario para reproducir resultados

### **❌ ELIMINAR si:**
- Es código de debugging que ya no se necesita
- Comentario se refiere a feature deprecado
- Variable no tiene referencias activas
- Función es duplicada o reemplazada

## 🚀 PROCESO DE LIMPIEZA

1. **Backup primero** - Guardar copia antes de cambios
2. **Buscar patrones** - Identificar código obsoleto
3. **Verificar funcionalidad** - Asegurar que todo sigue funcionando
4. **Testear** - Ejecutar `python test_phase1.py` después de limpiar
5. **Documentar** - Actualizar comentarios relevantes

## 🎯 RESULTADO ESPERADO

**Código limpio que:**
- Solo contiene funcionalidad activa y probada
- Tiene comentarios claros y actuales
- Mantiene 100% compatibilidad con resultados actuales
- Es fácil de mantener y extender
- Está listo para producción

## 📝 VERIFICACIÓN POST-LIMPIEZA

Después de limpiar, ejecutar:
```bash
python test_phase1.py
python main_phase1.py --iterations 10 --save_every 5 --save_path test_clean
```

**Si todo funciona igual, la limpieza fue exitosa!**