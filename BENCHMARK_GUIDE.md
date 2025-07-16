# 🏁 BENCHMARK Y ENTRENAMIENTO - Mercedes-Benz

## 🚀 Guía Rápida para vast.ai

### 1️⃣ **Primer paso: Medir velocidad**

```bash
# Test rápido (2-3 minutos)
python benchmark_mccfr_speed.py quick

# Benchmark completo (10-15 minutos) 
python benchmark_mccfr_speed.py
```

**Output esperado:**
```
⚡ Velocidad: X.XX it/s
⏰ Estimaciones:
   🎯 1K iter: XX minutos/horas
   🎯 5K iter: XX horas  
   🎯 10K iter: XX horas
```

### 2️⃣ **Segundo paso: Entrenamiento largo**

```bash
# Nivel profesional (1K iter, ~1-2 horas)
python launch_training.py professional

# Nivel elite (5K iter, ~5-8 horas)  
python launch_training.py elite

# Nivel super-humano (10K iter, ~10-15 horas)
python launch_training.py superhuman

# Custom (iteraciones específicas)
python launch_training.py custom 2000
```

## 📊 **Velocidades Esperadas**

| Hardware | Velocidad típica | 1K iter | 5K iter | 10K iter |
|----------|------------------|---------|---------|----------|
| RTX 4090 | 2-3 it/s | 5-8 min | 30-40 min | 1-1.5 h |
| RTX 3080 | 1-2 it/s | 8-15 min | 45-80 min | 1.5-3 h |
| RTX 2080 | 0.5-1 it/s | 15-30 min | 1.5-3 h | 3-6 h |

## 🎯 **Recomendaciones por GPU**

### **High-end (RTX 4090, A100)**
```bash
python launch_training.py superhuman  # 10K iter
```

### **Mid-range (RTX 3080, 3090)**  
```bash
python launch_training.py elite       # 5K iter
```

### **Budget (RTX 2080, 3070)**
```bash
python launch_training.py professional # 1K iter
```

## 🏆 **Archivos Generados**

Después del entrenamiento tendrás:
```
mccfr_elite_20241205_143022_final.pkl    # Modelo final
mccfr_elite_20241205_143022_iter_250.pkl # Checkpoints
mccfr_elite_20241205_143022_iter_500.pkl
...
```

## 🔍 **Monitoring Durante Entrenamiento**

El sistema muestra:
- ✅ Progreso cada 10% 
- 📊 Info sets entrenados
- ⏰ Tiempo estimado restante
- 💾 Checkpoints automáticos

## ⚠️ **Si algo sale mal**

1. **Out of memory**: Usa configuración "fast" o reduce batch size
2. **Muy lento**: Verifica que tienes GPU habilitada 
3. **Interrupción**: Los checkpoints te permiten continuar

## 💡 **Tips para vast.ai**

1. **Elegir instancia**: RTX 3080+ recomendado
2. **Storage**: Al menos 10GB para modelos grandes  
3. **Tiempo**: Reservar 2x el tiempo estimado por seguridad
4. **Monitoreo**: Revisar logs cada hora

## 🎉 **¿Listo?**

```bash
# 1. Medir velocidad
python benchmark_mccfr_speed.py quick

# 2. Decidir nivel según velocidad
python launch_training.py [professional|elite|superhuman]
```

¡A entrenar! 🚀 