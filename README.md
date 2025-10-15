# 🤖 Sistema de Machine Learning Universal

Sistema inteligente de regresión que funciona con **cualquier dataset CSV o Excel**. Automatiza todo el proceso de Machine Learning desde la carga hasta las predicciones.

## 🎯 ¿Qué hace?

- **Carga automática** de archivos CSV y Excel
- **Análisis exploratorio** completo con visualizaciones
- **Limpieza inteligente** de datos (valores nulos, duplicados)
- **Feature engineering** automático (fechas, interacciones)
- **Entrenamiento optimizado** con Ridge Regression
- **Evaluación detallada** con múltiples métricas
- **Exportación** de predicciones en CSV

## 📋 Requisitos

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn openpyxl
```

## 🚀 Uso

```bash
streamlit run ap.py
```

## 📊 Características Principales

### 1. **Carga de Datos**
- Soporta CSV, Excel (.xlsx, .xls)
- Validación automática de formato
- Vista previa interactiva

### 2. **Análisis Exploratorio (EDA)**
- Resumen estadístico completo
- Detección de valores faltantes
- Matrices de correlación
- Distribuciones y box plots

### 3. **Feature Engineering**
- **Temporal**: Extrae año, mes, día, trimestre de fechas
- **Interacciones**: Crea combinaciones entre variables numéricas
- Detección automática de columnas de fecha

### 4. **Preprocesamiento**
- Elimina duplicados automáticamente
- Maneja valores nulos (3 estrategias):
  - **Auto**: Imputación inteligente (recomendado)
  - **Drop**: Elimina filas con nulos
  - **Impute**: Rellena con mediana/moda
- Escalado de variables numéricas
- One-hot encoding para categóricas

### 5. **Entrenamiento**
- **Algoritmo**: Ridge Regression (regularización L2)
- **Validación cruzada**: 5-fold
- **Optimización**: GridSearchCV para alpha óptimo
- **Monitoreo**: Progress bar en tiempo real

### 6. **Evaluación**
- **Métricas**: RMSE, MAE, R², MAPE
- **Visualizaciones**:
  - Predicciones vs Valores Reales
  - Análisis de Residuos
  - Curvas de Aprendizaje
  - Distribución de Errores

### 7. **Exportación**
- Descarga predicciones en CSV
- Incluye errores absolutos y porcentuales

## 🎨 Interfaz

- **Tema oscuro elegante** con gradientes azules
- **7 pasos guiados** para el proceso completo
- **Visualizaciones interactivas** con matplotlib/seaborn
- **Métricas destacadas** con tarjetas visuales
- **Feedback en tiempo real** del proceso

## 📁 Estructura del Código

```
├── DataLoader              # Carga de archivos
├── EDAAnalyzer            # Análisis exploratorio
├── UniversalFeatureEngineer # Feature engineering
├── UniversalDataPreprocessor # Limpieza y preparación
├── ModelTrainer           # Entrenamiento y optimización
└── ModelEvaluator         # Evaluación y visualizaciones
```

## 💡 Ejemplo de Uso

1. **Carga** tu dataset (CSV/Excel)
2. **Explora** las estadísticas y gráficos
3. **Selecciona** características temporales/interacciones (opcional)
4. **Configura** variable objetivo y exclusiones
5. **Entrena** con un clic
6. **Evalúa** resultados y métricas
7. **Descarga** predicciones

## 🔧 Parámetros Configurables

- **Variable objetivo**: Cualquier columna numérica
- **Columnas a excluir**: IDs, nombres, etc.
- **% Test**: 10-40% (default: 20%)
- **Manejo de nulos**: Auto/Drop/Impute
- **Feature engineering**: Temporal, Interacciones

## 📊 Métricas Interpretadas

- **R²**: % de variabilidad explicada (>0.8 = Excelente)
- **RMSE**: Error cuadrático medio
- **MAE**: Error absoluto promedio
- **MAPE**: Error porcentual promedio

## ⚙️ Optimización Automática

El sistema busca el mejor **alpha** (regularización) entre:
- 0.01, 0.1, 1, 10, 100

Usa validación cruzada para evitar overfitting.

## 🎯 Casos de Uso

- Predicción de precios
- Estimación de ventas
- Forecasting de demanda
- Análisis de series temporales
- Cualquier problema de regresión

## ⚠️ Notas Importantes

- Mínimo **50 registros** recomendados (óptimo: 100+)
- Solo para **problemas de regresión** (variables numéricas)
- Elimina columnas con **>70% de valores nulos**
- Usa **regularización L2** para prevenir overfitting

## 🏆 Ventajas

✅ **100% automático** - Sin código manual  
✅ **Adaptable** - Funciona con cualquier dataset  
✅ **Interpretable** - Explicaciones claras  
✅ **Robusto** - Manejo de errores completo  
✅ **Visual** - Gráficos profesionales  

## 📝 Licencia

MIT License - Uso libre para proyectos personales y comerciales.

---

**Desarrollado con**: Python 3.8+ | Streamlit | Scikit-learn | Pandas

**Versión**: 3.0 Universal# Page
