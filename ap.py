import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================
# 🎨 CONFIGURACIÓN DE TEMA Y ESTILOS
# =============================================
def set_custom_style():
    st.markdown("""
        <style>
        /* Fondo general */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        
        /* Título principal */
        .main-title {
            font-size: 3.5em;
            font-weight: 800;
            background: linear-gradient(120deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 10px;
        }
        
        /* Subtítulo */
        .subtitle {
            text-align: center;
            font-size: 1.4em;
            color: #2d3748;
            margin-bottom: 30px;
            font-weight: 600;
        }
        
        /* Tarjetas personalizadas */
        .custom-card {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.15);
            margin: 20px 0;
            border: 2px solid #e2e8f0;
        }
        
        .custom-card h3, .custom-card h4 {
            color: #1a202c;
            margin-bottom: 15px;
            font-weight: 700;
        }
        
        .custom-card p, .custom-card li {
            color: #2d3748;
            line-height: 1.8;
            font-size: 1.05em;
        }
        
        .custom-card strong, .custom-card b {
            color: #1a202c;
            font-weight: 700;
        }
        
        /* Info Box destacado */
        .info-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
            margin: 20px 0;
        }
        
        .info-box h3, .info-box h4 {
            color: white;
            font-weight: 700;
            margin-bottom: 15px;
        }
        
        .info-box p, .info-box li {
            color: white;
            line-height: 1.8;
            font-size: 1.05em;
        }
        
        .info-box strong, .info-box b {
            color: #ffd700;
        }
        
        /* Métricas mejoradas */
        .metric-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 6px 15px rgba(102, 126, 234, 0.4);
        }
        
        .metric-value {
            font-size: 2.2em;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .metric-label {
            font-size: 1em;
            opacity: 0.95;
            font-weight: 600;
        }
        
        /* Badges */
        .badge {
            display: inline-block;
            padding: 8px 18px;
            border-radius: 25px;
            font-size: 0.9em;
            font-weight: 700;
            margin: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        
        .badge-success {
            background: #48bb78;
            color: white;
        }
        
        .badge-info {
            background: #4299e1;
            color: white;
        }
        
        .badge-warning {
            background: #ed8936;
            color: white;
        }
        
        /* Botones */
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            font-weight: 700;
            font-size: 1.1em;
            transition: all 0.3s ease;
            box-shadow: 0 4px 10px rgba(102, 126, 234, 0.3);
        }
        
        .stButton>button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
        }
        
        /* Alertas personalizadas */
        .custom-success {
            background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
            color: #1a4d2e;
            padding: 20px;
            border-radius: 12px;
            border-left: 6px solid #48bb78;
            font-weight: 700;
            box-shadow: 0 5px 12px rgba(72, 187, 120, 0.3);
            font-size: 1.05em;
        }
        
        .custom-info {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            color: #1a365d;
            padding: 20px;
            border-radius: 12px;
            border-left: 6px solid #4299e1;
            font-weight: 700;
            box-shadow: 0 5px 12px rgba(66, 153, 225, 0.3);
            font-size: 1.05em;
        }
        
        .custom-warning {
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            color: #7c2d12;
            padding: 20px;
            border-radius: 12px;
            border-left: 6px solid #ed8936;
            font-weight: 700;
            box-shadow: 0 5px 12px rgba(237, 137, 54, 0.3);
            font-size: 1.05em;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        }
        
        [data-testid="stSidebar"] * {
            color: white !important;
            font-weight: 600;
        }
        
        [data-testid="stSidebar"] .stMarkdown a {
            color: #ffd700 !important;
            text-decoration: none;
            font-weight: 700;
        }
        
        /* File uploader */
        [data-testid="stFileUploader"] {
            background: white;
            padding: 20px;
            border-radius: 12px;
            border: 2px dashed #667eea;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        
        [data-testid="stFileUploader"] label {
            color: #2d3748 !important;
            font-weight: 700 !important;
            font-size: 1.1em !important;
        }
        
        /* Streamlit elements styling */
        .stDataFrame {
            background: white;
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.15);
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background: white !important;
            color: #1a202c !important;
            font-weight: 700 !important;
            font-size: 1.2em !important;
            border-radius: 10px;
            box-shadow: 0 3px 8px rgba(0, 0, 0, 0.1);
            padding: 15px !important;
        }
        
        .streamlit-expanderHeader:hover {
            background: #f7fafc !important;
            color: #667eea !important;
        }
        
        .streamlit-expanderHeader p {
            color: #1a202c !important;
            font-weight: 700 !important;
        }
        
        /* Text elements */
        .stMarkdown p, .stMarkdown li {
            color: #2d3748;
            font-size: 1.05em;
        }
        
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
            color: #1a202c;
            font-weight: 700;
        }
        
        /* Metric labels */
        [data-testid="stMetricLabel"] {
            color: #2d3748 !important;
            font-weight: 700 !important;
            font-size: 1.1em !important;
        }
        
        [data-testid="stMetricValue"] {
            color: #1a202c !important;
            font-weight: 800 !important;
        }
        
        /* Selectbox y Slider */
        .stSelectbox label, .stSlider label, .stMultiSelect label {
            color: #2d3748 !important;
            font-weight: 700 !important;
            font-size: 1.1em !important;
        }
        
        /* Checkbox */
        .stCheckbox label {
            color: #2d3748 !important;
            font-weight: 700 !important;
            font-size: 1.1em !important;
        }
        
        /* Section headers */
        .section-header {
            color: #1a202c;
            font-size: 1.8em;
            font-weight: 800;
            margin: 30px 0 20px 0;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }
        </style>
    """, unsafe_allow_html=True)

# =============================================
# 🎯 INFORMACIÓN DEL PROYECTO
# =============================================
PROJECT_INFO = """
<div class="info-box">

### 🎯 **Sistema de Machine Learning Universal**

**📊 Características Principales**  
Sistema adaptable que funciona con **cualquier dataset CSV o Excel** para problemas de regresión.

**💡 Capacidades:**
- 🔄 Carga automática de cualquier formato (CSV, Excel)
- 🎯 Selección flexible de variable objetivo
- 🔧 Detección automática de tipos de datos
- 📊 Feature engineering inteligente
- 🤖 Optimización automática de hiperparámetros

**🤖 Algoritmo: Ridge Regression**
- ✅ Maneja multicolinealidad efectivamente
- ✅ Previene overfitting con regularización L2
- ✅ Coeficientes interpretables
- ✅ Entrenamiento eficiente
- ✅ Funciona con datos numéricos y categóricos

</div>
"""

# =============================================
# CLASES DEL SISTEMA (DINÁMICAS)
# =============================================

class DataLoader:
    """Carga datos desde cualquier formato."""
    
    def __init__(self, file):
        self.file = file

    def load_file(self):
        """Carga archivos CSV o Excel con validación."""
        file_name = self.file.name.lower()
        try:
            if file_name.endswith(".csv"):
                df = pd.read_csv(self.file)
            elif file_name.endswith((".xls", ".xlsx")):
                df = pd.read_excel(self.file)
            else:
                raise ValueError("❌ Formato no soportado. Use CSV o Excel")
            
            if df.shape[0] < 50:
                st.warning("⚠️ Dataset muy pequeño. Se recomienda mínimo 100 registros para mejores resultados.")
            
            return df
        except Exception as e:
            st.error(f"❌ Error al cargar archivo: {str(e)}")
            return None


class EDAAnalyzer:
    """Análisis exploratorio dinámico."""
    
    def __init__(self, df):
        self.df = df
    
    def show_basic_info(self):
        """Dashboard de información básica."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class='metric-box'>
                    <div class='metric-label'>Registros</div>
                    <div class='metric-value'>{self.df.shape[0]:,}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class='metric-box'>
                    <div class='metric-label'>Variables</div>
                    <div class='metric-value'>{self.df.shape[1]}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            numeric_cols = len(self.df.select_dtypes(include=[np.number]).columns)
            st.markdown(f"""
                <div class='metric-box'>
                    <div class='metric-label'>Numéricas</div>
                    <div class='metric-value'>{numeric_cols}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            cat_cols = len(self.df.select_dtypes(include=['object']).columns)
            st.markdown(f"""
                <div class='metric-box'>
                    <div class='metric-label'>Categóricas</div>
                    <div class='metric-value'>{cat_cols}</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Tabla de información detallada
        type_df = pd.DataFrame({
            'Columna': self.df.columns,
            'Tipo': self.df.dtypes.values,
            'No Nulos': self.df.count().values,
            'Nulos': self.df.isnull().sum().values,
            '% Nulos': (self.df.isnull().sum() / len(self.df) * 100).round(2)
        })
        st.dataframe(type_df, use_container_width=True, height=300)
    
    def show_statistics(self):
        """Estadísticas descriptivas."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) > 0:
            stats = numeric_df.describe().T
            stats['CV (%)'] = (stats['std'] / stats['mean'] * 100).round(2)
            
            st.dataframe(
                stats.style.background_gradient(cmap='Blues', subset=['mean', '50%']).format("{:.2f}"),
                use_container_width=True,
                height=400
            )
        else:
            st.warning("⚠️ No hay variables numéricas en el dataset.")
    
    def show_missing_data(self):
        """Análisis de datos faltantes."""
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        
        if missing.sum() == 0:
            st.markdown("<div class='custom-success'>✅ <b>Excelente:</b> No hay valores faltantes</div>", 
                       unsafe_allow_html=True)
            return
        
        missing_df = pd.DataFrame({
            'Columna': missing.index,
            'Valores Faltantes': missing.values,
            'Porcentaje (%)': missing_pct.values
        })
        missing_df = missing_df[missing_df['Valores Faltantes'] > 0].sort_values('Valores Faltantes', ascending=False)
        
        # Gráfico
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(missing_df['Columna'], missing_df['Porcentaje (%)'], color='#764ba2')
        ax.set_xlabel('Porcentaje de Valores Faltantes (%)', fontweight='bold')
        ax.set_title('Datos Faltantes por Variable', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.dataframe(missing_df, use_container_width=True)
    
    def plot_distributions(self, selected_cols=None):
        """Distribuciones de variables numéricas."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            st.warning("⚠️ No hay variables numéricas para visualizar.")
            return
        
        if selected_cols is None:
            selected_cols = st.multiselect(
                "📊 Selecciona variables a visualizar:", 
                numeric_cols, 
                default=numeric_cols[:min(3, len(numeric_cols))]
            )
        
        if selected_cols:
            for col in selected_cols:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Histograma
                ax1.hist(self.df[col].dropna(), bins=30, color='#667eea', edgecolor='black', alpha=0.7)
                ax1.set_title(f'Distribución de {col}', fontweight='bold')
                ax1.set_xlabel(col, fontweight='bold')
                ax1.set_ylabel('Frecuencia', fontweight='bold')
                ax1.grid(alpha=0.3)
                
                # Boxplot
                ax2.boxplot(self.df[col].dropna(), vert=True)
                ax2.set_title(f'Box Plot de {col}', fontweight='bold')
                ax2.set_ylabel(col, fontweight='bold')
                ax2.grid(alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
    
    def plot_correlation_matrix(self):
        """Matriz de correlación."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            st.warning("⚠️ Se necesitan al menos 2 variables numéricas.")
            return
        
        corr_matrix = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        ax.set_title('Matriz de Correlación', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        st.pyplot(fig)


class UniversalFeatureEngineer:
    """Ingeniería de características universal."""
    
    def __init__(self, df):
        self.df = df.copy()
        self.new_features = []
        self.date_columns = []
    
    def detect_date_columns(self):
        """Detecta columnas de fecha automáticamente."""
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                try:
                    pd.to_datetime(self.df[col], errors='raise')
                    self.date_columns.append(col)
                except:
                    pass
        return self.date_columns
    
    def create_temporal_features(self, date_col=None):
        """Crea características temporales desde cualquier columna de fecha."""
        if date_col is None:
            date_cols = self.detect_date_columns()
            if not date_cols:
                return self.df
            date_col = date_cols[0]
        
        if date_col in self.df.columns:
            self.df[date_col] = pd.to_datetime(self.df[date_col], errors="coerce")
            
            self.df[f"{date_col}_Year"] = self.df[date_col].dt.year
            self.df[f"{date_col}_Month"] = self.df[date_col].dt.month
            self.df[f"{date_col}_DayOfWeek"] = self.df[date_col].dt.dayofweek
            self.df[f"{date_col}_Quarter"] = self.df[date_col].dt.quarter
            self.df[f"{date_col}_IsWeekend"] = self.df[date_col].dt.dayofweek.isin([5, 6]).astype(int)
            
            self.new_features.extend([
                f"{date_col}_Year", f"{date_col}_Month", f"{date_col}_DayOfWeek", 
                f"{date_col}_Quarter", f"{date_col}_IsWeekend"
            ])
        
        return self.df
    
    def create_numeric_interactions(self, numeric_cols=None, max_interactions=5):
        """Crea interacciones entre variables numéricas."""
        if numeric_cols is None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            interactions_created = 0
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    if interactions_created >= max_interactions:
                        break
                    
                    # Multiplicación
                    new_col_name = f"{col1}_x_{col2}"
                    self.df[new_col_name] = self.df[col1] * self.df[col2]
                    self.new_features.append(new_col_name)
                    interactions_created += 1
        
        return self.df
    
    def engineer_all_features(self, create_interactions=False):
        """Aplica todas las transformaciones."""
        self.create_temporal_features()
        
        if create_interactions:
            self.create_numeric_interactions()
        
        return self.df, self.new_features


class UniversalDataPreprocessor:
    """Preprocesamiento universal de datos."""
    
    def __init__(self, df, target_col, columns_to_exclude=None, handle_nulls='auto'):
        self.df = df
        self.target_col = target_col
        self.columns_to_exclude = columns_to_exclude or []
        self.handle_nulls = handle_nulls  # 'auto', 'drop', 'impute'
        self.X = None
        self.y = None
        self.removed_rows = 0
        self.imputed_cols = []

    def preprocess(self):
        """Limpieza y preparación universal."""
        df = self.df.copy()
        initial_rows = len(df)
        
        # Limpiar nombres de columnas
        df.columns = df.columns.str.strip()
        
        # Eliminar duplicados
        duplicates = df.duplicated().sum()
        df = df.drop_duplicates()
        if duplicates > 0:
            st.info(f"🔄 Se eliminaron {duplicates} filas duplicadas")
        
        # Eliminar filas con target nulo
        target_nulls = df[self.target_col].isnull().sum()
        df = df.dropna(subset=[self.target_col])
        if target_nulls > 0:
            st.warning(f"⚠️ Se eliminaron {target_nulls} filas con target nulo")
        
        # Eliminar columnas con muchos nulos (>70%)
        null_threshold = 0.7
        cols_removed = []
        for col in df.columns:
            null_pct = df[col].isnull().sum() / len(df)
            if null_pct > null_threshold:
                cols_removed.append(f"{col} ({null_pct*100:.1f}%)")
                df = df.drop(columns=[col])
        
        if cols_removed:
            st.warning(f"🗑️ Columnas eliminadas por exceso de nulos: {', '.join(cols_removed)}")
        
        # Preparar X e y
        exclude_cols = [self.target_col] + self.columns_to_exclude
        self.X = df.drop(columns=exclude_cols, errors="ignore")
        self.y = df[self.target_col]
        
        # Eliminar columnas de fecha que quedaron
        date_cols = self.X.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            self.X = self.X.drop(columns=date_cols)
        
        # MANEJAR NULOS RESTANTES
        total_nulls = self.X.isnull().sum().sum()
        
        if total_nulls > 0:
            st.info(f"🔍 Detectados {total_nulls} valores nulos en features")
            
            if self.handle_nulls == 'drop':
                # Opción 1: Eliminar filas con cualquier nulo
                before = len(self.X)
                self.X = self.X.dropna()
                self.y = self.y[self.X.index]
                after = len(self.X)
                st.warning(f"🗑️ Se eliminaron {before - after} filas con valores nulos")
            
            elif self.handle_nulls in ['impute', 'auto']:
                # Opción 2: Imputar valores (RECOMENDADO)
                numeric_cols = self.X.select_dtypes(include=[np.number]).columns
                categorical_cols = self.X.select_dtypes(include=['object', 'category']).columns
                
                # Imputar numéricos con la mediana
                for col in numeric_cols:
                    if self.X[col].isnull().sum() > 0:
                        median_val = self.X[col].median()
                        nulls_count = self.X[col].isnull().sum()
                        self.X[col].fillna(median_val, inplace=True)
                        self.imputed_cols.append(f"{col} (num: {nulls_count})")
                
                # Imputar categóricos con la moda
                for col in categorical_cols:
                    if self.X[col].isnull().sum() > 0:
                        mode_val = self.X[col].mode()[0] if not self.X[col].mode().empty else 'Unknown'
                        nulls_count = self.X[col].isnull().sum()
                        self.X[col].fillna(mode_val, inplace=True)
                        self.imputed_cols.append(f"{col} (cat: {nulls_count})")
                
                if self.imputed_cols:
                    st.success(f"✅ Imputados valores nulos en {len(self.imputed_cols)} columnas")
                    with st.expander("📋 Ver detalles de imputación"):
                        for col_info in self.imputed_cols:
                            st.write(f"• {col_info}")
        
        self.removed_rows = initial_rows - len(df)
        
        return self.X, self.y


class ModelTrainer:
    """Entrenamiento del modelo."""
    
    def __init__(self, model=None):
        self.model = model or Ridge()
        self.best_model = None
        self.best_score = None
        self.cv_scores = None
        self.best_params = None

    def train(self, X_train, y_train):
        """Entrena el modelo con cualquier tipo de datos."""
        # Identificar columnas por tipo
        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

        # Crear preprocesador
        transformers = []
        if num_cols:
            transformers.append(("num", StandardScaler(), num_cols))
        if cat_cols:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
        
        preprocessor = ColumnTransformer(transformers)

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", self.model)
        ])

        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🔄 Validación cruzada...")
            progress_bar.progress(30)
            self.cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')

            status_text.text("🔍 Optimizando hiperparámetros...")
            progress_bar.progress(60)
            param_grid = {"model__alpha": [0.01, 0.1, 1, 10, 100]}
            grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
            grid.fit(X_train, y_train)

            self.best_model = grid.best_estimator_
            self.best_score = grid.best_score_
            self.best_params = grid.best_params_
            
            progress_bar.progress(100)
            status_text.text("✅ Entrenamiento completado")
        
        except Exception as e:
            st.error(f"❌ Error en entrenamiento: {str(e)}")
            return None
        
        return self.best_model
    
    def plot_learning_curves(self, X_train, y_train):
        """Curvas de aprendizaje."""
        try:
            train_sizes, train_scores, val_scores = learning_curve(
                self.best_model, X_train, y_train, cv=5, scoring='r2',
                train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
            )
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(train_sizes, train_mean, label='Entrenamiento', marker='o', color='#667eea', linewidth=2)
            ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#667eea')
            ax.plot(train_sizes, val_mean, label='Validación', marker='s', color='#764ba2', linewidth=2)
            ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='#764ba2')
            
            ax.set_xlabel('Tamaño del Conjunto de Entrenamiento', fontsize=12, fontweight='bold')
            ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
            ax.set_title('Curvas de Aprendizaje - Detección de Overfitting', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=11)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ Error al generar curvas de aprendizaje: {str(e)}")


class ModelEvaluator:
    """Evaluación del modelo."""
    
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def evaluate(self):
        """Calcula métricas."""
        try:
            y_pred = self.model.predict(self.X_test)
            
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            # MAPE solo para valores no cero
            mask = self.y_test != 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((self.y_test[mask] - y_pred[mask]) / self.y_test[mask])) * 100
            else:
                mape = 0
            
            return {"RMSE": rmse, "MAE": mae, "R²": r2, "MAPE (%)": mape}, y_pred
        
        except Exception as e:
            st.error(f"❌ Error en evaluación: {str(e)}")
            return None, None

    def plot_residuals(self, y_pred):
        """Gráfico de residuos."""
        try:
            residuals = self.y_test - y_pred
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(self.y_test, residuals, alpha=0.6, edgecolors='k', color='#667eea')
            ax.axhline(0, color='red', linestyle='--', linewidth=2)
            ax.set_xlabel("Valores Reales", fontsize=12, fontweight='bold')
            ax.set_ylabel("Residuos", fontsize=12, fontweight='bold')
            ax.set_title("Análisis de Residuos", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ Error al generar gráfico de residuos: {str(e)}")
    
    def plot_predictions_vs_actual(self, y_pred):
        """Predicciones vs reales."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.scatter(self.y_test, y_pred, alpha=0.6, edgecolors='k', color='#764ba2')
            ax.plot([self.y_test.min(), self.y_test.max()], 
                    [self.y_test.min(), self.y_test.max()], 
                    'r--', lw=2, label='Predicción perfecta')
            ax.set_xlabel("Valores Reales", fontsize=12, fontweight='bold')
            ax.set_ylabel("Predicciones", fontsize=12, fontweight='bold')
            ax.set_title("Predicciones vs Valores Reales", fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ Error al generar gráfico de predicciones: {str(e)}")


# =============================================
# 🧠 APLICACIÓN PRINCIPAL
# =============================================
def main():
    st.set_page_config(
        page_title="ML Universal System",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    set_custom_style()
    
    # Header
    st.markdown("<h1 class='main-title'>🤖 Sistema de Machine Learning Universal</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>🚀 Funciona con Cualquier Dataset CSV/Excel</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Configuración")
        st.markdown("---")
        st.markdown("### 📊 Estado del Sistema")
        st.markdown("<span class='badge badge-success'>Sistema Activo</span>", unsafe_allow_html=True)
        st.markdown("<span class='badge badge-info'>v3.0 Universal</span>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📚 Recursos")
        st.markdown("[📖 Documentación](https://scikit-learn.org)")
        st.markdown("[🎓 Tutoriales](https://streamlit.io)")
        
        st.markdown("---")
        st.markdown("### 💡 Características")
        st.markdown("✅ Cualquier CSV/Excel")
        st.markdown("✅ Detección automática")
        st.markdown("✅ Feature engineering")
        st.markdown("✅ Optimización automática")
    
    # Información del proyecto
    st.markdown(PROJECT_INFO, unsafe_allow_html=True)
    st.markdown("---")
    
    # Sección de carga
    st.markdown("<h2 class='section-header'>📂 Paso 1: Carga tu Dataset</h2>", unsafe_allow_html=True)
    
    col_upload, col_config = st.columns([1, 1])
    
    with col_upload:
        st.markdown("### 📤 Cargar Dataset")
        uploaded_file = st.file_uploader(
            "Arrastra o selecciona tu archivo",
            type=["csv", "xlsx", "xls"],
            help="El sistema se adapta automáticamente a cualquier estructura de datos"
        )
    
    with col_config:
        st.markdown("### 📋 Información")
        st.markdown("""
        <div class='custom-card'>
        <p><b>El sistema es 100% dinámico:</b></p>
        <ul>
            <li>✅ Detecta automáticamente tipos de datos</li>
            <li>✅ Identifica variables numéricas y categóricas</li>
            <li>✅ Encuentra columnas de fecha automáticamente</li>
            <li>✅ Limpia y preprocesa los datos</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if uploaded_file:
            st.markdown("<span class='badge badge-success'>✅ Archivo Cargado</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='badge badge-warning'>⏳ Esperando archivo...</span>", unsafe_allow_html=True)

    if uploaded_file:
        # Carga
        with st.spinner("🔄 Cargando datos..."):
            df = DataLoader(uploaded_file).load_file()
            
            if df is not None:
                st.markdown(f"<div class='custom-success'>✅ <b>Archivo cargado exitosamente:</b> {df.shape[0]:,} filas × {df.shape[1]} columnas</div>", 
                           unsafe_allow_html=True)
                
                with st.expander("👁️ Vista Previa del Dataset", expanded=True):
                    st.dataframe(df.head(20), use_container_width=True)
        
        # EDA
        st.markdown("---")
        st.markdown("<h2 class='section-header'>📊 Paso 2: Análisis Exploratorio</h2>", unsafe_allow_html=True)
        
        eda = EDAAnalyzer(df)
        
        with st.expander("📋 Resumen del Dataset", expanded=True):
            eda.show_basic_info()
        
        with st.expander("📈 Estadísticas Descriptivas"):
            eda.show_statistics()
        
        with st.expander("🔍 Análisis de Valores Faltantes"):
            eda.show_missing_data()
        
        with st.expander("📉 Distribuciones de Variables"):
            eda.plot_distributions()
        
        with st.expander("🔥 Matriz de Correlación"):
            eda.plot_correlation_matrix()
        
        # Feature Engineering
        st.markdown("---")
        st.markdown("<h2 class='section-header'>🔧 Paso 3: Ingeniería de Características</h2>", unsafe_allow_html=True)
        
        fe_col1, fe_col2 = st.columns(2)
        
        with fe_col1:
            apply_temporal = st.checkbox("📅 Crear características temporales", value=True, 
                                        help="Detecta y procesa automáticamente columnas de fecha")
        
        with fe_col2:
            apply_interactions = st.checkbox("🔗 Crear interacciones numéricas", value=False,
                                            help="Crea productos entre variables numéricas (aumenta complejidad)")
        
        if apply_temporal or apply_interactions:
            with st.spinner("🔄 Creando nuevas características..."):
                engineer = UniversalFeatureEngineer(df)
                
                # Detectar columnas de fecha
                date_cols = engineer.detect_date_columns()
                if date_cols:
                    st.info(f"🗓️ Columnas de fecha detectadas: {', '.join(date_cols)}")
                
                df, new_features = engineer.engineer_all_features(create_interactions=apply_interactions)
                
                if new_features:
                    st.markdown(f"<div class='custom-success'>✅ <b>Se crearon {len(new_features)} nuevas características:</b><br>{', '.join(new_features)}</div>", 
                               unsafe_allow_html=True)
                else:
                    st.markdown("<div class='custom-info'>ℹ️ No se crearon nuevas características</div>", 
                               unsafe_allow_html=True)
        
        # Configuración del Modelo
        st.markdown("---")
        st.markdown("<h2 class='section-header'>🎯 Paso 4: Configuración del Modelo</h2>", unsafe_allow_html=True)
        
        config_col1, config_col2 = st.columns([2, 1])
        
        with config_col1:
            # Selección de variable objetivo
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                st.error("❌ No hay variables numéricas para predecir. El dataset debe tener al menos una columna numérica como objetivo.")
                return
            
            target_col = st.selectbox(
                "🎯 Selecciona la variable objetivo (a predecir):",
                numeric_cols,
                help="Variable que deseas predecir (debe ser numérica)"
            )
            
            # Selección de columnas a excluir
            all_cols = df.columns.tolist()
            all_cols.remove(target_col)
            
            cols_to_exclude = st.multiselect(
                "🚫 Columnas a excluir del entrenamiento (opcional):",
                all_cols,
                help="Columnas que no quieres usar como features (ej: IDs, nombres)"
            )
            
            # NUEVO: Opción de manejo de nulos
            st.markdown("---")
            null_handling = st.radio(
                "🔧 Manejo de valores nulos:",
                options=['auto', 'drop', 'impute'],
                index=0,
                help="""
                • Auto: Imputación inteligente (Recomendado)
                • Drop: Eliminar filas con nulos
                • Impute: Rellenar con mediana/moda
                """,
                horizontal=True
            )
        
        with config_col2:
            test_size = st.slider(
                "📊 % Prueba:", 
                min_value=10, 
                max_value=40, 
                value=20,
                help="Porcentaje del dataset para validación final"
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.metric("🔢 Train/Test", f"{100-test_size}/{test_size}")
        
        # Mostrar resumen de configuración
        st.markdown(f"""
        <div class='custom-info'>
        <b>📋 Configuración seleccionada:</b><br>
        🎯 <b>Variable objetivo:</b> {target_col}<br>
        🚫 <b>Columnas excluidas:</b> {len(cols_to_exclude) + 1} ({target_col}, {', '.join(cols_to_exclude[:3])}{'...' if len(cols_to_exclude) > 3 else ''})<br>
        📊 <b>Features disponibles:</b> {len(all_cols) - len(cols_to_exclude)}
        </div>
        """, unsafe_allow_html=True)
        
        # Preprocesamiento
        st.markdown("---")
        st.markdown("### 🔄 Preprocesamiento de Datos")
        
        with st.spinner("🔄 Limpiando y preparando datos..."):
            preprocessor = UniversalDataPreprocessor(df, target_col, cols_to_exclude, handle_nulls=null_handling)
            X, y = preprocessor.preprocess()
            
            if preprocessor.removed_rows > 0:
                st.markdown(f"<div class='custom-warning'>⚠️ <b>Atención:</b> Se eliminaron {preprocessor.removed_rows} filas inválidas</div>", 
                           unsafe_allow_html=True)
        
        # Verificar que tenemos datos válidos
        if X.shape[0] < 10:
            st.error("❌ Datos insuficientes después del preprocesamiento. Necesitas al menos 10 registros válidos.")
            return
        
        # Métricas de preprocesamiento
        prep_col1, prep_col2, prep_col3, prep_col4 = st.columns(4)
        
        prep_col1.metric("🎯 Features", X.shape[1])
        prep_col2.metric("📊 Registros", f"{len(y):,}")
        prep_col3.metric("📈 Calidad", f"{(len(y)/df.shape[0]*100):.1f}%")
        prep_col4.metric("🗑️ Eliminados", preprocessor.removed_rows)
        
        # Mostrar tipos de features
        num_features = len(X.select_dtypes(include=[np.number]).columns)
        cat_features = len(X.select_dtypes(include=['object', 'category']).columns)
        
        st.markdown(f"""
        <div class='custom-card'>
        <b>📊 Composición de Features:</b><br>
        🔢 Variables numéricas: <b>{num_features}</b><br>
        📝 Variables categóricas: <b>{cat_features}</b>
        </div>
        """, unsafe_allow_html=True)
        
        # División de datos
        try:
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=test_size*2/100, random_state=42
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42
            )
            
            st.markdown(f"""
            <div class='custom-info'>
            <b>📊 División de Datos Completada:</b><br><br>
            🟢 <b>Entrenamiento:</b> {len(X_train):,} registros ({(len(X_train)/len(y)*100):.1f}%)<br>
            🟡 <b>Validación:</b> {len(X_val):,} registros ({(len(X_val)/len(y)*100):.1f}%)<br>
            🔴 <b>Prueba:</b> {len(X_test):,} registros ({(len(X_test)/len(y)*100):.1f}%)
            </div>
            """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"❌ Error al dividir los datos: {str(e)}")
            return
        
        # Entrenamiento
        st.markdown("---")
        st.markdown("<h2 class='section-header'>🚀 Paso 5: Entrenamiento del Modelo</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='custom-card'>
        <h4>🎯 ¿Listo para entrenar?</h4>
        <p>El sistema realizará:</p>
        <ul>
            <li>✅ Validación cruzada de 5 pliegues</li>
            <li>🔍 Optimización automática de hiperparámetros</li>
            <li>📊 Evaluación completa de métricas</li>
            <li>📈 Análisis de curvas de aprendizaje</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Iniciar Entrenamiento", type="primary", use_container_width=True):
            st.markdown("---")
            
            trainer = ModelTrainer()
            best_model = trainer.train(X_train, y_train)
            
            if best_model is None:
                st.error("❌ El entrenamiento falló. Verifica tus datos.")
                return
            
            # Resultados del Entrenamiento
            st.markdown("<h2 class='section-header'>✅ Resultados del Entrenamiento</h2>", unsafe_allow_html=True)
            
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                st.markdown("### 📊 Validación Cruzada (5-Fold)")
                cv_col1, cv_col2 = st.columns(2)
                cv_col1.metric("📈 R² Promedio", f"{np.mean(trainer.cv_scores):.4f}")
                cv_col2.metric("📉 Desv. Est.", f"{np.std(trainer.cv_scores):.4f}")
                
                # Gráfico CV
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(range(1, 6), trainer.cv_scores, color='#667eea', edgecolor='black', linewidth=1.5)
                ax.set_xlabel('Fold', fontsize=11, fontweight='bold')
                ax.set_ylabel('R² Score', fontsize=11, fontweight='bold')
                ax.set_title('Scores de Validación Cruzada', fontsize=12, fontweight='bold')
                ax.grid(alpha=0.3, linestyle='--')
                ax.set_ylim([min(trainer.cv_scores) - 0.1, max(trainer.cv_scores) + 0.1])
                plt.tight_layout()
                st.pyplot(fig)
            
            with result_col2:
                st.markdown("### 🎯 Optimización de Hiperparámetros")
                opt_col1, opt_col2 = st.columns(2)
                opt_col1.metric("🏆 Mejor R²", f"{trainer.best_score:.4f}")
                opt_col2.metric("⚙️ Alpha Óptimo", f"{trainer.best_params['model__alpha']}")
                
                # Indicador de calidad
                if trainer.best_score > 0.8:
                    quality, badge_class, emoji = "Excelente", "badge-success", "🎉"
                elif trainer.best_score > 0.6:
                    quality, badge_class, emoji = "Bueno", "badge-info", "👍"
                else:
                    quality, badge_class, emoji = "Regular", "badge-warning", "⚠️"
                
                st.markdown(f"<br><span class='badge {badge_class}'>{emoji} Calidad: {quality}</span>", unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class='custom-card' style='margin-top: 20px;'>
                <p><b>💡 Interpretación:</b></p>
                <p>El modelo explica el <b>{trainer.best_score*100:.1f}%</b> de la variabilidad.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Curvas de aprendizaje
            st.markdown("---")
            st.markdown("### 📈 Análisis de Overfitting")
            trainer.plot_learning_curves(X_train, y_train)
            
            # Evaluación
            st.markdown("---")
            st.markdown("<h2 class='section-header'>🎯 Paso 6: Evaluación Final</h2>", unsafe_allow_html=True)
            
            evaluator = ModelEvaluator(best_model, X_test, y_test)
            metrics, y_pred = evaluator.evaluate()
            
            if metrics is None:
                st.error("❌ La evaluación falló.")
                return
            
            # Métricas principales
            st.markdown("### 📊 Métricas de Desempeño")
            
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            
            with m_col1:
                st.markdown(f"""
                    <div class='metric-box'>
                        <div class='metric-label'>RMSE</div>
                        <div class='metric-value'>{metrics['RMSE']:.2f}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with m_col2:
                st.markdown(f"""
                    <div class='metric-box'>
                        <div class='metric-label'>MAE</div>
                        <div class='metric-value'>{metrics['MAE']:.2f}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with m_col3:
                st.markdown(f"""
                    <div class='metric-box'>
                        <div class='metric-label'>R²</div>
                        <div class='metric-value'>{metrics['R²']:.4f}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with m_col4:
                st.markdown(f"""
                    <div class='metric-box'>
                        <div class='metric-label'>MAPE</div>
                        <div class='metric-value'>{metrics['MAPE (%)']:.2f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Interpretación
            st.markdown("---")
            st.markdown("### 💡 Interpretación de Resultados")
            
            conclusion = (
                "🎉 Modelo EXCELENTE - Listo para producción" if metrics['R²'] > 0.8 
                else "👍 Modelo ACEPTABLE - Considerar mejoras" if metrics['R²'] > 0.6 
                else "🔧 Modelo necesita MEJORAS significativas"
            )
            
            st.markdown(f"""
            <div class='custom-card'>
            <h4>📊 Capacidad Explicativa</h4>
            <p>El modelo explica el <b style='color: #667eea; font-size: 1.2em;'>{metrics['R²']*100:.1f}%</b> de la variabilidad en <b>{target_col}</b>.</p>
            
            <h4>🎯 Precisión</h4>
            <ul>
                <li><b>Error absoluto promedio (MAE):</b> {metrics['MAE']:.2f}</li>
                <li><b>Error porcentual (MAPE):</b> {metrics['MAPE (%)']:.1f}%</li>
            </ul>
            
            <h4>✅ Conclusión</h4>
            <p><b>{conclusion}</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Visualizaciones
            st.markdown("---")
            st.markdown("### 📊 Análisis Visual")
            
            vis_col1, vis_col2 = st.columns(2)
            
            with vis_col1:
                st.markdown("#### 📉 Análisis de Residuos")
                evaluator.plot_residuals(y_pred)
            
            with vis_col2:
                st.markdown("#### 📈 Predicciones vs Reales")
                evaluator.plot_predictions_vs_actual(y_pred)
            
            # Predicciones
            st.markdown("---")
            st.markdown("<h2 class='section-header'>🔮 Paso 7: Predicciones</h2>", unsafe_allow_html=True)
            
            results_df = pd.DataFrame({
                'Valor Real': y_test.values[:10],
                'Predicción': y_pred[:10],
                'Error Absoluto': np.abs(y_test.values[:10] - y_pred[:10]),
                'Error (%)': np.abs((y_test.values[:10] - y_pred[:10]) / y_test.values[:10] * 100)
            })
            
            st.dataframe(
                results_df.style.background_gradient(subset=['Error Absoluto', 'Error (%)'], cmap='Reds')
                                .format({'Valor Real': '{:.2f}', 'Predicción': '{:.2f}', 
                                       'Error Absoluto': '{:.2f}', 'Error (%)': '{:.2f}%'}),
                use_container_width=True
            )
            
            # Distribución de errores
            st.markdown("### 📊 Distribución de Errores")
            errors = np.abs(y_test - y_pred)
            
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.hist(errors, bins=30, color='#764ba2', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Error Absoluto', fontsize=12, fontweight='bold')
            ax.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
            ax.set_title('Distribución de Errores', fontsize=14, fontweight='bold')
            ax.axvline(errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {errors.mean():.2f}')
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Estadísticas de error
            e_col1, e_col2, e_col3, e_col4 = st.columns(4)
            e_col1.metric("📉 Mínimo", f"{errors.min():.2f}")
            e_col2.metric("📊 Promedio", f"{errors.mean():.2f}")
            e_col3.metric("📍 Mediano", f"{errors.median():.2f}")
            e_col4.metric("📈 Máximo", f"{errors.max():.2f}")
            
            # Exportación
            st.markdown("---")
            st.markdown("<h2 class='section-header'>💾 Exportación</h2>", unsafe_allow_html=True)
            
            export_df = pd.DataFrame({
                'Real': y_test.values,
                'Prediccion': y_pred,
                'Error': y_test.values - y_pred,
                'Error_Absoluto': np.abs(y_test.values - y_pred)
            })
            
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Descargar Predicciones (CSV)",
                data=csv,
                file_name=f"predicciones_{target_col}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Resumen final
            st.markdown("---")
            st.markdown(f"""
            <div class='custom-success'>
            🎉 <b>¡Pipeline completado!</b><br>
            Modelo entrenado para predecir: <b>{target_col}</b><br>
            Features utilizados: <b>{X.shape[1]}</b> | R² Score: <b>{metrics['R²']:.4f}</b>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
    
    else:
        st.markdown("""
        <div class='custom-info' style='margin-top: 30px;'>
        <h3>👆 Carga tu dataset para comenzar</h3>
        <p>El sistema es completamente dinámico y se adapta a cualquier estructura de datos.</p>
        <p><b>Formatos soportados:</b> CSV, Excel (.xlsx, .xls)</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()