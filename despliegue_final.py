# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# Configuración de página
st.set_page_config(
    page_title="Predicción de Enfermedad Cardíaca",
    page_icon="🫀",
    layout="centered"
)

# Estilos
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #c0392b;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        width: 100%;
        padding: 10px;
        border: none;
    }
    .stButton>button:hover { background-color: #a93226; }
    .titulo { text-align: center; color: #c0392b; font-size: 36px; font-weight: bold; }
    .subtitulo { text-align: center; color: #555; font-size: 16px; margin-bottom: 30px; }
    </style>
""", unsafe_allow_html=True)

# NECESARIO para cargar el pipeline
def preparar_datos(df):
    df = df.copy()
    df.loc[df['RestingBP'] == 0, 'RestingBP'] = np.nan
    df['RestingBP'] = df['RestingBP'].fillna(df['RestingBP'].median())
    if 'Cholesterol' in df.columns:
        df = df.drop(columns=['Cholesterol'])
    df = pd.get_dummies(df, columns=['Sex', 'ExerciseAngina'], drop_first=True, dtype=int)
    df = pd.get_dummies(df, columns=['ChestPainType', 'RestingECG', 'ST_Slope'], drop_first=False, dtype=int)
    if 'ST_Slope_Flat' in df.columns:
        df = df.drop(columns=['ST_Slope_Flat'])
    df['HR_Reserve'] = df['MaxHR'] - df['Age']
    df['AgeGroup'] = df['Age'].apply(lambda age: 0 if age < 45 else (1 if age < 60 else 2))
    df['BP_Category'] = df['RestingBP'].apply(lambda bp: 0 if bp < 120 else (1 if bp < 140 else 2)) if 'RestingBP' in df.columns else 0
    df = df.drop(columns=['Age', 'RestingBP'], errors='ignore')
    if 'HeartDisease' in df.columns:
        df = df.drop(columns=['HeartDisease'])
    return df

# Cargar modelo
filename = os.path.join(os.path.dirname(__file__), 'modelo_finallr.pkl')
pipeline = pickle.load(open(filename, 'rb'))

# Encabezado
st.markdown('<p class="titulo">🫀 Predicción de Enfermedad Cardíaca</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitulo">Complete los datos clínicos del paciente para obtener una predicción</p>', unsafe_allow_html=True)
st.divider()

# Entradas en dos columnas
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📋 Datos Generales")
    age = st.slider('Edad (años)', min_value=18, max_value=100, value=50, step=1)
    sex = st.selectbox('Sexo biológico del paciente', ['Masculino', 'Femenino'])
    resting_bp = st.slider('Presión Arterial en Reposo (mm Hg)', min_value=80, max_value=200, value=120, step=1)
    max_hr = st.slider('Frecuencia Cardíaca Máxima (lpm)', min_value=60, max_value=220, value=150, step=1)
    oldpeak = st.slider('Depresión del Segmento ST (Oldpeak)', min_value=0.0, max_value=6.0, value=1.0, step=0.1)

with col2:
    st.markdown("### 🩺 Datos Clínicos")
    fasting_bs = st.selectbox('¿Azúcar en ayunas > 120 mg/dl?', ['No', 'Sí'])
    chest_pain = st.selectbox('Tipo de Dolor en el Pecho', [
        'Asintomático (ASY)', 'Angina Típica (TA)',
        'Angina Atípica (ATA)', 'Dolor No Anginoso (NAP)'
    ])
    resting_ecg = st.selectbox('Electrocardiograma en Reposo', [
        'Normal', 'Anormalidad de onda ST-T (ST)',
        'Hipertrofia Ventricular Izquierda (LVH)'
    ])
    exercise_angina = st.selectbox('¿Angina inducida por ejercicio?', ['No', 'Sí'])
    st_slope = st.selectbox('Pendiente del Segmento ST', [
        'Ascendente (Up)', 'Plana (Flat)', 'Descendente (Down)'
    ])

st.divider()

# Predicción
if st.button('🔍 Predecir'):
    sex_m = 1 if sex == 'Masculino' else 0
    exercise_angina_y = 1 if exercise_angina == 'Sí' else 0
    chest_asy = 1 if chest_pain == 'Asintomático (ASY)' else 0
    chest_ata = 1 if chest_pain == 'Angina Atípica (ATA)' else 0
    chest_nap = 1 if chest_pain == 'Dolor No Anginoso (NAP)' else 0
    chest_ta = 1 if chest_pain == 'Angina Típica (TA)' else 0
    ecg_lvh = 1 if resting_ecg == 'Hipertrofia Ventricular Izquierda (LVH)' else 0
    ecg_normal = 1 if resting_ecg == 'Normal' else 0
    ecg_st = 1 if resting_ecg == 'Anormalidad de onda ST-T (ST)' else 0
    slope_down = 1 if st_slope == 'Descendente (Down)' else 0
    slope_up = 1 if st_slope == 'Ascendente (Up)' else 0
    hr_reserve = max_hr - age
    age_group = 0 if age < 45 else (1 if age < 60 else 2)
    bp_category = 0 if resting_bp < 120 else (1 if resting_bp < 140 else 2)
    fasting = 1 if fasting_bs == 'Sí' else 0

    datos = pd.DataFrame([[fasting, max_hr, oldpeak, sex_m, exercise_angina_y,
                           chest_asy, chest_ata, chest_nap, chest_ta,
                           ecg_lvh, ecg_normal, ecg_st, slope_down, slope_up,
                           hr_reserve, age_group, bp_category]],
                         columns=['FastingBS', 'MaxHR', 'Oldpeak', 'Sex_M', 'ExerciseAngina_Y',
                                  'ChestPainType_ASY', 'ChestPainType_ATA', 'ChestPainType_NAP',
                                  'ChestPainType_TA', 'RestingECG_LVH', 'RestingECG_Normal',
                                  'RestingECG_ST', 'ST_Slope_Down', 'ST_Slope_Up',
                                  'HR_Reserve', 'AgeGroup', 'BP_Category'])

    prediccion = pipeline.named_steps['modelo'].predict(datos)[0]
    probabilidad = pipeline.named_steps['modelo'].predict_proba(datos)[0][1]

    st.divider()
    if prediccion == 1:
        st.error(f'⚠️ El paciente TIENE riesgo de enfermedad cardíaca')
    else:
        st.success(f'✅ El paciente NO tiene riesgo de enfermedad cardíaca')

    st.progress(float(probabilidad))
    st.info(f'📊 Probabilidad de enfermedad: **{probabilidad*100:.1f}%**')
    st.warning('⚕️ Este resultado es orientativo. Consulte siempre a un médico especialista.')
