# prison_break_stonks_ml.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import datetime
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                           silhouette_score, mean_squared_error, r2_score)
from sklearn.preprocessing import StandardScaler
import joblib
from io import BytesIO
import base64
import time

# ========================================
# Prison Break Theme
# ========================================

st.set_page_config(
    page_title="Prison Break: ML",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with Prison Break theme
st.markdown("""
<style>
:root {
    --orange: #FF555F;
    --blue: #4169E1;
    --dark: #132882;
    --light: #E000E0E0;
    --white: #FFFFFF ;
}

body {
    background-color: var(--dark);
    color: var(--light);
    font-family: 'Courier New', monospace;
}

.stApp {
    background-image: url('https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcTZtbmVlM3FncTlqMTM5MWRxbmtrN282Y2MxdWc0YnRmbTNrODd5eSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/14qXTOs1UIgFsk/giphy.gif');
    background-size: cover;
    background-attachment: fixed;
}

.stButton>button {
    background-color: var(--orange);
    color: black;
    border: 2px solid var(--white);
    border-radius: 5px;
    padding: 8px 16px;
    font-weight: bold;
    transition: all 0.3s;
}

.stButton>button:hover {
    background-color: var(--blue);
    color: neon;
    transform: scale(1.05);
    box-shadow: 0 0 10px var(--orange);
}

.stSidebar {
    background-color: rgba(128, 128, 128);
    border-right: 2px solid var(--orange);
}

.stDataFrame {
    background-color: rgba(0, 0, 0, 0.7) !important;
    border: 1px solid var(--blue) !important;
    border-radius: 5px;
}

.stAlert {
    border-left: 4px solid var(--orange) !important;
}

.stProgress > div > div > div {
    background-color: var(--orange) !important;
}

@keyframes flicker {
    0% { opacity: 1; }
    50% { opacity: 0.8; }
    100% { opacity: 1; }
}

.flicker {
    animation: flicker 2s infinite;
}

.prison-title {
    color: var(--orange);
    text-shadow: 0 0 10px var(--red);
    font-size: 3rem;
    text-align: center;
    margin-bottom: 1rem;
}

.prison-subtitle {
    color: var(--white);
    border-bottom: 2px solid var(--white);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

.metric-card {
    background: rgba(0, 0, 0, 0.7);
    border: 1px solid var(--blue);
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
    box-shadow: 0 0 15px var(--orange);
}
</style>
""", unsafe_allow_html=True)

# ========================================
# Helper Functions
# ========================================

def add_prison_break_gif():
    """Add Prison Break themed GIF"""
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExenM3Z3JubGpvOHg1c3hleGljdjUyNGE5bmF0NTVubTJmanAzd3dzcCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/3o7abnQiguzMTaYlOM/giphy.gif" 
             width="60%" style="border: 3px solid #FF5F15; border-radius: 5px;">
    </div>
    """, unsafe_allow_html=True)

def add_tattoo_animation():
    """Add Prison Break tattoo blueprint animation"""
    st.markdown("""
    <div style="text-align: center; margin: 1rem 0;">
        <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExa3B3Mzh4NDMxb3l6bTJpejNuejN0N3VveHMycWJmbzdmdmMyZ2VqYyZlcD12MV9naWZzX3NlYXJjaCZjdD1n/OTD0gCMNZpL3cQGrH1/giphy.gif" 
             width="30%" style="border: 2px solid #4169E1; border-radius: 5px;">
    </div>
    """, unsafe_allow_html=True)

def validate_data(df):
    """Validate and preprocess financial data"""
    try:
        # Clean column names
        df.columns = [col.lower().strip().replace(' ', '_') for col in df.columns]
        
        # Identify close price column
        close_aliases = ['close', 'closing_price', 'last_price', 'adj_close', 'price']
        for alias in close_aliases:
            if alias in df.columns:
                df = df.rename(columns={alias: 'close'})
                break
        
        # Fallback to first numeric column if no close found
        if 'close' not in df.columns:
            numeric_cols = df.select_dtypes(include=np.number).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in dataset")
            df = df.rename(columns={numeric_cols[0]: 'close'})
            st.warning(f"⚠️ Using '{numeric_cols[0]}' as close price")
        
        # Convert to numeric and handle missing values
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['close'].ffill(inplace=True)
        df['close'].bfill(inplace=True)
        
        # Handle dates
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
        
        return df
    except Exception as e:
        st.error(f"🔴 Data validation failed: {str(e)}")
        st.stop()

# ========================================
# Main Application
# ========================================

def main():
    # Title with Prison Break theme
    st.markdown("""
    <div style="text-align: center;">
        <h1 class="prison-title flicker">PRISON BREAK: ML</h1>
        <p style="color: #4169E1; font-size: 1.2rem;">Escape the market with machine learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add Prison Break GIF
    add_prison_break_gif()

    # Initialize session state
    if 'pipeline_step' not in st.session_state:
        st.session_state.update({
            'pipeline_step': 0,
            'raw_df': None,
            'df': None,
            'model': None,
            'X_train': None,
            'X_test': None,
            'y_train': None,
            'y_test': None
        })

    # Sidebar Configuration
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1rem;">
            <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExenM3Z3JubGpvOHg1c3hleGljdjUyNGE5bmF0NTVubTJmanAzd3dzcCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/rMS1sUPhv95f2/giphy.gif" width="80%" style="border: 2px solid #FF5F15; border-radius: 5px;">
        </div>
        """, unsafe_allow_html=True)
        
        data_source = st.radio("DATA SOURCE:", ["📤 Upload Dataset", "🌐 Yahoo Finance"])
        model_type = st.selectbox("ANALYSIS TYPE:", 
                                ["📈 Regression", "🔮 Classification", "🌀 Clustering"])
        
        st.markdown("---")
        st.markdown("**PRISON BREAK PROGRESS:**")
        steps = ["Data Loaded", "Preprocessed", "Features Engineered", 
                "Data Split", "Model Trained", "Evaluated"]
        current_step = st.session_state.pipeline_step
        st.markdown("\n".join([f"{'🔓' if i < current_step else '🔒'} {step}" 
                             for i, step in enumerate(steps)]))

    # Data Loading
    if st.session_state.pipeline_step == 0:
        with st.container():
            st.markdown("## 🔓 PHASE 1: DATA ACQUISITION")
            
            if data_source == "📤 Upload Dataset":
                uploaded_file = st.file_uploader("UPLOAD DATASET (CSV)", type=["csv"])
                if uploaded_file and st.button("🚀 LOAD DATA"):
                    try:
                        df = pd.read_csv(uploaded_file)
                        df = validate_data(df)
                        st.session_state.raw_df = df
                        st.session_state.pipeline_step = 1
                        st.success("✅ Data loaded successfully!")
                        st.dataframe(df).head(3).style.set_properties(**{
                            'background-color': 'white',
                            'color': '#FF5666',
                            'border-color': '#4169E1'
                        })
                    except Exception as e:
                        st.error(f"🔴 {str(e)}")

            else:
                col1, col2 = st.columns(2)
                with col1:
                    ticker = st.text_input("TICKER SYMBOL", "AAPL").upper()
                with col2:
                    start = st.date_input("START DATE", datetime(2020,1,1))
                    end = st.date_input("END DATE", datetime.today())
                
                if st.button("🌐 FETCH MARKET DATA"):
                    with st.spinner("Accessing market data..."):
                        try:
                            df = yf.download(ticker, start=start, end=end)
                            if not df.empty:
                                df = df.reset_index().rename(columns={'Close': 'close'})
                                df = validate_data(df)
                                st.session_state.raw_df = df
                                st.session_state.pipeline_step = 1
                                st.success("✅ Data fetched successfully!")
                                st.dataframe(df).head(3)
                            else:
                                st.error("🔴 Invalid ticker or date range")
                        except Exception as e:
                            st.error(f"🔴 API Error: {str(e)}")

    # Data Preprocessing
    if st.session_state.pipeline_step == 1:
        with st.container():
            st.markdown("## 🔓 PHASE 2: DATA PREPARATION")
            
            if st.session_state.raw_df is not None:
                df = st.session_state.raw_df.copy()
                
                # Missing Data Handling
                with st.expander("🔍 MISSING DATA HANDLING"):
                    original_count = len(df)
                    missing = df.isnull().sum()
                    
                    st.write("Missing values per column:")
                    st.write(missing)
                    
                    method = st.radio("HANDLING METHOD:", 
                                    ["🚮 Remove Missing", "📊 Fill with Mean", "📈 Fill with Median"])
                    
                    if st.button("🔧 APPLY METHOD"):
                        if method == "🚮 Remove Missing":
                            df = df.dropna()
                        else:
                            for col in df.select_dtypes(include=np.number):
                                if method == "📊 Fill with Mean":
                                    df[col] = df[col].fillna(df[col].mean())
                                else:
                                    df[col] = df[col].fillna(df[col].median())
                        
                        st.info(f"Removed {original_count - len(df)} rows")
                        if len(df) == 0:
                            st.error("🔴 All data removed during cleaning!")
                            st.stop()
                
                # Outlier Detection
                with st.expander("📊 ANOMALY DETECTION"):
                    cols = st.multiselect("SELECT FEATURES FOR ANALYSIS:", 
                                        df.select_dtypes(include=np.number).columns)
                    
                    original_count = len(df)
                    if cols:
                        for col in cols:
                            q1 = df[col].quantile(0.25)
                            q3 = df[col].quantile(0.75)
                            iqr = q3 - q1
                            lower_bound = q1 - 1.5*iqr
                            upper_bound = q3 + 1.5*iqr
                            
                            if st.button(f"🔪 REMOVE OUTLIERS IN {col}", key=f"outlier_{col}"):
                                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                                
                    if len(df) < original_count * 0.1:
                        st.error("🔴 Over 90% data removed! Check outlier thresholds")
                        st.stop()
                
                if st.button("⚡ FINALIZE PREPROCESSING"):
                    st.session_state.df = df
                    st.session_state.pipeline_step = 2
                    st.success(f"✅ Preprocessing complete! {len(df)} samples remaining")
                    st.dataframe(df.describe())

    # Feature Engineering
    if st.session_state.pipeline_step == 2:
        with st.container():
            st.markdown("## 🔓 PHASE 3: FEATURE ENGINEERING")
            
            df = st.session_state.df.copy()
            try:
                # Feature calculations
                df['ma20'] = df['close'].rolling(20, min_periods=1).mean()
                df['ma50'] = df['close'].rolling(50, min_periods=1).mean()
                delta = df['close'].diff().fillna(0)
                gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
                rs = gain / (loss + 1e-10)
                df['rsi'] = 100 - (100 / (1 + rs))
                
                if model_type != "🌀 Clustering":
                    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
                    df = df.dropna()
                
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                if model_type == "🔮 Classification":
                    numeric_cols.remove('target')
                elif model_type == "📈 Regression":
                    numeric_cols.remove('close')
                
                selected_features = st.multiselect("SELECT FEATURES:", 
                                                 numeric_cols,
                                                 default=numeric_cols)
                
                if len(selected_features) == 0:
                    st.error("🔴 Select at least one feature!")
                    return
                
                if st.button("⚙️ FINALIZE FEATURES"):
                    keep_cols = selected_features + ['target'] if model_type == "🔮 Classification" else selected_features + ['close']
                    st.session_state.df = df[keep_cols]
                    st.session_state.pipeline_step = 3
                    st.success(f"✅ Selected {len(selected_features)} features")
                    
            except Exception as e:
                st.error(f"🔴 Feature engineering error: {str(e)}")
                st.session_state.pipeline_step = 1

    # Data Splitting
    if st.session_state.pipeline_step == 3:
        with st.container():
            st.markdown("## 🔓 PHASE 4: DATA PARTITIONING")
            
            df = st.session_state.df.copy()
            try:
                if model_type != "🌀 Clustering":
                    target = 'target' if model_type == "🔮 Classification" else 'close'
                    test_size = st.slider("TEST SIZE (%)", 10, 40, 20)
                    
                    if len(df) < 20:
                        st.warning("⚠️ Small dataset - consider reducing test size")
                        test_size = min(test_size, 30)
                    
                    X = df.drop(columns=[target])
                    y = df[target]
                    
                    min_test_samples = max(1, int(len(df)*0.05))
                    if (len(df)*test_size/100) < min_test_samples:
                        test_size = max(test_size, int(min_test_samples/len(df)*100))
                        st.warning(f"⚠️ Adjusted test size to {test_size}% for minimum {min_test_samples} samples")
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, 
                        test_size=test_size/100, 
                        random_state=42,
                        shuffle=True
                    )
                    
                    fig = px.pie(names=['Train', 'Test'], 
                                values=[len(X_train), len(X_test)],
                                color_discrete_sequence=['#FF5F15', '#4169E1'])
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    X_train = df
                    X_test = df
                    y_train = pd.Series()
                    y_test = pd.Series()
                
                if st.button("🔀 FINALIZE SPLIT"):
                    st.session_state.update({
                        'X_train': X_train,
                        'X_test': X_test,
                        'y_train': y_train,
                        'y_test': y_test,
                        'pipeline_step': 4
                    })
                    st.success("✅ Data split completed!")
                    
            except Exception as e:
                st.error(f"🔴 Data splitting failed: {str(e)}")
                st.session_state.pipeline_step = 2

    # Model Training
    if st.session_state.pipeline_step == 4:
        with st.container():
            st.markdown("## 🔓 PHASE 5: MODEL TRAINING")
            
            try:
                model_options = {
                    "📈 Regression": ["Linear Regression"],
                    "🔮 Classification": ["Logistic Regression", "Random Forest"],
                    "🌀 Clustering": ["K-Means"]
                }
                
                model_choice = st.selectbox("SELECT MODEL", model_options[model_type])
                
                model = None
                if model_type == "📈 Regression":
                    model = make_pipeline(StandardScaler(), LinearRegression())
                        
                elif model_type == "🔮 Classification":
                    if model_choice == "Logistic Regression":
                        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
                    else:
                        model = make_pipeline(StandardScaler(), RandomForestClassifier())
                        
                else:
                    n_clusters = st.slider("NUMBER OF CLUSTERS", 2, 5, 3)
                    model = make_pipeline(StandardScaler(), KMeans(n_clusters=n_clusters))
                
                if st.button("🧠 TRAIN MODEL"):
                    with st.spinner("Training in progress..."):
                        if model_type != "🌀 Clustering":
                            model.fit(st.session_state.X_train, st.session_state.y_train)
                        else:
                            model.fit(st.session_state.X_train)
                            
                        st.session_state.model = model
                        st.session_state.pipeline_step = 5
                        st.success("✅ Model trained successfully!")
                        
            except Exception as e:
                st.error(f"🔴 Training failed: {str(e)}")
                st.session_state.pipeline_step = 3

    # Model Evaluation
    if st.session_state.pipeline_step == 5:
        with st.container():
            st.markdown("## 🔓 PHASE 6: MODEL EVALUATION")
            
            try:
                model = st.session_state.model
                X_test = st.session_state.X_test
                y_test = st.session_state.y_test
                
                if model_type == "📈 Regression":
                    preds = model.predict(X_test)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, preds)):.2f}")
                        st.metric("R² SCORE", f"{r2_score(y_test, preds):.2f}")
                    with col2:
                        fig = px.scatter(x=y_test, y=preds, 
                                        labels={'x': 'Actual', 'y': 'Predicted'},
                                        trendline="ols",
                                        color_discrete_sequence=['#FF5F15'])
                        st.plotly_chart(fig, use_container_width=True)
                        
                elif model_type == "🔮 Classification":
                    preds = model.predict(X_test)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("ACCURACY", f"{accuracy_score(y_test, preds):.2%}")
                    with col2:
                        fig = px.imshow(confusion_matrix(y_test, preds),
                                      color_continuous_scale=['#121212', '#FF5F15'],
                                      labels=dict(x="Predicted", y="Actual"))
                        st.plotly_chart(fig, use_container_width=True)
                        
                else:
                    score = silhouette_score(X_test, model.predict(X_test))
                    st.metric("SILHOUETTE SCORE", f"{score:.2f}")
                    fig = px.scatter(X_test, color=model.predict(X_test),
                                  color_discrete_sequence=['#FF5F15', '#4169E1', '#00FF00'])
                    st.plotly_chart(fig, use_container_width=True)
                
                st.session_state.pipeline_step = 6
                
            except Exception as e:
                st.error(f"🔴 Evaluation failed: {str(e)}")
                st.session_state.pipeline_step = 4

    # Results Export
    if st.session_state.pipeline_step == 6:
        with st.container():
            st.markdown("## 🔓 PHASE 7: RESULTS EXPORT")
            
            # Add Prison Break tattoo animation
            add_tattoo_animation()
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 DOWNLOAD PREDICTIONS"):
                    try:
                        results = pd.DataFrame({
                            'Actual': st.session_state.y_test,
                            'Predicted': st.session_state.model.predict(st.session_state.X_test)
                        })
                        csv = results.to_csv(index=False)
                        st.download_button("DOWNLOAD CSV", csv, "prison_break_predictions.csv")
                    except Exception as e:
                        st.error(f"🔴 Export failed: {str(e)}")
                        
            with col2:
                if st.button("💾 SAVE MODEL"):
                    try:
                        with BytesIO() as buffer:
                            joblib.dump(st.session_state.model, buffer)
                            st.download_button("DOWNLOAD MODEL", 
                                             buffer.getvalue(), 
                                             "prison_break_model.pkl")
                    except Exception as e:
                        st.error(f"🔴 Model save failed: {str(e)}")
            
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center;">
                <h3 style="color: #FF5F15;">ESCAPE COMPLETE! 🎉</h3>
                <p>You've successfully broken out with your financial predictions!</p>
                <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdWh6ZGRidmRham8wa3cxc2wwZzAzZmxuaXh4bXhqNnU2d2kwYjJteCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/Jo1dQMDNhO7MmFe0x1/giphy.gif">
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()