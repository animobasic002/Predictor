import pandas as pd
import numpy as np
from scipy import stats
# Removed sklearn to avoid Application Control DLL blocks

def generate_amino_acid_data():
    amino_acids = ['Glycine', 'Alanine', 'Valine', 'Leucine', 'Proline', 'Serine', 'Threonine', 'Arginine']
    data = []
    for aa in amino_acids:
        data.append({
            'Amino Acid': aa,
            'Solubility Enhancement Factor': round(np.random.uniform(1.0, 5.0), 2),
            'Stability Score (0-100)': round(np.random.uniform(50, 100), 1),
            'pH Buffer Capacity': round(np.random.uniform(0.1, 1.0), 2)
        })
    df = pd.DataFrame(data)
    df.loc[df['Amino Acid'] == 'Arginine', 'Stability Score (0-100)'] = 95.5
    return df

def generate_qbd_formulations():
    formulations = []
    for i in range(1, 10): 
        formulations.append({
            'Formulation ID': f'F{i}',
            'Arginine (%)': round(np.random.uniform(1, 10), 1),
            'Binder (%)': round(np.random.uniform(2, 5), 1),
            'Dissolution (%)': round(np.random.uniform(80, 100), 1),
            'Degradation (%)': round(np.random.uniform(0.1, 1.5), 2)
        })
    return pd.DataFrame(formulations)

def generate_stability_data(formulation_id):
    months = [0, 1, 3, 6]
    is_failing = formulation_id in ['F1', 'F4', 'F7']
    
    data = []
    base_assay = 99.5
    base_degradation = 0.1
    
    for m in months:
        if is_failing:
            assay = base_assay - (m * np.random.uniform(1.0, 2.0))
            degradation = base_degradation + (m * np.random.uniform(0.4, 0.8))
        else:
            assay = base_assay - (m * np.random.uniform(0.1, 0.3))
            degradation = base_degradation + (m * np.random.uniform(0.05, 0.15))
            
        data.append({
            'Timepoint (Months)': m,
            'Formulation ID': formulation_id,
            'Assay (%)': round(max(0, assay), 1),
            'Degradation (%)': round(degradation, 2)
        })
    return pd.DataFrame(data)

def calculate_shelf_life(df, parameter='Assay (%)', limit=90.0, is_decreasing=True):
    x = df['Timepoint (Months)'].values
    y = df[parameter].values
    
    if len(x) < 3:
        return None, None
        
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    n = len(x)
    t_val = stats.t.ppf(0.95, n-2) # 95% one-sided confidence
    
    x_line = np.linspace(0, 36, 100)
    y_line = intercept + slope * x_line
    
    mean_x = np.mean(x)
    ss_x = np.sum((x - mean_x)**2)
    s_y_x = np.sqrt(np.sum((y - (intercept + slope * x))**2) / (n - 2))
    margin_error = t_val * s_y_x * np.sqrt(1/n + (x_line - mean_x)**2 / ss_x)
    
    if is_decreasing:
        bound = y_line - margin_error
        cross_idx = np.where(bound < limit)[0]
    else:
        bound = y_line + margin_error
        cross_idx = np.where(bound > limit)[0]
        
    shelf_life = x_line[cross_idx[0]] if len(cross_idx) > 0 else 36.0
    
    ci_data = pd.DataFrame({
        'Time': x_line,
        'Predicted': y_line,
        'Bound': bound
    })
    
    return round(shelf_life, 1), ci_data

def calculate_arrhenius(k_25, k_30, k_40):
    # Temperatures in Kelvin
    T = np.array([25 + 273.15, 30 + 273.15, 40 + 273.15])
    k_vals = np.array([k_25, k_30, k_40])
    
    inv_T = 1 / T
    ln_k = np.log(k_vals)
    
    slope, intercept, _, _, _ = stats.linregress(inv_T, ln_k)
    
    R = 8.314 # J/(mol*K)
    Ea = -slope * R / 1000 # kJ/mol
    
    # Generate line data
    inv_T_line = np.linspace(min(inv_T)*0.98, max(inv_T)*1.02, 50)
    ln_k_line = intercept + slope * inv_T_line
    
    return round(Ea, 2), inv_T, ln_k, inv_T_line, ln_k_line

class MockRandomForest:
    def __init__(self):
        self.feature_importances_ = [0.65, 0.25, 0.10]
        
    def predict(self, df):
        # Risk is high (1) if Moisture is high AND Arginine is low
        risk = ((df['Moisture (%)'] > 2.5) & (df['Arginine (%)'] < 5.0)).astype(int)
        return risk.values

def train_ml_model():
    return MockRandomForest()
