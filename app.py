import streamlit as st
import pandas as pd
import numpy as np
import joblib
import nibabel as nib
import os
import tempfile

# 1. Configure the Web Page
st.set_page_config(page_title="Neuro-Oncology Prognosis AI", layout="wide")
st.title("🧠 PINN-Powered Brain Tumor Prognosis")
st.markdown("Upload a patient's **Baseline (Scan 1)** and **Follow-up (Scan 2)** 3D MRI masks to predict treatment response.")

# 2. Load the Saved Machine Learning Models
@st.cache_resource
def load_models():
    scaler = joblib.load("pinn_scaler.pkl")
    model = joblib.load("pinn_rf_model.pkl")
    return scaler, model

try:
    scaler, model = load_models()
except FileNotFoundError:
    st.error("Model files not found! Ensure 'pinn_scaler.pkl' and 'pinn_rf_model.pkl' are in the same folder as this script.")
    st.stop()

# 3. Helper Function to Calculate Volume from 3D NIfTI Files
def calculate_volume(uploaded_file):
    # Save uploaded file to a temporary location to read with nibabel
    with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    # Load image and count tumor voxels (non-zero pixels)
    img = nib.load(tmp_path)
    data = img.get_fdata()
    voxel_count = np.count_nonzero(data)
    
    os.remove(tmp_path) # Clean up
    return voxel_count

# 4. Clinical User Interface: File Uploaders
col1, col2 = st.columns(2)

with col1:
    st.subheader("1️⃣ Baseline Scan (Pre-Treatment)")
    scan1_file = st.file_uploader("Upload Scan 1 Mask (.nii.gz)", type=["nii.gz"], key="scan1")

with col2:
    st.subheader("2️⃣ Follow-up Scan (Post-Treatment)")
    scan2_file = st.file_uploader("Upload Scan 2 Mask (.nii.gz)", type=["nii.gz"], key="scan2")

# 5. Pipeline Execution
if scan1_file and scan2_file:
    if st.button("Run Prognosis AI", type="primary"):
        with st.spinner("Calculating 3D volumes and biological momentum..."):
            # Step A: Extract Physical Volumes
            vol1 = calculate_volume(scan1_file)
            vol2 = calculate_volume(scan2_file)
            
            # Step B: Calculate PINN Biological Momentum
            growth_speed = (np.log(vol2 + 1) - np.log(vol1 + 1)) / 3.0
            
            # Step C: Prepare Data for the Model
            patient_data = np.array([[vol1, vol2, growth_speed]])
            patient_scaled = scaler.transform(patient_data)
            
            # Step D: Predict Outcome
            prediction = model.predict(patient_scaled)[0]
            probability = model.predict_proba(patient_scaled)[0]
            
        # 6. Display Clinical Results
        st.divider()
        st.subheader("🧬 Clinical Analysis Results")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Baseline Volume", f"{vol1:,} voxels")
        res_col2.metric("Follow-up Volume", f"{vol2:,} voxels")
        res_col3.metric("PINN Growth Momentum", f"{growth_speed:.3f}", delta=f"{growth_speed:.3f}", delta_color="inverse")
        
        st.markdown("### 🎯 AI Prognosis")
        if prediction == 0:
            st.success(f"**POSITIVE OUTCOME:** The patient is classified as a **Treatment Responder**.")
            st.info(f"Model Confidence: {probability[0]*100:.1f}%")
        else:
            st.error(f"**NEGATIVE OUTCOME:** The patient is classified as a **Non-Responder (Progressive Disease)**.")
            st.warning(f"Model Confidence: {probability[1]*100:.1f}%")
