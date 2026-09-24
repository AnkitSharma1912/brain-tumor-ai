import streamlit as st
import pandas as pd
import numpy as np
import joblib
import nibabel as nib
import os
import tempfile
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

st.set_page_config(page_title="Neuro-Oncology Prognosis AI", layout="wide")
st.title("🧠 PINN-Based Brain Tumor Prognosis AI")
st.markdown("Upload a patient's **Baseline (Scan 1)** and **Follow-up (Scan 2)** 3D MRI masks to predict treatment response.")

class ProgressionCNN3D(nn.Module):
    def __init__(self):
        super(ProgressionCNN3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 8 * 8 * 8, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

@st.cache_resource
def load_ml_models():
    return joblib.load("pinn_scaler.pkl"), joblib.load("pinn_rf_model.pkl")

@st.cache_resource
def load_dl_model():
    model = ProgressionCNN3D()
    model.load_state_dict(torch.load("tumor_cnn_model.pth", map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model

try:
    ml_scaler, ml_model = load_ml_models()
    dl_model = load_dl_model()
except Exception as e:
    st.error(f"Error loading models. Please ensure .pkl and .pth files are uploaded.")
    st.stop()

def get_3d_array(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    data = nib.load(tmp_path).get_fdata()
    os.remove(tmp_path)
    return data

def resize_3d(arr, target_shape=(32, 32, 32)):
    factors = (target_shape[0]/arr.shape[0], target_shape[1]/arr.shape[1], target_shape[2]/arr.shape[2])
    return zoom(arr, factors, order=0)

mode = st.radio("⚙️ Select AI Engine:", ["Classical ML (PINN + Random Forest)", "Deep Learning (PINN + 3D CNN + Explainable AI)"], horizontal=True)

col1, col2 = st.columns(2)
with col1: scan1_file = st.file_uploader("Upload Scan 1 Mask [Baseline]", type=["nii.gz"])
with col2: scan2_file = st.file_uploader("Upload Scan 2 Mask [Follow-up]", type=["nii.gz"])

if scan1_file and scan2_file and st.button("Run AI Analysis", type="primary"):
    with st.spinner("Processing 3D Volumes..."):
        arr1 = get_3d_array(scan1_file)
        arr2 = get_3d_array(scan2_file)
        st.divider()
        
        if "Classical ML" in mode:
            st.subheader("🧬 PINN Biophysical Analysis")
            vol1, vol2 = np.count_nonzero(arr1), np.count_nonzero(arr2)
            growth_speed = (np.log(vol2 + 1) - np.log(vol1 + 1)) / 3.0
            
            patient_scaled = ml_scaler.transform([[vol1, vol2, growth_speed]])
            prediction = ml_model.predict(patient_scaled)[0]
            prob = ml_model.predict_proba(patient_scaled)[0]
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Baseline Volume", f"{vol1:,} voxels")
            c2.metric("Follow-up Volume", f"{vol2:,} voxels")
            c3.metric("Biological Momentum", f"{growth_speed:.3f}")
            
            if prediction == 0: st.success(f"**OUTCOME:** Treatment Responder (Confidence: {prob[0]*100:.1f}%)")
            else: st.error(f"**OUTCOME:** Non-Responder / Progressive Disease (Confidence: {prob[1]*100:.1f}%)")

        else:
            st.subheader("👁️ Deep Learning XAI (Grad-CAM)")
            arr1_res, arr2_res = resize_3d(arr1), resize_3d(arr2)
            delta_map = (arr2_res.astype(np.float32) - arr1_res.astype(np.float32))
            input_tensor = torch.tensor(delta_map).unsqueeze(0).unsqueeze(0)
            
            activations, gradients = None, None
            def fw_hook(m, i, o): global activations; activations = o
            def bw_hook(m, gi, go): global gradients; gradients = go[0]
            h1 = dl_model.conv2.register_forward_hook(fw_hook)
            h2 = dl_model.conv2.register_full_backward_hook(bw_hook)
            
            dl_model.zero_grad()
            pred = dl_model(input_tensor)
            pred.backward()
            h1.remove(); h2.remove()
            
            pred_class = 1 if pred.item() >= 0.5 else 0
            conf = pred.item() if pred_class == 1 else 1 - pred.item()
            
            if pred_class == 0: st.success(f"**OUTCOME:** Treatment Responder (Confidence: {conf*100:.1f}%)")
            else: st.error(f"**OUTCOME:** Non-Responder / Progressive Disease (Confidence: {conf*100:.1f}%)")

            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3, 4])
            for i in range(activations.size(1)): activations[:, i, :, :, :] *= pooled_gradients[i]
            
            heatmap = torch.mean(activations, dim=1).squeeze().detach().numpy()
            heatmap = np.maximum(heatmap, 0)
            if np.max(heatmap) > 0: heatmap /= np.max(heatmap)
            
            mid_slice = 16
            hm_slice = zoom(heatmap[mid_slice//2, :, :], (4, 4), order=1)
            orig_slice = delta_map[mid_slice, :, :]
            
            fig, ax = plt.subplots(1, 3, figsize=(10, 3))
            ax[0].imshow(orig_slice, cmap='gray'); ax[0].set_title("Delta Map"); ax[0].axis('off')
            ax[1].imshow(hm_slice, cmap='jet'); ax[1].set_title("CNN Focus Area"); ax[1].axis('off')
            ax[2].imshow(orig_slice, cmap='gray'); ax[2].imshow(hm_slice, cmap='jet', alpha=0.5)
            ax[2].set_title("XAI Overlay"); ax[2].axis('off')
            st.pyplot(fig)
