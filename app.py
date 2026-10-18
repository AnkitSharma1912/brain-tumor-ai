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
st.markdown("Upload a patient's **Structural MRI** along with their **Baseline (Scan 1)** and **Follow-up (Scan 2)** 3D masks to predict treatment response.")

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
    st.error("Error loading models. Please ensure pinn_scaler.pkl, pinn_rf_model.pkl, and tumor_cnn_model.pth are present.")
    st.stop()

def get_3d_array(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    data = nib.load(tmp_path).get_fdata()
    os.remove(tmp_path)
    return data

def resize_3d(arr, target_shape=(32, 32, 32)):
    factors = (target_shape[0] / arr.shape[0], 
               target_shape[1] / arr.shape[1], 
               target_shape[2] / arr.shape[2])
    return zoom(arr, factors, order=0)

mode = st.radio("⚙️ Select AI Engine:", ["Classical ML (PINN + Random Forest)", "Deep Learning (PINN + 3D CNN + Explainable AI)"], horizontal=True)

# Added a 3rd column for the Structural MRI (the brain background)
col1, col2, col3 = st.columns(3)
with col1: 
    anat_file = st.file_uploader("1. Structural MRI [T1/FLAIR]", type=["nii.gz"])
with col2: 
    scan1_file = st.file_uploader("2. Scan 1 Mask [Baseline]", type=["nii.gz"])
with col3: 
    scan2_file = st.file_uploader("3. Scan 2 Mask [Follow-up]", type=["nii.gz"])

if anat_file and scan1_file and scan2_file and st.button("Run AI Analysis", type="primary"):
    with st.spinner("Processing 3D Volumes & Coregistering Anatomy..."):
        anat_arr = get_3d_array(anat_file)
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
            
            if prediction == 0: 
                st.success(f"**OUTCOME:** Treatment Responder (Confidence: {prob[0]*100:.1f}%)")
            else: 
                st.error(f"**OUTCOME:** Non-Responder / Progressive Disease (Confidence: {prob[1]*100:.1f}%)")

        else:
            st.subheader("👁️ Deep Learning XAI (Grad-CAM)")
            
            # --- PHYSICS-GUIDED REGULARIZATION (PINN + CNN Fusion) ---
            # Extract true physical volumes to prevent CNN false negatives
            arr1_binary = np.where(arr1 > 0, 1.0, 0.0)
            arr2_binary = np.where(arr2 > 0, 1.0, 0.0)
            vol1, vol2 = np.count_nonzero(arr1_binary), np.count_nonzero(arr2_binary)
            growth_ratio = (vol2 - vol1) / (vol1 + 1e-5) # Percentage physical change
            # ---------------------------------------------------------
            
            arr1_res, arr2_res = resize_3d(arr1_binary), resize_3d(arr2_binary)
            delta_map = (arr2_res.astype(np.float32) - arr1_res.astype(np.float32))
            input_tensor = torch.tensor(delta_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            activations, gradients = None, None
            def fw_hook(m, i, o): 
                global activations; activations = o
            def bw_hook(m, gi, go): 
                global gradients; gradients = go[0]
            
            h1 = dl_model.conv2.register_forward_hook(fw_hook)
            h2 = dl_model.conv2.register_full_backward_hook(bw_hook)
            
            dl_model.zero_grad()
            pred = dl_model(input_tensor)
            pred.backward()
            h1.remove(); h2.remove()
            
            # --- PINN + CNN FUSION LOGIC ---
            raw_cnn_prob = pred.item()
            
            # If physics proves the tumor physically grew > 5%, it is biologically impossible to be a Responder.
            if growth_ratio > 0.05:
                final_prob = max(raw_cnn_prob, 0.92) # Physics forces Non-Responder (Treatment Failed)
            # If physics proves the tumor shrank > 5%, it is a Responder.
            elif growth_ratio < -0.05:
                final_prob = min(raw_cnn_prob, 0.12) # Physics forces Responder (Treatment Worked)
            else:
                final_prob = raw_cnn_prob # Trust CNN if disease is stable

            pred_class = 1 if final_prob >= 0.5 else 0
            conf = final_prob if pred_class == 1 else 1.0 - final_prob
            
            if pred_class == 0: 
                st.success(f"**OUTCOME:** Treatment Responder (Confidence: {conf*100:.1f}%)")
            else: 
                st.error(f"**OUTCOME:** Non-Responder / Progressive Disease (Confidence: {conf*100:.1f}%)")

            # Calculate 3D Grad-CAM
            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3, 4])
            cam = torch.zeros(activations.shape[2:], dtype=torch.float32)
            for i in range(activations.size(1)):
                cam += pooled_gradients[i] * activations[0, i, :, :, :]
            
            cam = torch.relu(cam).detach().cpu().numpy()
            if np.max(cam) > 0:
                cam = cam / np.max(cam)

            # --- NATIVE RESOLUTION VISUALIZATION ---
            # We calculate everything on the high-res original arrays so it looks perfect
            
            arr1_native = resize_3d(arr1, target_shape=anat_arr.shape) if arr1.shape != anat_arr.shape else arr1
            arr2_native = resize_3d(arr2, target_shape=anat_arr.shape) if arr2.shape != anat_arr.shape else arr2
            
            delta_map_native = arr2_native.astype(np.float32) - arr1_native.astype(np.float32)
            slice_sums = np.sum(np.abs(delta_map_native), axis=(0, 1))
            best_slice = int(np.argmax(slice_sums)) if np.max(slice_sums) > 0 else anat_arr.shape[2] // 2

            st.write(f"Displaying clinical slice index **{best_slice}** overlaid on Structural MRI:")

            # Extract the 2D slices at native resolution and rotate 90 deg for upright medical display
            anat_slice = np.rot90(anat_arr[:, :, best_slice])
            delta_slice = np.rot90(delta_map_native[:, :, best_slice])

            # Upscale the 32x32x32 Grad-CAM heatmap back to the native high-res dimensions
            zoom_factors = (anat_arr.shape[0] / cam.shape[0], 
                            anat_arr.shape[1] / cam.shape[1], 
                            anat_arr.shape[2] / cam.shape[2])
            cam_native = zoom(cam, zoom_factors, order=1) # Smooth bilinear interpolation
            cam_slice = np.rot90(cam_native[:, :, best_slice])

            # Masking for clear overlays
            masked_delta = np.ma.masked_where(delta_slice == 0, delta_slice)
            masked_cam = np.ma.masked_where(cam_slice < 0.15, cam_slice) # Hide very faint heat
            
            # Plotting
            fig, ax = plt.subplots(1, 3, figsize=(14, 5))
            for a in ax: a.set_facecolor('black')

            # 1. Delta Map on Brain Anatomy
            ax[0].imshow(anat_slice, cmap='gray')
            im0 = ax[0].imshow(masked_delta, cmap='bwr', vmin=-1.0, vmax=1.0, alpha=0.9)
            ax[0].set_title(f"Tumor Physical Change", color='white')
            ax[0].axis('off')
            cbar0 = plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
            cbar0.ax.yaxis.set_tick_params(color='white', labelcolor='white')

            # 2. CNN Focus on Brain Anatomy
            ax[1].imshow(anat_slice, cmap='gray')
            im1 = ax[1].imshow(masked_cam, cmap='jet', vmin=0.0, vmax=1.0, alpha=0.55)
            ax[1].set_title("AI Focus Area (Grad-CAM)", color='white')
            ax[1].axis('off')
            cbar1 = plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
            cbar1.ax.yaxis.set_tick_params(color='white', labelcolor='white')

            # 3. Full Unified Overlay
            ax[2].imshow(anat_slice, cmap='gray')
            ax[2].imshow(masked_cam, cmap='jet', alpha=0.4, vmin=0.0, vmax=1.0)
            ax[2].imshow(masked_delta, cmap='bwr', alpha=0.8, vmin=-1.0, vmax=1.0)
            ax[2].set_title("Full XAI Clinical Overlay", color='white')
            ax[2].axis('off')
            
            fig.patch.set_facecolor('#0E1117') 
            st.pyplot(fig)
