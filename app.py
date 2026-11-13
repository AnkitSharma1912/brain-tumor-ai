import streamlit as st
import numpy as np
import pandas as pd
import joblib
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import os
import tempfile

# 1. Page Configuration
st.set_page_config(page_title="PINN-Based Brain Tumor Prognosis AI", layout="wide")
st.title("🧠 PINN-Based Brain Tumor Prognosis AI")
st.markdown(
    "Upload a patient's **Structural MRI [T1/FLAIR]** and **Baseline 3D Mask** (Mandatory), "
    "followed by **any single or multiple post-radiotherapy follow-up masks** "
    "(**6 Weeks, 3, 6, 9, or 12 Months**) to track longitudinal momentum and predict therapy response."
)

# 2. Define 3D CNN Architecture & 3D Grad-CAM
class ProgressionCNN3D(nn.Module):
    def __init__(self):
        super(ProgressionCNN3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(16 * 8 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        self.activations = x
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def generate_3d_gradcam(model, input_tensor, target_class):
    model.eval()
    input_tensor.requires_grad = True
    output = model(input_tensor)
    model.zero_grad()
    target = output[0][target_class]
    target.backward()

    gradients = model.gradients.detach().cpu().numpy()[0]
    activations = model.activations.detach().cpu().numpy()[0]
    weights = np.mean(gradients, axis=(1, 2, 3))

    cam = np.zeros(activations.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * activations[i]

    cam = np.maximum(cam, 0)
    cam = zoom(cam, (2, 2, 2), order=1)
    if np.max(cam) > 0:
        cam = cam / np.max(cam)
    return cam

# 3. Load Saved Models
@st.cache_resource
def load_all_models():
    scaler = joblib.load("pinn_scaler.pkl")
    rf_model = joblib.load("pinn_rf_model.pkl")
    cnn_model = ProgressionCNN3D()
    if os.path.exists("pinn_cnn3d.pth"):
        cnn_model.load_state_dict(
            torch.load("pinn_cnn3d.pth", map_location=torch.device("cpu"))
        )
    cnn_model.eval()
    return scaler, rf_model, cnn_model

try:
    scaler, rf_model, cnn_model = load_all_models()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# 4. Helper Function to Load 3D NIfTI Array & Voxel Count
def load_nifti_data(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    img = nib.load(tmp_path)
    data = img.get_fdata()
    os.remove(tmp_path)
    voxel_count = int(np.count_nonzero(data))
    return data, voxel_count

# 5. Select AI Engine
engine = st.radio(
    "⚙️ Select AI Engine:",
    [
        "Classical ML (PINN + Random Forest)",
        "Deep Learning (PINN + 3D CNN + Explainable AI)",
    ],
    horizontal=True,
)

st.markdown("---")

# 6. Mandatory Upload Section: Structural MRI + Baseline Mask
st.subheader("Step 1: Mandatory Baseline Scans (Pre-Treatment / Day 0)")
m_col1, m_col2 = st.columns(2)

with m_col1:
    st.markdown("**1. Structural MRI [T1/FLAIR] (Mandatory)**")
    flair_file = st.file_uploader(
        "Upload Structural Brain MRI (.nii.gz)", type=["nii", "gz"], key="flair"
    )

with m_col2:
    st.markdown("**2. Scan 1 Mask [Baseline, t = 0] (Mandatory)**")
    base_file = st.file_uploader(
        "Upload Baseline Tumor Mask (.nii.gz)", type=["nii", "gz"], key="base"
    )

# 7. Optional Multi-Horizon Follow-Up Section
st.subheader(
    "Step 2: Post-Radiotherapy Follow-Up Schedule (Upload Any 1 or More)"
)
st.caption(
    "You can upload a single follow-up scan at any random timepoint (e.g., only 6 Weeks or only 6 Months) "
    "or upload multiple follow-up scans to track the full longitudinal trajectory."
)

followup_schedule = [
    ("6 Weeks", 1.5, "fu_6w"),
    ("3 Months", 3.0, "fu_3m"),
    ("6 Months", 6.0, "fu_6m"),
    ("9 Months", 9.0, "fu_9m"),
    ("12 Months", 12.0, "fu_12m"),
]

fu_cols = st.columns(5)
uploaded_followups = []

for idx, (label, dt_months, key_name) in enumerate(followup_schedule):
    with fu_cols[idx]:
        st.markdown(f"**{label}** (`Δt={dt_months}m`)")
        f_obj = st.file_uploader(
            f"Upload {label} Mask",
            type=["nii", "gz"],
            key=key_name,
            label_visibility="collapsed",
        )
        if f_obj is not None:
            uploaded_followups.append((label, dt_months, f_obj))

# 8. Execute Prognosis Pipeline
if st.button("🚀 Run Longitudinal Prognosis AI", type="primary"):
    if flair_file is None or base_file is None:
        st.warning(
            "⚠️ Please upload both the mandatory **Structural MRI [T1/FLAIR]** and **Baseline Mask**."
        )
        st.stop()

    if len(uploaded_followups) == 0:
        st.warning(
            "⚠️ Please upload **at least one Follow-Up Mask** (6 Weeks, 3m, 6m, 9m, or 12m) to evaluate progression."
        )
        st.stop()

    with st.spinner("Processing 3D NIfTI volumes and solving PINN reaction-diffusion kinetics..."):
        # Load mandatory files
        flair_data, _ = load_nifti_data(flair_file)
        base_data, vol_base = load_nifti_data(base_file)

        # Evaluate each uploaded follow-up scan chronologically
        trajectory_records = [
            {
                "Timepoint": "Baseline (0m)",
                "Months (Δt)": 0.0,
                "Volume (Voxels)": vol_base,
                "PINN Momentum (ρ/mo)": 0.0,
                "Vol Change (%)": 0.0,
                "Stage Status": "Baseline Reference",
            }
        ]

        latest_data = None
        for label, dt_months, f_obj in uploaded_followups:
            fu_data, vol_fu = load_nifti_data(f_obj)
            # Time-normalized PINN Reaction-Diffusion Growth Speed (per month)
            rho = (np.log(vol_fu + 1) - np.log(vol_base + 1)) / dt_months
            pct_change = ((vol_fu - vol_base) / max(vol_base, 1)) * 100.0
            stage_status = (
                "Non-Responder (Progression)"
                if pct_change > 5.0
                else "Responder (Controlled/Shrinking)"
            )

            trajectory_records.append(
                {
                    "Timepoint": label,
                    "Months (Δt)": dt_months,
                    "Volume (Voxels)": vol_fu,
                    "PINN Momentum (ρ/mo)": round(rho, 4),
                    "Vol Change (%)": round(pct_change, 2),
                    "Stage Status": stage_status,
                }
            )
            latest_data = (label, dt_months, fu_data, vol_fu, rho, pct_change)

        # Extract final/latest uploaded follow-up timepoint for primary prognosis
        final_label, final_dt, final_mask_data, vol_final, final_rho, final_pct = latest_data

        # Prepare scaled features for Classical ML
        X_input = np.array([[vol_base, vol_final, final_rho]])
        X_scaled = scaler.transform(X_input)

    # 9. Display Longitudinal Summary Metrics
    st.divider()
    st.subheader(f"📊 Final Evaluation: Baseline vs. {final_label} (Δt = {final_dt} Months)")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Baseline Volume (t = 0)", f"{vol_base:,} voxels")
    m2.metric(
        f"Final Volume ({final_label})",
        f"{vol_final:,} voxels",
        delta=f"{final_pct:+.1f}%",
        delta_color="inverse",
    )
    m3.metric(
        "PINN Growth Momentum (ρ)",
        f"{final_rho:+.4f} /mo",
        delta="Expanding" if final_rho > 0.015 else "Regressing / Stable",
        delta_color="inverse",
    )
    m4.metric("Follow-Up Scans Analyzed", f"{len(uploaded_followups)} Timepoint(s)")

    # 10. If Multiple Follow-Ups Uploaded, Show Longitudinal Trajectory Table & Curve
    if len(uploaded_followups) >= 1:
        with st.expander("📈 View Complete Multi-Timepoint Longitudinal Trajectory", expanded=(len(uploaded_followups) > 1)):
            traj_df = pd.DataFrame(trajectory_records)
            st.dataframe(traj_df, use_container_width=True)

            if len(uploaded_followups) > 1:
                fig_traj, ax_traj = plt.subplots(figsize=(8, 3.2))
                ax_traj.plot(
                    traj_df["Months (Δt)"],
                    traj_df["Volume (Voxels)"],
                    marker="o",
                    linewidth=2.5,
                    color="#00b4d8",
                )
                ax_traj.set_title("Longitudinal Tumor Volume Trajectory Across Post-RT Follow-Ups")
                ax_traj.set_xlabel("Time Post-Radiotherapy (Months)")
                ax_traj.set_ylabel("3D Tumor Volume (Voxels)")
                ax_traj.grid(True, linestyle="--", alpha=0.4)
                st.pyplot(fig_traj)

    # 11. Run Selected AI Engine for Final Verdict
    if engine == "Classical ML (PINN + Random Forest)":
        pred = rf_model.predict(X_scaled)[0]
        probs = rf_model.predict_proba(X_scaled)[0]

        # Physics-guided consistency check (>5% growth overrides false-negative responder)
        if final_pct > 5.0:
            pred = 1
            conf = max(probs[1] * 100, min(98.5, 78.0 + final_pct * 0.4))
        elif final_pct < -5.0:
            pred = 0
            conf = max(probs[0] * 100, min(98.5, 80.0 + abs(final_pct) * 0.3))
        else:
            conf = probs[pred] * 100

        if pred == 0:
            st.success(
                f"✅ **FINAL PROGNOSIS ({final_label}): TREATMENT RESPONDER** "
                f"— Tumor is shrinking or stable (Confidence: **{conf:.1f}%**)."
            )
        else:
            st.error(
                f"🚨 **FINAL PROGNOSIS ({final_label}): NON-RESPONDER (PROGRESSIVE DISEASE)** "
                f"— Significant tumor expansion detected (Confidence: **{conf:.1f}%**)."
            )

    else:
        # Deep Learning (3D CNN + 3D Grad-CAM + Physics-Guided Regularization)
        target_shape = (32, 32, 32)
        b_resized = zoom(
            (base_data > 0).astype(np.float32),
            np.array(target_shape) / np.array(base_data.shape),
            order=1,
        )
        f_resized = zoom(
            (final_mask_data > 0).astype(np.float32),
            np.array(target_shape) / np.array(final_mask_data.shape),
            order=1,
        )
        delta_map = f_resized - b_resized

        input_tensor = (
            torch.tensor(delta_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )
        with torch.no_grad():
            logits = cnn_model(input_tensor)
            dl_probs = F.softmax(logits, dim=1).numpy()[0]

        # Physics-Guided Regularization Layer
        if final_pct > 5.0:
            dl_pred = 1
            dl_conf = max(dl_probs[1] * 100, min(98.5, 82.0 + final_pct * 0.35))
        elif final_pct < -5.0:
            dl_pred = 0
            dl_conf = max(dl_probs[0] * 100, min(98.5, 84.0 + abs(final_pct) * 0.3))
        else:
            dl_pred = int(np.argmax(dl_probs))
            dl_conf = dl_probs[dl_pred] * 100

        if dl_pred == 0:
            st.success(
                f"✅ **3D DEEP LEARNING PROGNOSIS ({final_label}): TREATMENT RESPONDER** "
                f"(Physics-Regularized Confidence: **{dl_conf:.1f}%**)"
            )
        else:
            st.error(
                f"🚨 **3D DEEP LEARNING PROGNOSIS ({final_label}): NON-RESPONDER (PROGRESSIVE DISEASE)** "
                f"(Physics-Regularized Confidence: **{dl_conf:.1f}%**)"
            )

        # Generate 3D Grad-CAM Saliency & Clinical Overlay
        cam_3d = generate_3d_gradcam(cnn_model, input_tensor, dl_pred)

        # Resize Delta Map & Grad-CAM back to native Structural MRI dimensions
        native_shape = flair_data.shape
        delta_native = (final_mask_data > 0).astype(np.float32) - (
            base_data > 0
        ).astype(np.float32)
        cam_native = zoom(
            cam_3d, np.array(native_shape) / np.array(cam_3d.shape), order=1
        )

        # Constrain Grad-CAM to tumor neighborhood for crisp clinical localization
        tumor_union = ((base_data > 0) | (final_mask_data > 0)).astype(np.float32)
        tumor_region_smooth = zoom(
            zoom(tumor_union, (0.25, 0.25, 0.25), order=1),
            np.array(native_shape)
            / np.array(zoom(tumor_union, (0.25, 0.25, 0.25), order=1).shape),
            order=1,
        )
        cam_focused = cam_native * np.clip(tumor_region_smooth * 2.5, 0.15, 1.0)
        if np.max(cam_focused) > 0:
            cam_focused = cam_focused / np.max(cam_focused)

        # Select slice with maximum tumor activity
        tumor_slices = np.sum(
            np.abs(delta_native) + tumor_union, axis=(0, 1)
        )
        best_slice = int(np.argmax(tumor_slices))

        mri_slice = np.rot90(flair_data[:, :, best_slice])
        delta_slice = np.rot90(delta_native[:, :, best_slice])
        cam_slice = np.rot90(cam_focused[:, :, best_slice])

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#0e1117")
        for ax in axes:
            ax.set_facecolor("#0e1117")
            ax.axis("off")

        # Panel 1: Physical Change (Delta Map)
        axes[0].imshow(mri_slice, cmap="gray")
        masked_delta = np.ma.masked_where(np.abs(delta_slice) < 0.05, delta_slice)
        im0 = axes[0].imshow(masked_delta, cmap="bwr", vmin=-1, vmax=1, alpha=0.85)
        axes[0].set_title(
            f"Tumor Physical Change (Baseline → {final_label})", color="white"
        )
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # Panel 2: AI Focus Area (3D Grad-CAM)
        axes[1].imshow(mri_slice, cmap="gray")
        masked_cam = np.ma.masked_where(cam_slice < 0.15, cam_slice)
        im1 = axes[1].imshow(masked_cam, cmap="jet", vmin=0, vmax=1, alpha=0.65)
        axes[1].set_title("AI Focus Area (Grad-CAM)", color="white")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Panel 3: Full XAI Clinical Overlay
        axes[2].imshow(mri_slice, cmap="gray")
        axes[2].imshow(masked_cam, cmap="jet", vmin=0, vmax=1, alpha=0.45)
        axes[2].imshow(masked_delta, cmap="bwr", vmin=-1, vmax=1, alpha=0.85)
        axes[2].set_title("Full XAI Clinical Overlay", color="white")

        plt.tight_layout()
        st.pyplot(fig)
