# Autoencoders for Image Compression, Denoising & (Optional) Anomaly Detection — 4‑Part Project

## 📚 Dataset
- Dataset link: **https://www.kaggle.com/datasets/jessicali9530/lfw-dataset** 

This project explores **autoencoders**—neural networks that learn to **compress** images into a compact **latent space** and then **reconstruct** them. By optimizing reconstruction quality, autoencoders discover features that capture the essence of each image without labels. We leverage this to solve two practical problems:

1) **Image Denoising:** Train a **denoising autoencoder** to remove noise from images. At training time we corrupt inputs and ask the network to recover the clean version.  
2) **Compact Representations (Compression):** Use the **latent vector (e.g., 256‑D)** as a learned embedding for downstream tasks or efficient storage.  
3) **Anomaly Detection:** If an autoencoder only sees “normal” data, unusual images will reconstruct poorly. We can flag anomalies via **reconstruction error** thresholds.

Across **Parts 1–4**, we progress from a basic autoencoder to stronger **convolutional** and **denoising** variants, then refine the architecture.

---

## 🔎 What’s inside (auto‑detected from the notebooks)

- **Framework:** PyTorch (CUDA supported on your run)
- **Dataset loader:** `ImageFolder` style (provide/train/val/test directories)
- **Architectures:** `ConvAutoencoder`, `ImprovedCAE` (convolutional); also MLP layers appear in early/baseline variants
- **Latent dimension:** ~**256** (in code)
- **Losses used:** **MSE** and **MAE/L1**
- **Optimizer:** **Adam**
- **Artifacts:** model checkpoints are saved (`torch.save(...)`)
- **Dataset link:** _not found in the notebooks_ → add here if public: **<ADD_DATASET_LINK_HERE>**

> If your dataset has a different layout or path variables, update the early cells accordingly.

---

## 📁 Repository Layout
```
.
├── Autoencoders_Part1.ipynb   # Step 1: Baseline AE (MLP/Conv start)
├── Autoencoders_Part2.ipynb   # Step 2: Convolutional Autoencoder (CAE)
├── Autoencoders_Part3.ipynb   # Step 3: Denoising Autoencoder (DAE)
├── Autoencoders_Part4.ipynb   # Step 4: Improved CAE (tuning, deeper, 256‑D latent)
└── README.md                  # This combined guide
```

---

## 🧰 Environment & Setup

**Dependencies:** `torch`, `torchvision`, `pillow`, `numpy`, `matplotlib`, `scikit-learn`

Install:
```bash
pip install -U torch torchvision torchaudio pillow numpy matplotlib scikit-learn
```

**Data layout (example):**
```
data/
├── train/
│   ├── class_0/ ... class_K/
├── val/
│   ├── class_0/ ... class_K/
└── test/
    ├── class_0/ ... class_K/
```
> For purely **unsupervised** training, class folders are optional; the model uses images only.

---

## ✅ Step 1 — Baseline Autoencoder (Part 1)
**Notebook:** `Autoencoders_Part1.ipynb`

- Build a minimal autoencoder and verify the training loop.  
- Start with simple **MLP** layers or a shallow **Conv** encoder/decoder.  
- Train on clean images; visualize **reconstructions**.

Run:
```bash
jupyter notebook "Autoencoders_Part1.ipynb"
```

---

## 🧱 Step 2 — Convolutional Autoencoder (Part 2)
**Notebook:** `Autoencoders_Part2.ipynb`

- Switch to a **Convolutional Autoencoder (CAE)** to better capture spatial structure.  
- Use downsampling (Conv/Pool/Stride) → **latent (≈256‑D)** → upsampling (ConvTranspose/Interpolate).  
- Evaluate reconstruction quality and training curves.

Run:
```bash
jupyter notebook "Autoencoders_Part2.ipynb"
```

---

## 🧽 Step 3 — Denoising Autoencoder (Part 3)
**Notebook:** `Autoencoders_Part3.ipynb`

- Corrupt inputs with Gaussian noise (`noise_factor`) or similar; targets remain **clean**.  
- Train the network to **remove noise**; compare PSNR/visual quality qualitatively (and MSE/MAE numerically).  
- Save sample **before/after** images.

Run:
```bash
jupyter notebook "Autoencoders_Part3.ipynb"
```

---

## 🔧 Step 4 — Improved CAE (Part 4)
**Notebook:** `Autoencoders_Part4.ipynb`

- Refine the architecture (**`ImprovedCAE`**): deeper encoder/decoder, **BatchNorm/Dropout** if present, and a **256‑D latent**.  
- Try **MSE** vs **MAE** objectives; keep **Adam** as the optimizer (or experiment).  
- Export the best checkpoint (`torch.save`) and visualize final reconstructions.

Run:
```bash
jupyter notebook "Autoencoders_Part4.ipynb"
```

---

## 📈 Evaluation & Reporting

Common measurements:
- **Reconstruction loss** on **val/test** (MSE or MAE).
- **Qualitative** grids: original vs reconstruction (and noised input vs denoised output).

> The notebooks don’t print a single “final score” across parts; after you run them, copy key numbers or images into this README.

---

## 🚨 (Optional) Anomaly Detection via Reconstruction Error

Train the autoencoder on **normal** data only. At test time:
1) Reconstruct each image.
2) Compute an error metric (e.g., **MSE** per image).
3) Choose a threshold (e.g., 95th percentile of training errors).  
4) Flag samples whose error exceeds the threshold as **anomalies**.

Minimal PyTorch snippet:
```python
import torch, torch.nn.functional as F

# x: [N, C, H, W] batch of images
with torch.no_grad():
    recon = model(x)
    per_sample_mse = F.mse_loss(recon, x, reduction="none").view(x.size(0), -1).mean(dim=1)

threshold = torch.quantile(per_sample_mse, 0.95)  # set using train/val stats
pred_is_anomaly = per_sample_mse > threshold
```

---

## 🚀 Inference / Embedding Extraction

To get the **latent code** (useful as a learned embedding):
```python
model.eval()
with torch.no_grad():
    z = model.encoder(x)   # shape: [N, latent_dim]  (≈ 256 from these notebooks)
```

To denoise:
```python
model.eval()
with torch.no_grad():
    x_denoised = model(x_noisy)
```

---

## 🧭 Tips & Next Steps
- Try different **latent sizes** (128, 256, 512) and compare reconstruction quality.  
- Add **BatchNorm/Dropout** in the encoder/decoder for stability/regularization.  
- Experiment with **schedulers** and **weight decay**; early stopping around the validation loss can help.  
- For richer structure, move to **U‑Net** style decoders or **skip connections**.  
- If you want a probabilistic latent space, extend to a **Variational Autoencoder (VAE)** (add `mu`, `logvar`, and a **KLD** term).

---

## 📚 Dataset & Reproducibility
- Dataset link: **https://www.kaggle.com/datasets/jessicali9530/lfw-dataset** 
- Set seeds for Python/NumPy/PyTorch for repeatable results.  
- Keep a simple result table and sample grids in an `images/` folder.
