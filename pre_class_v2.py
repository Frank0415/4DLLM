"""
Interactive 4DSTEM clustering for directories with UNDO feature.
Processes .npy files sequentially. For each file, it clusters, then prompts
the user for semantic labels via CLI. The user can type '-1' to undo the
previous label. Finally, it generates a labeled map AND overwrites the 'class'
field in the original .npy file.
"""
import os, math, platform, subprocess
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn.functional as F

# =========================
#           CONFIG
# =========================
OBJECT_NAME = 'Standard\Au_Ag'
NPY_DIR = rf"E:\zhr\Multi4D_v2\Final_version\{OBJECT_NAME}\dataset\exp"  # <--- SET YOUR INPUT DIRECTORY
OUT_ROOT = rf"E:\zhr\Multi4D_v2\Final_version\{OBJECT_NAME}\results\classification_maps"  # <--- SET YOUR OUTPUT DIRECTORY

BATCH = 1024
NBINS = 4
HARMONICS = (2, 4, 6, 8, 10, 12)
RAD_BANDS = 6
CENTER_MASK_RADIUS = 8
RMAX = 111.5
NTHETA = 360
SEED = 0

K = 16
KMEANS_ITERS = 50
INIT_RESTARTS = 2

SAVE_MONTAGE = True
MONTAGE_GRID = (10, 10)
MAX_EMB_POINTS = 65536

CATEGORIES = ["empty", "amorphous", "crystalline", "mixing"]
FINAL_COLOR_MAP = {
    "empty": '#440154', "amorphous": '#3b528b',
    "crystalline": '#21908d', "mixing": '#5dc863',
}


# =========================
#         HELPERS
# =========================
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def stemmed_outdir(npy_path: Path, out_root: str) -> Path:
    stem = npy_path.stem
    out = Path(out_root) / stem
    out.mkdir(parents=True, exist_ok=True)
    return out


def open_file_explorer(path: Path):
    try:
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])
    except Exception as e:
        print(f"\n[WARNING] Could not open file explorer: {e}\nManually open: {path.resolve()}")


def xy_plot_final(xs, ys, labels, user_map, color_map, path: Path, title: str):
    xs, ys = np.asarray(xs, dtype=np.int32), np.asarray(ys, dtype=np.int32)
    semantic_labels = np.array([user_map.get(l, "unlabeled") for l in labels])
    plt.figure(figsize=(8, 7), dpi=150)
    ax = plt.gca()
    for label_name, color in color_map.items():
        mask = (semantic_labels == label_name)
        if np.any(mask):
            plt.scatter(ys[mask], xs[mask], s=6, alpha=0.9, label=label_name, color=color)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(ys.min(), ys.max())
    ax.set_ylim(xs.min(), xs.max())
    ax.invert_yaxis()
    plt.xlabel("Scan Pixel Y")
    plt.ylabel("Scan Pixel X")
    plt.legend(markerscale=3, fontsize=10, frameon=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def preprocess_cpu_for_montage(img224):
    x = img224.astype(np.float32, copy=False)
    x = np.maximum(x, 0.0)
    x = np.log1p(x)
    hi = np.percentile(x, 99.5)
    if not np.isfinite(hi) or hi <= 0: hi = x.max() if x.max() > 0 else 1.0
    return np.clip(x, 0, hi) / (hi + 1e-8)


def center_crop_cpu(img, size=224):
    h, w = img.shape
    if h < size or w < size:
        pad_h, pad_w = max(0, size - h), max(0, size - w)
        img = np.pad(img, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)), mode='edge')
        h, w = img.shape
    top, left = (h - size) // 2, (w - size) // 2
    return img[top:top + size, left:left + size]


# =========================
#     GPU FEATURE EXTRACT & CLUSTERING (Unchanged)
# =========================
class GPUExtractor:
    def __init__(self, device, nbins, harmonics, rad_bands, center_mask_radius, rmax, ntheta, batch):
        self.device, self.nbins, self.harmonics, self.rad_bands, self.center_mask_radius, self.rmax, self.ntheta, self.batch = device, int(
            nbins), tuple(int(m) for m in harmonics), int(rad_bands), float(center_mask_radius), float(rmax), int(
            ntheta), int(batch)
        self.H = self.W = 224
        self.R = max(8, min(int(rmax), 112))
        self.center_mask = self._make_center_mask().to(self.device)
        self.polar_grid = self._make_polar_grid().to(self.device)
        self.bin_idx_nb, self.counts_nb = self._make_bin_index(self.R, self.nbins)
        self.bin_idx_rb, self.counts_rb = self._make_bin_index(self.R, self.rad_bands)

    def _make_center_mask(self):
        yy, xx = torch.meshgrid(torch.arange(self.H, dtype=torch.float32), torch.arange(self.W, dtype=torch.float32),
                                indexing='ij')
        cy, cx = (self.H - 1) / 2.0, (self.W - 1) / 2.0
        rr = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        return (rr <= self.center_mask_radius).float().unsqueeze(0).unsqueeze(0)

    def _make_polar_grid(self):
        rmax_allowed = (min(self.H, self.W) - 1) / 2.0
        rmax = min(self.rmax, rmax_allowed - 1e-6)
        r = torch.linspace(0.0, rmax, self.R)
        theta = torch.arange(self.ntheta, dtype=torch.float32) * (2.0 * math.pi / self.ntheta)
        rr, tt = torch.meshgrid(r, theta, indexing='ij')
        cy, cx = (self.H - 1) / 2.0, (self.W - 1) / 2.0
        yy = cy + rr * torch.sin(tt)
        xx = cx + rr * torch.cos(tt)
        x_norm = ((xx / (self.W - 1)) * 2.0 - 1.0).clamp(-1.0, 1.0)
        y_norm = ((yy / (self.H - 1)) * 2.0 - 1.0).clamp(-1.0, 1.0)
        return torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0).to(torch.float32)

    def _make_bin_index(self, R, num_bins):
        edges = torch.linspace(0, R, num_bins + 1)
        ridx = torch.arange(R, dtype=torch.float32)
        bin_idx = (torch.bucketize(ridx, edges) - 1).clamp(0, num_bins - 1).to(torch.long)
        counts = torch.zeros(num_bins, dtype=torch.float32)
        counts.scatter_add_(0, bin_idx, torch.ones(R, dtype=torch.float32))
        return bin_idx.to(self.device), counts.to(self.device)

    @torch.no_grad()
    def _pool_bins(self, BR, bin_idx_R, counts, num_bins):
        B, R = BR.shape
        idx = bin_idx_R.view(1, R).expand(B, R)
        out = torch.zeros(B, num_bins, device=BR.device, dtype=BR.dtype)
        out.scatter_add_(1, idx, BR)
        return out / (counts.view(1, num_bins) + 1e-8)

    @torch.no_grad()
    def _preprocess(self, x_bchw):
        B, C, H, W = x_bchw.shape
        top = (H - self.H) // 2
        left = (W - self.W) // 2
        x = x_bchw[:, :, top:top + self.H, left:left + self.W].clamp_min_(0.0)
        x = torch.log1p(x)
        if self.center_mask_radius > 0:
            mask = self.center_mask
            outside_sum = (x * (1.0 - mask)).sum(dim=(1, 2, 3), keepdim=True)
            outside_cnt = (1.0 - mask).sum(dim=(1, 2, 3), keepdim=True)
            outside_mean = outside_sum / (outside_cnt + 1e-8)
            x = x * (1.0 - mask) + outside_mean * mask
        q = torch.quantile(x.view(B, -1), q=0.995, dim=1, keepdim=True).view(B, 1, 1, 1).clamp_min_(1e-6)
        x = torch.minimum(x, q) / (q + 1e-8)
        return x

    @torch.no_grad()
    def _angular_features(self, polar_BRT, polar_BRT_dc):
        B, R, T = polar_BRT.shape
        Fr = torch.fft.rfft(polar_BRT, dim=-1)
        T_half = Fr.shape[-1]
        dc = polar_BRT_dc + 1e-8
        mags = []
        for m in self.harmonics:
            if m >= T_half: mags.append(torch.zeros(B, self.rad_bands, device=Fr.device, dtype=Fr.real.dtype)); continue
            mag_BR = Fr[..., m].abs() / dc
            mags.append(self._pool_bins(mag_BR, self.bin_idx_rb, self.counts_rb, self.rad_bands))
        return torch.cat(mags, dim=1)

    @torch.no_grad()
    def extract(self, data_array):
        N, H, W = data_array.shape
        Fdim = self.nbins + self.nbins + 20 + 4 + len(self.harmonics) * self.rad_bands
        X_out = np.zeros((N, Fdim), dtype=np.float32)
        steps = math.ceil(N / self.batch)
        for step in tqdm(range(steps), desc="GPU feature extraction"):
            s, e = step * self.batch, min(N, (step + 1) * self.batch)
            chunk = np.array(data_array[s:e], dtype=np.float32, copy=False)
            t = torch.from_numpy(chunk).unsqueeze(1).to(self.device, non_blocking=True)
            cart = self._preprocess(t)
            B = cart.size(0)
            pol = F.grid_sample(cart, self.polar_grid.expand(B, -1, -1, -1), mode='bilinear', padding_mode='border',
                                align_corners=True).squeeze(1)
            pol_centered = pol - pol.mean(dim=-1, keepdim=True)
            Fr0 = torch.fft.rfft(pol, dim=-1)
            dc_mag = Fr0[..., 0].abs()
            ang_feat = self._angular_features(pol_centered, dc_mag)
            rad_mean_r, rad_std_r = pol.mean(dim=-1), pol.std(dim=-1)
            rad_mean_b = self._pool_bins(rad_mean_r, self.bin_idx_nb, self.counts_nb, self.nbins)
            rad_std_b = self._pool_bins(rad_std_r, self.bin_idx_nb, self.counts_nb, self.nbins)
            rad_mean_b /= (rad_mean_b.max(dim=1, keepdim=True).values + 1e-8)
            rad_std_b /= (rad_std_b.max(dim=1, keepdim=True).values + 1e-8)
            kpk = min(10, self.nbins)
            vals, idxs = torch.topk(rad_mean_b, k=kpk, dim=1)
            pk = torch.zeros(B, 20, device=rad_mean_b.device, dtype=rad_mean_b.dtype)
            if kpk > 0: pk[:, :kpk] = idxs.float() / max(1, self.nbins - 1); pk[:, 10:10 + kpk] = vals
            mid = self.nbins // 3
            lowE = rad_mean_b[:, :mid].mean(dim=1, keepdim=True)
            highE = rad_mean_b[:, mid:].mean(dim=1, keepdim=True)
            ratio = lowE / (highE + 1e-8)
            flat = cart.view(B, -1)
            q10 = torch.quantile(flat, 0.1, dim=1, keepdim=True)
            q50 = torch.quantile(flat, 0.5, dim=1, keepdim=True)
            q90 = torch.quantile(flat, 0.9, dim=1, keepdim=True)
            base = torch.cat([rad_mean_b, rad_std_b, pk, ratio, q10, q50, q90], dim=1)
            feat = torch.cat([base, ang_feat], dim=1)
            X_out[s:e] = feat.detach().cpu().numpy()
        return X_out


@torch.no_grad()
def kmeans_torch(X, K, iters=50, restarts=2, seed=0):
    N, D = X.shape
    C = kmeanspp_init(X, K, restarts=restarts, seed=seed)
    labels_prev = None
    for _ in range(iters):
        d2 = pairwise_dist2(X, C)
        labels = d2.argmin(dim=1)
        if labels_prev is not None and torch.equal(labels, labels_prev): break
        labels_prev = labels
        for k in range(K):
            m = (labels == k)
            if m.any():
                C[k] = X[m].mean(dim=0)
            else:
                C[k] = X[torch.randint(0, N, (1,), device=X.device).item()]
    d2 = pairwise_dist2(X, C)
    min_d2, labels = d2.min(dim=1)
    return labels, min_d2, C


@torch.no_grad()
def kmeanspp_init(X, K, restarts=2, seed=0):
    N = X.shape[0]
    best_C, best_inertia = None, None
    g = torch.Generator(device=X.device)
    g.manual_seed(seed)
    for _ in range(max(1, restarts)):
        i0 = torch.randint(0, N, (1,), generator=g, device=X.device).item()
        centers = [X[i0:i0 + 1]]
        d2 = pairwise_dist2(X, centers[0])[:, 0]
        for _k in range(1, K):
            probs = d2 / (d2.sum() + 1e-8)
            idx = torch.multinomial(probs, 1, generator=g).item()
            centers.append(X[idx:idx + 1])
            C = torch.cat(centers, dim=0)
            d2 = torch.minimum(d2, pairwise_dist2(X, C)[..., -1])
        C = torch.cat(centers, dim=0)
        inertia = d2.mean()
        if (best_inertia is None) or (inertia < best_inertia): best_inertia, best_C = inertia, C
    return best_C.clone()


@torch.no_grad()
def pairwise_dist2(X, C):
    x2 = (X * X).sum(dim=1, keepdim=True)
    c2 = (C * C).sum(dim=1).unsqueeze(0)
    d2 = x2 - 2. * (X @ C.t()) + c2
    return torch.clamp(d2, min=0.0)


@torch.no_grad()
def pca_torch(X, n_components):
    Xc = X - X.mean(dim=0, keepdim=True)
    C = (Xc.T @ Xc) / (X.shape[0] - 1)
    evals, evecs = torch.linalg.eigh(C)
    evals, evecs = torch.flip(evals, dims=[0]), torch.flip(evecs, dims=[1])
    C = min(n_components, X.shape[1])
    comps = evecs[:, :C]
    Z = Xc @ comps
    return Z, comps, evals[:C]


def save_cluster_montages(data_array, labels_np, min_d2_np, out_dir, grid):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, cols = grid
    per = rows * cols
    uniq = sorted(np.unique(labels_np))
    for k in tqdm(uniq, desc="Saving initial montages"):
        idxs = np.where(labels_np == k)[0]
        if len(idxs) == 0: continue
        order = np.argsort(min_d2_np[idxs])
        take = idxs[order[:per]] if len(order) >= per else np.pad(idxs[order], (0, per - len(order)), mode='wrap')
        canvas = np.zeros((rows * 224, cols * 224), dtype=np.float32)
        for i, idx in enumerate(take):
            r, c = i // cols, i % cols
            img = center_crop_cpu(data_array[idx], 224)
            img = preprocess_cpu_for_montage(img)
            canvas[r * 224:(r + 1) * 224, c * 224:(c + 1) * 224] = img
        plt.figure(figsize=(cols, rows), dpi=150)
        plt.imshow(canvas, cmap='gray', vmin=0.0, vmax=1.0)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(out_dir / f"cluster_{k}.png", bbox_inches="tight", pad_inches=0)
        plt.close()


# ==================================
#      INTERACTIVE WORKFLOW
# ==================================
def run_clustering_for_file(device, npy_file_path: Path):
    arr = np.load(npy_file_path, allow_pickle=True)
    data, xs, ys = arr["data"], arr["x"], arr["y"]
    N, H, W = data.shape
    print(f"[INFO] Loaded {N} frames from: {npy_file_path.name}")
    out_dir = stemmed_outdir(npy_file_path, OUT_ROOT)
    extractor = GPUExtractor(device, NBINS, HARMONICS, RAD_BANDS, CENTER_MASK_RADIUS, RMAX, NTHETA, BATCH)
    X = extractor.extract(data)
    print(f"[INFO] Feature extraction complete. Shape={X.shape}")
    Xt = torch.from_numpy(X).to(device)
    Z, _, _ = pca_torch(Xt, n_components=min(50, Xt.shape[1]))
    labels_t, min_d2_t, _ = kmeans_torch(Z, K=K, iters=KMEANS_ITERS, restarts=INIT_RESTARTS, seed=SEED)
    labels_np, min_d2_np = labels_t.cpu().numpy().astype(np.int32), min_d2_t.cpu().numpy().astype(np.float32)
    print(f"[RESULT] Clustering complete. Found K={K} clusters.")
    if SAVE_MONTAGE:
        montage_dir = out_dir / "montages"
        save_cluster_montages(data, labels_np, min_d2_np, montage_dir, grid=MONTAGE_GRID)
        print(f"[INFO] Montages saved to {montage_dir}")
        open_file_explorer(montage_dir)
    return {
        "original_arr": arr, "labels": labels_np, "xs": xs, "ys": ys,
        "K": K, "out_dir": out_dir, "file_path": npy_file_path
    }


def get_user_labels_interactive(K: int, file_name: str) -> dict:
    """Handles the CLI interaction to get semantic labels, with an undo feature."""
    print("\n" + "=" * 60)
    print(f"      INTERACTIVE LABELING for: {file_name}")
    print("=" * 60)
    print("The 'montages' folder has been opened. Please inspect the images.")

    user_label_map = {}
    labeled_order = []  # Keep track of the order clusters were labeled in
    current_k = 0

    while current_k < K:
        print("\n" + "-" * 50)
        print(f"Enter the label for CLUSTER {current_k}:")
        for i, cat in enumerate(CATEGORIES): print(f"  [{i}] {cat}")
        print("  [-1] Undo last entry")

        user_input = input(f"Label for cluster {current_k} > ").lower().strip()

        if user_input == "-1":
            if not labeled_order:
                print("\n[INFO] No labels entered yet. Nothing to undo.")
                continue
            last_k = labeled_order.pop()
            del user_label_map[last_k]
            current_k = last_k
            print(f"\n[UNDO] Removed label for Cluster {last_k}. Please re-enter.")
            continue
        elif user_input.isdigit() and 0 <= int(user_input) < len(CATEGORIES):
            label = CATEGORIES[int(user_input)]
            user_label_map[current_k] = label
            labeled_order.append(current_k)
            print(f"  >> Cluster {current_k} labeled as: {label}")
            current_k += 1
        elif user_input in CATEGORIES:
            label = user_input
            user_label_map[current_k] = label
            labeled_order.append(current_k)
            print(f"  >> Cluster {current_k} labeled as: {label}")
            current_k += 1
        else:
            print("\n[ERROR] Invalid input. Please enter a valid number or category name.")

    print("\n" + "=" * 60)
    print("All clusters for this file have been labeled.")
    print("=" * 60)
    return user_label_map


def finalize_and_save(cluster_results: dict, user_map: dict):
    out_dir, file_path = cluster_results["out_dir"], cluster_results["file_path"]
    final_xy_path = out_dir / f"{file_path.stem}_xy_map_FINAL.png"
    print(f"[INFO] Generating final semantic XY map...")
    xy_plot_final(
        xs=cluster_results["xs"], ys=cluster_results["ys"], labels=cluster_results["labels"],
        user_map=user_map, color_map=FINAL_COLOR_MAP, path=final_xy_path,
        title=f"Final Labeled Map for {file_path.stem}"
    )
    print(f"[SUCCESS] Final map saved to: {final_xy_path}")

    print(f"[INFO] Updating 'class' field in original file: {file_path.name}")
    original_arr, numeric_labels = cluster_results["original_arr"], cluster_results["labels"]
    max_len = max(len(s) for s in CATEGORIES)
    final_class_labels = np.array(
        [user_map[k].encode('utf-8') for k in numeric_labels], dtype=f'|S{max_len}'
    )
    original_arr['class'] = final_class_labels
    np.save(file_path, original_arr)
    print(f"[SUCCESS] Original file updated and saved.")


# =========================
#            MAIN
# =========================
def main():
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device = {device}")

    npy_dir = Path(NPY_DIR)
    if not npy_dir.is_dir():
        print(f"[ERROR] Input directory not found: {npy_dir}")
        return
    file_list = sorted(list(npy_dir.glob("*.npy")))
    if not file_list:
        print(f"[ERROR] No .npy files found in directory: {npy_dir}")
        return

    print(f"[INFO] Found {len(file_list)} .npy files to process.")

    for i, file_path in enumerate(file_list):
        print("\n" + "#" * 70)
        print(f"# PROCESSING FILE {i + 1} of {len(file_list)}: {file_path.name}")
        print("#" * 70)
        try:
            cluster_results = run_clustering_for_file(device, file_path)
            user_label_map = get_user_labels_interactive(cluster_results["K"], file_path.name)
            finalize_and_save(cluster_results, user_label_map)
        except Exception as e:
            print("\n" + "!" * 70)
            print(f"! An error occurred while processing {file_path.name}:")
            print(f"! ERROR: {e}")
            print("! Skipping to the next file.")
            print("!" * 70)
            continue

    print("\n[ALL FILES PROCESSED. SCRIPT FINISHED.]")


if __name__ == "__main__":
    main()