"""
train_clock2map_generative.py — CONDITIONAL GENERATIVE CLOCK -> RNFL MAP DECODER

  12 clock-hour RNFL thickness values  ->  200 x 200 RNFL thickness map (um)

WHY THIS DECODER
  The kNN exemplar decoder reproduces real-map texture but must ship a bank of
  real development-site thickness maps (patient-derived data). This decoder ships
  ONLY network weights: a generator trained on every eligible development-site
  scan learns the population mapping clocks -> map, including bundle texture, and
  no individual map is stored anywhere in the artifact.

DESIGN
  * Data: identical eligibility, unit conversion (axial px x 1.955 = um) and
    OS -> OD mirroring (map flip + clock permutation h -> 12-h) as the kNN
    decoder. Optional cup-centred registration (REGISTER_CUP).
  * Conditioning: standardized 12 clocks -> MLP latent, plus a geometric
    "pseudo-map" rendered from the clocks (sector values painted on the 3.46 mm
    circle) fed as a spatial channel at every decoder scale, plus coordinate
    channels. The pseudo-map gives the generator the angular geometry for free;
    the network learns bundle anatomy and texture from 200k maps.
  * Generator: 25x25 latent -> 50 -> 100 -> 200 upsampling CNN with FiLM
    conditioning at each scale; softplus output (um/100 scale).
  * Loss: masked MSE on valid target pixels (conditional-mean fidelity)
        + GRAD_WEIGHT  * masked Sobel-gradient L1   (edge sharpness)
        + ADV_WEIGHT   * hinge adversarial term from a conditional PatchGAN
          discriminator (texture realism; set ADV_WEIGHT = 0 for pure MSE).
    Texture added by the adversarial term beyond what 12 numbers determine is
    INVENTED detail. The generated maps are population-typical syntheses that are
    consistent with the 12 clock-hour values, not reconstructions of an individual
    scan; state this explicitly wherever they are used (for example as input to a
    model that was trained on real maps).
  * Honest evaluation: patient-grouped 80/20 split, early stopping on held-out
    masked RMSE, report RMSE / SSIM / ring-sector agreement vs mean-map
    baseline. Then refit on ALL scans for the selected number of epochs and
    save the deployment decoder.
  * Privacy: each patient contributes a tiny fraction of ~200k samples, so
    memorisation risk is low; DP-SGD is not implemented here — add it if the
    IRB wants a formal guarantee.

INPUT  (paths come from environment variables, see CONFIG)
  C2M_META_CSV   one row per scan: clockhour1 ... clockhour12 (mean RNFL thickness,
                 um), datadir (scan sub-folder of C2M_SCANS_DIR), id (patient
                 identifier, used only to keep each patient on one side of the
                 split), and optionally righteye (1 = right eye, 0 = left eye) and
                 signalstrength
  C2M_SCANS_DIR  one folder per scan holding segmentation_rnfl_to_gcl.csv and
                 segmentation_ilm.csv (200 x 200 boundary maps in axial pixels),
                 optionally cup_coordinates_x.csv / cup_coordinates_y.csv

DATA HYGIENE
  CACHE_DIR holds patient-level arrays (maps, clock values, patient ids) and
  OUT_DIR the example figure of real held-out maps: keep both outside the git
  repository.

OUTPUTS (OUT_DIR)
  clock2map_generative_all_available.pth   generator weights (ship this)
  decoder_generative_clock2map_all_available.npz
                                            scaler, disc mask, config, weights
                                            sha256 — the sidecar the dataset
                                            builder / fine-tune scripts read
  decoder_generative_report_all_available.json
  generative_examples_heldout.png          real | clocks | generated (no IDs)

USAGE AT INFERENCE (also importable):
  from train_clock2map_generative import GenerativeDecoder
  dec = GenerativeDecoder(npz_path)            # finds the .pth next to it
  m_od = dec.decode(clocks_um, eye='OS')       # 200x200 um map in OD orientation
                                               # (mirror back for OS exactly as
                                               #  for the kNN / ridge decoders)
"""

import os, gc, json, time, hashlib, multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit

# =============================================================================
# CONFIG
# =============================================================================
# paths: set the environment variables (or edit the defaults); nothing site-specific is hard-coded
DATA_ROOT = os.environ.get('C2M_DATA_ROOT', 'path/to/oct_export')
META_CSV  = os.environ.get('C2M_META_CSV',  os.path.join(DATA_ROOT, 'metadata.csv'))
SCANS_DIR = os.environ.get('C2M_SCANS_DIR', os.path.join(DATA_ROOT, 'scans'))
OUT_DIR   = os.environ.get('C2M_OUT_DIR',   os.path.expanduser('~/clock2map_generative/out'))
CACHE_DIR = os.environ.get('C2M_CACHE_DIR', os.path.expanduser('~/clock2map_generative/cache'))   # patient-level arrays

MAX_SCANS      = None        # None = every eligible scan; set e.g. 5000 for a smoke test
N_LOAD_WORKERS = 16          # CSV loading processes for the cache build
CACHE_CHUNK    = 4000        # candidates submitted to the pool at a time (bounds result buffering; checkpoint granularity)
CACHE_DTYPE    = np.float16  # np.uint8 halves the cache (~8 GB for 200k scans): integer um clipped at 255, negligible loss for this decoder
REGISTER_CUP   = False       # True: translate each map so its cup centroid is at the map centre (needs cup_coordinates_*.csv)
MIRROR_OS      = True        # verified convention
MIN_SIGNAL     = 7
UM_PER_AXIAL_PX = 1.955
MAP_SIZE       = 200
SEED           = 42

# training
TEST_FRAC      = 0.20
BATCH          = 64
EPOCHS         = 30
PATIENCE       = 5
LR_G, LR_D     = 2e-4, 2e-4
WEIGHT_DECAY   = 1e-4
GRAD_WEIGHT    = 0.5         # masked Sobel-gradient L1
ADV_WEIGHT     = 0.02        # hinge adversarial term (0 -> pure regression decoder)
BASE_CH        = 128         # generator width
NUM_WORKERS    = 6
# =============================================================================

mpl.rcParams['font.family'] = 'DejaVu Sans'
CIRCLE_R_PX = (3.46 / 2) / (6.0 / MAP_SIZE)                      # 57.7 px
_yy, _xx = np.mgrid[0:MAP_SIZE, 0:MAP_SIZE]
_cx = _cy = (MAP_SIZE - 1) / 2
_r = np.hypot(_xx - _cx, _yy - _cy)
_ring = np.abs(_r - CIRCLE_R_PX) <= 3
_ang = np.degrees(np.arctan2(_xx - _cx, -(_yy - _cy))) % 360       # 0 = top, clockwise (image coords)
_sector = ((_ang + 15) // 30).astype(int) % 12                   # 0 = hour 12, 1 = hour 1, ..., 11 = hour 11

# =============================================================================
# training-image conversion (1/t, jet_r, white invalid) — figures only
# =============================================================================
def convert_array_to_pil(depth_map):
    depth_map = np.where(depth_map == 0, np.nan, depth_map)
    disp = 1.0 / depth_map
    mask = np.isfinite(disp)
    vmax, vmin = (np.percentile(disp[mask], 95), np.percentile(disp[mask], 5)) if mask.any() else (1.0, 0.0)
    mapper = cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap='jet_r')
    out = (mapper.to_rgba(disp)[:, :, :3] * 255).astype(np.uint8)
    out[~np.repeat(mask[..., None], 3, -1)] = 255
    return out

# =============================================================================
# conventions shared with the kNN / ridge decoders
# =============================================================================
def normalize_pid(x):
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()

def mirror_clocks_os(clocks):
    """OS clock vector -> OD-space orientation: h -> 12-h (12 and 6 fixed)."""
    clocks = np.asarray(clocks, dtype=float); out = np.empty(12)
    for h in range(1, 13):
        src = 12 if h == 12 else 12 - h
        out[h - 1] = clocks[(src - 1) % 12]
    return out

def sector_means(m):
    """12 ring-sector means (index 0 = hour 12, ..., 11 = hour 11) of a map in OD space, ring at map centre."""
    valid = m > 0
    return np.array([np.nanmean(m[_ring & valid & (_sector == s)]) if (_ring & valid & (_sector == s)).any() else np.nan
                     for s in range(12)])

def sectors_to_clock_order(sec):
    """ring-sector array (hour 12 first) -> clock column order (hour 1 ... 12)"""
    return np.r_[sec[1:], sec[:1]]

def map_means(m):
    valid = m > 0
    return float(m[valid].mean()), float(m[_ring & valid].mean())

# =============================================================================
# map loading (worker-safe)
# =============================================================================
def _load_map_um(scan_dir, eye):
    f1, f2 = os.path.join(scan_dir, 'segmentation_rnfl_to_gcl.csv'), os.path.join(scan_dir, 'segmentation_ilm.csv')
    if not (os.path.exists(f1) and os.path.exists(f2)):
        return None
    try:
        vec = (pd.read_csv(f1, header=None).values - pd.read_csv(f2, header=None).values).astype(np.float64).ravel()
    except Exception:
        return None
    if vec.size != MAP_SIZE * MAP_SIZE:
        return None
    m = vec.reshape(MAP_SIZE, MAP_SIZE)
    valid = np.isfinite(m) & (m > 0)
    if valid.mean() < 0.50:
        return None
    m = np.where(valid, m * UM_PER_AXIAL_PX, 0.0)
    if REGISTER_CUP:
        fx, fy = os.path.join(scan_dir, 'cup_coordinates_x.csv'), os.path.join(scan_dir, 'cup_coordinates_y.csv')
        if os.path.exists(fx) and os.path.exists(fy):
            try:
                cx = float(np.nanmean(pd.read_csv(fx, header=None).values)); cy = float(np.nanmean(pd.read_csv(fy, header=None).values))
                from scipy.ndimage import shift as nd_shift
                m = nd_shift(m, (_cy - cy, _cx - cx), order=1, mode='constant', cval=0.0)
                m[m < 0] = 0.0
            except Exception:
                pass
    if MIRROR_OS and eye == 'OS':
        m = m[:, ::-1]
    return m.astype(np.float16)

def _worker(args):
    i, scan_dir, eye = args
    return i, _load_map_um(scan_dir, eye)

# =============================================================================
# cache: load every eligible scan once into a memmap
# =============================================================================
CACHE_DTYPE = np.dtype(CACHE_DTYPE)
ROW_BYTES = MAP_SIZE * MAP_SIZE * CACHE_DTYPE.itemsize  # one map
RAW_PATH  = os.path.join(CACHE_DIR, f'maps_{CACHE_DTYPE.name}.raw')   # headerless rows, appended with buffered writes (never mmap'd for writing)

def _to_cache_dtype(m):
    if CACHE_DTYPE == np.uint8:
        return np.clip(np.rint(m.astype(np.float32)), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(m, dtype=CACHE_DTYPE)
KEEP_PATH = os.path.join(CACHE_DIR, 'keep_idx.npy')
PROG_PATH = os.path.join(CACHE_DIR, 'progress.json')

def _candidates():
    """Deterministic candidate table (same order every run, so a resumed build is consistent)."""
    df = pd.read_csv(META_CSV, low_memory=False)
    clk = sorted([c for c in df.columns if c.lower().startswith('clockhour')],
                 key=lambda c: int(''.join(filter(str.isdigit, c))))
    assert len(clk) == 12, clk
    ok = df[clk].notna().all(axis=1) & df['datadir'].notna()
    if 'signalstrength' in df.columns:
        ok &= pd.to_numeric(df['signalstrength'], errors='coerce') >= MIN_SIGNAL
    cols = clk + ['datadir', 'id'] + (['righteye'] if 'righteye' in df.columns else [])
    cand = df.loc[ok, cols].reset_index(drop=True)
    del df; gc.collect()                                  # free the full metadata frame before the pool forks
    if MAX_SCANS is not None:
        cand = cand.sample(n=min(MAX_SCANS, len(cand)), random_state=SEED).reset_index(drop=True)
    eyes_c = np.where(cand['righteye'].isin([1, 1.0]), 'OD', 'OS') if 'righteye' in cand.columns else np.full(len(cand), 'OD')
    clocks_c = cand[clk].astype(float).values
    for i in np.where(eyes_c == 'OS')[0]:
        clocks_c[i] = mirror_clocks_os(clocks_c[i])
    return cand, eyes_c, clocks_c

def build_cache():
    """Load every eligible map once into a raw float16 file. Chunked, checkpointed and resumable:
    a crash (OOM, SIGBUS, reboot) loses at most CACHE_CHUNK candidates of work."""
    meta_path = os.path.join(CACHE_DIR, 'meta.json')
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        if (meta.get('complete') and meta.get('max_scans') == MAX_SCANS and meta.get('register_cup') == REGISTER_CUP
                and meta.get('dtype') == CACHE_DTYPE.name):
            print(f'cache found: {meta["n"]:,} scans in {CACHE_DIR}'); return meta
    print('Loading metadata...')
    cand, eyes_c, clocks_c = _candidates()
    print(f'candidate scans with complete clocks: {len(cand):,}  (cache will need ~{len(cand) * ROW_BYTES / 1e9:.1f} GB)')
    jobs = [(i, os.path.join(SCANS_DIR, str(cand.at[i, 'datadir']).strip()), eyes_c[i]) for i in range(len(cand))]

    # ---- resume if a checkpoint matches this candidate table ----
    next_cand, n, keep = 0, 0, []
    if os.path.exists(PROG_PATH) and os.path.exists(RAW_PATH) and os.path.exists(KEEP_PATH):
        p = json.load(open(PROG_PATH))
        if (p.get('n_candidates') == len(cand) and p.get('max_scans') == MAX_SCANS and p.get('register_cup') == REGISTER_CUP
                and p.get('dtype') == CACHE_DTYPE.name):
            next_cand, n, keep = int(p['next_cand']), int(p['n']), list(np.load(KEEP_PATH))
            with open(RAW_PATH, 'r+b') as f:
                f.truncate(n * ROW_BYTES)                 # drop any partial row from the crash
            print(f'resuming at candidate {next_cand:,} ({n:,} scans already cached)')
    if next_cand == 0 and os.path.exists(RAW_PATH):
        os.remove(RAW_PATH)

    t0 = time.time()
    with open(RAW_PATH, 'ab', buffering=8 << 20) as f, mp.Pool(N_LOAD_WORKERS, maxtasksperchild=2000) as pool:
        for start in range(next_cand, len(jobs), CACHE_CHUNK):
            chunk = jobs[start:start + CACHE_CHUNK]
            for i, m in pool.imap(_worker, chunk, chunksize=16):   # bounded: at most one chunk of results in flight
                if m is None:
                    continue
                f.write(_to_cache_dtype(m).tobytes()); keep.append(i); n += 1
            f.flush(); os.fsync(f.fileno())
            np.save(KEEP_PATH, np.asarray(keep, dtype=np.int64))
            json.dump({'next_cand': start + len(chunk), 'n': n, 'n_candidates': len(cand), 'dtype': CACHE_DTYPE.name,
                       'max_scans': MAX_SCANS, 'register_cup': REGISTER_CUP}, open(PROG_PATH, 'w'))
            print(f'  {start + len(chunk):,}/{len(jobs):,} candidates -> {n:,} scans cached '
                  f'({time.time() - t0:.0f}s, {os.path.getsize(RAW_PATH) / 1e9:.1f} GB on disk)', flush=True)

    keep = np.asarray(keep, dtype=np.int64)
    np.save(os.path.join(CACHE_DIR, 'clocks.npy'), clocks_c[keep].astype(np.float32))
    np.save(os.path.join(CACHE_DIR, 'pids.npy'), np.array([normalize_pid(x) for x in cand.loc[keep, 'id']]))
    np.save(os.path.join(CACHE_DIR, 'eyes.npy'), eyes_c[keep])
    meta = {'n': int(n), 'n_candidates': int(len(cand)), 'max_scans': MAX_SCANS, 'register_cup': REGISTER_CUP,
            'dtype': CACHE_DTYPE.name, 'units': 'um', 'um_per_axial_px': UM_PER_AXIAL_PX, 'mirror_os': MIRROR_OS, 'complete': True}
    json.dump(meta, open(meta_path, 'w'), indent=2)
    print(f'cache built: {n:,} scans ({len(cand) - n:,} missing/invalid maps)')
    return meta

class MapCache:
    def __init__(self, n):
        assert os.path.getsize(RAW_PATH) >= n * ROW_BYTES, 'raw cache shorter than meta says — rebuild'
        self.maps = np.memmap(RAW_PATH, dtype=CACHE_DTYPE, mode='r', shape=(n, MAP_SIZE * MAP_SIZE))  # read-only mapping; rows are um
        self.clocks = np.load(os.path.join(CACHE_DIR, 'clocks.npy'))[:n]
        self.pids = np.load(os.path.join(CACHE_DIR, 'pids.npy'), allow_pickle=True)[:n]
        self.eyes = np.load(os.path.join(CACHE_DIR, 'eyes.npy'), allow_pickle=True)[:n]

class ClockMapDataset(Dataset):
    def __init__(self, cache, idx, x_mu, x_sd):
        self.c, self.idx, self.mu, self.sd = cache, np.asarray(idx), x_mu.astype(np.float32), x_sd.astype(np.float32)
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        m = np.asarray(self.c.maps[j], dtype=np.float32).reshape(MAP_SIZE, MAP_SIZE) / 100.0      # um/100
        c = self.c.clocks[j]
        return {'x_std': torch.from_numpy((c - self.mu) / self.sd), 'x_um': torch.from_numpy(c.astype(np.float32)),
                'map': torch.from_numpy(m)[None], 'idx': int(j)}

# =============================================================================
# geometric pseudo-map (torch, differentiable) — conditioning channel
# =============================================================================
_ang_t = torch.tensor(_ang, dtype=torch.float32)
_r_t = torch.tensor(_r, dtype=torch.float32)
_f = _ang_t / 30.0
_h_lo = torch.floor(_f).long()
_w_hi = (_f - _h_lo.float())
_idx_lo = (_h_lo - 1) % 12                  # clock column index (hour 1 = 0 ... hour 12 = 11)
_idx_hi = _h_lo % 12
_env = (torch.exp(-0.5 * ((_r_t - CIRCLE_R_PX) / 18.0) ** 2) * 0.7 + 0.3) * (_r_t > 14).float()

def render_pseudo(clocks_um):
    """(B,12) um, OD orientation -> (B,1,200,200) um/100: sector values interpolated angularly, radial envelope."""
    dev = clocks_um.device
    lo, hi, w, env = _idx_lo.to(dev), _idx_hi.to(dev), _w_hi.to(dev), _env.to(dev)
    v = (1 - w)[None] * clocks_um[:, lo] + w[None] * clocks_um[:, hi]          # (B,200,200)
    return (v * env[None] / 100.0)[:, None]

_coords = torch.stack([torch.tensor((_xx - _cx) / _cx, dtype=torch.float32),
                       torch.tensor((_yy - _cy) / _cy, dtype=torch.float32)])   # (2,200,200)

# =============================================================================
# model
# =============================================================================
class FiLM(nn.Module):
    def __init__(self, cond, ch):
        super().__init__(); self.lin = nn.Linear(cond, 2 * ch)
    def forward(self, x, c):
        g, b = self.lin(c).chunk(2, dim=1)
        return x * (1 + g[:, :, None, None]) + b[:, :, None, None]

class UpBlock(nn.Module):
    def __init__(self, cin, cout, cond):
        super().__init__()
        self.conv1 = nn.Conv2d(cin + 3, cout, 3, padding=1)        # +1 pseudo, +2 coords
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.n1, self.n2 = nn.GroupNorm(8, cout), nn.GroupNorm(8, cout)
        self.film = FiLM(cond, cout)
    def forward(self, x, c, pseudo, coords):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        s = x.shape[-1]
        p = F.interpolate(pseudo, size=(s, s), mode='bilinear', align_corners=False)
        k = F.interpolate(coords, size=(s, s), mode='bilinear', align_corners=False)
        x = F.gelu(self.n1(self.conv1(torch.cat([x, p, k], 1))))
        x = F.gelu(self.n2(self.film(self.conv2(x), c)))
        return x

class ClockGen(nn.Module):
    """12 standardized clocks (+ pseudo-map) -> 200x200 map (um/100), OD orientation."""
    def __init__(self, base=BASE_CH, cond=256):
        super().__init__()
        self.cond = nn.Sequential(nn.Linear(12, cond), nn.GELU(), nn.Linear(cond, cond), nn.GELU())
        self.lat = nn.Linear(cond, base * 25 * 25)
        self.up1 = UpBlock(base, base, cond)             # 25 -> 50
        self.up2 = UpBlock(base, base // 2, cond)        # 50 -> 100
        self.up3 = UpBlock(base // 2, base // 4, cond)   # 100 -> 200
        self.head = nn.Sequential(nn.Conv2d(base // 4 + 3, base // 4, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(base // 4, 1, 1))
        self.base = base
    def forward(self, x_std, x_um):
        c = self.cond(x_std)
        pseudo = render_pseudo(x_um)
        coords = _coords.to(x_std.device)[None].expand(x_std.shape[0], -1, -1, -1)
        h = self.lat(c).view(-1, self.base, 25, 25)
        h = self.up1(h, c, pseudo, coords); h = self.up2(h, c, pseudo, coords); h = self.up3(h, c, pseudo, coords)
        out = self.head(torch.cat([h, pseudo, coords], 1))
        return F.softplus(out + pseudo)                   # pseudo-map as residual anchor; output >= 0

class PatchDisc(nn.Module):
    """conditional PatchGAN: (map, pseudo) -> patch logits"""
    def __init__(self, ch=64):
        super().__init__()
        def blk(cin, cout, stride):
            return nn.Sequential(nn.Conv2d(cin, cout, 4, stride, 1), nn.GroupNorm(8, cout), nn.LeakyReLU(0.2))
        self.net = nn.Sequential(nn.Conv2d(2, ch, 4, 2, 1), nn.LeakyReLU(0.2),
                                 blk(ch, ch * 2, 2), blk(ch * 2, ch * 4, 2), blk(ch * 4, ch * 4, 1),
                                 nn.Conv2d(ch * 4, 1, 4, 1, 1))
    def forward(self, m, pseudo):
        return self.net(torch.cat([m, pseudo], 1))

# =============================================================================
# losses
# =============================================================================
_sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)[None, None] / 8
_sobel_y = _sobel_x.transpose(2, 3)
def masked_mse(pred, tgt, mask):
    return (((pred - tgt) ** 2) * mask).sum() / mask.sum().clamp(min=1)
def masked_grad_l1(pred, tgt, mask):
    sx, sy = _sobel_x.to(pred.device), _sobel_y.to(pred.device)
    gp = torch.cat([F.conv2d(pred, sx, padding=1), F.conv2d(pred, sy, padding=1)], 1)
    gt = torch.cat([F.conv2d(tgt, sx, padding=1), F.conv2d(tgt, sy, padding=1)], 1)
    m = F.max_pool2d(1 - mask, 3, 1, 1)                  # erode: exclude pixels touching invalid area
    m = (1 - m).expand_as(gp)
    return ((gp - gt).abs() * m).sum() / m.sum().clamp(min=1)

# =============================================================================
# train / evaluate
# =============================================================================
def masked_rmse_um(pred_um, tgt_um, disc_mask):
    mask = (tgt_um > 0) & ~disc_mask[None]
    return float(np.sqrt(np.mean(((pred_um - tgt_um) ** 2)[mask])))

@torch.no_grad()
def predict_maps(G, cache, idx, x_mu, x_sd, device, bs=256):
    G.eval(); out = np.zeros((len(idx), MAP_SIZE * MAP_SIZE), np.float32)
    dl = DataLoader(ClockMapDataset(cache, idx, x_mu, x_sd), batch_size=bs, shuffle=False, num_workers=NUM_WORKERS)
    for k, b in enumerate(dl):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            p = G(b['x_std'].to(device), b['x_um'].to(device))
        out[k * bs:k * bs + len(p)] = (p.float().cpu().numpy()[:, 0] * 100.0).reshape(len(p), -1)
    return out

def train_generator(cache, tr_idx, va_idx, x_mu, x_sd, disc_mask, device, epochs, tag, use_val=True):
    torch.manual_seed(SEED); np.random.seed(SEED)
    G, D = ClockGen().to(device), (PatchDisc().to(device) if ADV_WEIGHT > 0 else None)
    oG = torch.optim.AdamW(G.parameters(), lr=LR_G, weight_decay=WEIGHT_DECAY, betas=(0.5, 0.99))
    oD = torch.optim.AdamW(D.parameters(), lr=LR_D, weight_decay=WEIGHT_DECAY, betas=(0.5, 0.99)) if D else None
    dl = DataLoader(ClockMapDataset(cache, tr_idx, x_mu, x_sd), batch_size=BATCH, shuffle=True,
                    num_workers=NUM_WORKERS, drop_last=True, pin_memory=device.type == 'cuda', persistent_workers=NUM_WORKERS > 0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(oG, T_max=epochs * len(dl))
    dm_t = torch.tensor(disc_mask, dtype=torch.float32, device=device)[None, None]
    amp = device.type == 'cuda'
    best, best_ep, best_state, patience, hist = np.inf, 0, None, 0, []
    tgt_va = np.asarray(cache.maps[va_idx], dtype=np.float32) if (use_val and len(va_idx)) else None
    for ep in range(1, epochs + 1):
        G.train(); t0 = time.time(); agg = {'mse': 0, 'grad': 0, 'adv': 0, 'd': 0}; n = 0
        for b in dl:
            x_std, x_um, tgt = b['x_std'].to(device), b['x_um'].to(device), b['map'].to(device)
            mask = ((tgt > 0).float()) * (1 - dm_t)                                   # valid, non-disc
            pseudo = render_pseudo(x_um)
            # --- discriminator
            if D is not None:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                    fake = G(x_std, x_um).detach()
                    d_real, d_fake = D(tgt, pseudo), D(fake, pseudo)
                    d_loss = F.relu(1 - d_real).mean() + F.relu(1 + d_fake).mean()
                oD.zero_grad(set_to_none=True); d_loss.backward(); oD.step(); agg['d'] += d_loss.item()
            # --- generator
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                pred = G(x_std, x_um)
                l_mse = masked_mse(pred.float(), tgt, mask)
                l_grad = masked_grad_l1(pred.float(), tgt, mask) if GRAD_WEIGHT > 0 else torch.zeros((), device=device)
                l_adv = (-D(pred, pseudo).mean()) if D is not None else torch.zeros((), device=device)
                loss = l_mse + GRAD_WEIGHT * l_grad + ADV_WEIGHT * l_adv
            oG.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0); oG.step(); sched.step()
            agg['mse'] += l_mse.item(); agg['grad'] += l_grad.item(); agg['adv'] += l_adv.item(); n += 1
        line = f'[{tag}] ep{ep:02d} mse {agg["mse"]/n:.4f} grad {agg["grad"]/n:.4f} adv {agg["adv"]/n:+.3f} d {agg["d"]/n:.3f} ({time.time()-t0:.0f}s)'
        rec = {'epoch': ep, **{k: v / n for k, v in agg.items()}}
        if tgt_va is not None:
            pred_va = predict_maps(G, cache, va_idx, x_mu, x_sd, device)
            rm = masked_rmse_um(pred_va, tgt_va, disc_mask.ravel())
            rec['val_rmse_um'] = rm; line += f' | val RMSE {rm:.2f} um'
            if rm < best - 1e-3:
                best, best_ep, patience = rm, ep, 0
                best_state = {k: v.detach().cpu().clone() for k, v in G.state_dict().items()}
            else:
                patience += 1
        hist.append(rec); print(line)
        if tgt_va is not None and patience >= PATIENCE:
            print(f'[{tag}] early stop at ep{ep} (best ep{best_ep}, val RMSE {best:.2f})'); break
    if best_state is not None:
        G.load_state_dict(best_state)
    return G, (best_ep if tgt_va is not None else epochs), hist

# =============================================================================
# deployment decoder (importable)
# =============================================================================
class GenerativeDecoder:
    """Load the saved generative decoder and decode clocks -> 200x200 um map in OD orientation."""
    def __init__(self, npz_path, device=None):
        d = np.load(npz_path, allow_pickle=True)
        assert str(d['kind']) == 'generative', f'{npz_path} is not a generative decoder'
        self.mu, self.sd, self.disc_mask = d['clock_mean'].astype(np.float32), d['clock_sd'].astype(np.float32), d['disc_mask']
        self.mirror_os, self.cfg = bool(d['mirror_os']), json.loads(str(d['model_cfg']))
        w = os.path.join(os.path.dirname(npz_path), str(d['weights_file']))
        assert hashlib.sha256(open(w, 'rb').read()).hexdigest() == str(d['weights_sha256']), 'weights file hash mismatch'
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.G = ClockGen(base=int(self.cfg['base_ch'])).to(self.device)
        self.G.load_state_dict(torch.load(w, map_location=self.device, weights_only=True)); self.G.eval()
    @torch.no_grad()
    def decode(self, clocks_um, eye='OD'):
        c = np.asarray(clocks_um, dtype=np.float32)
        if self.mirror_os and eye == 'OS':
            c = mirror_clocks_os(c).astype(np.float32)
        x_std = torch.from_numpy((c - self.mu) / self.sd)[None].to(self.device)
        x_um = torch.from_numpy(c)[None].to(self.device)
        m = self.G(x_std, x_um)[0, 0].float().cpu().numpy() * 100.0
        m = m.astype(np.float64); m[self.disc_mask] = 0.0
        return m                                          # OD orientation; caller mirrors back for OS

# =============================================================================
# main
# =============================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(CACHE_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    meta = build_cache(); cache = MapCache(meta['n']); N = meta['n']
    print(f'\nscans {N:,} | patients {len(np.unique(cache.pids)):,} | OD {int((cache.eyes=="OD").sum()):,} OS {int((cache.eyes=="OS").sum()):,}')

    # scale check (ring vs clocks) on a sample
    smp = np.random.default_rng(SEED).choice(N, size=min(2000, N), replace=False)
    ring = np.array([map_means(np.asarray(cache.maps[j], np.float32).reshape(MAP_SIZE, MAP_SIZE))[1] for j in smp])
    print(f'scale check: ring mean {ring.mean():.1f} um | clock mean {cache.clocks[smp].mean():.1f} um | ratio {ring.mean()/cache.clocks[smp].mean():.3f}')

    # ---- honest evaluation split (patient-grouped) ----
    tr, te = next(GroupShuffleSplit(1, test_size=TEST_FRAC, random_state=SEED).split(np.zeros(N), groups=cache.pids))
    tr_fit, va = next(GroupShuffleSplit(1, test_size=0.05, random_state=SEED + 1).split(tr, groups=cache.pids[tr]))
    tr_fit, va = tr[tr_fit], tr[va]
    x_mu, x_sd = cache.clocks[tr].mean(0), cache.clocks[tr].std(0) + 1e-6
    rng = np.random.default_rng(SEED)
    sub = np.sort(rng.choice(tr, size=min(5000, len(tr)), replace=False))          # for disc mask + mean-map baseline
    sub_maps = np.asarray(cache.maps[sub], np.float32)
    disc_mask = ((sub_maps == 0).mean(0) > 0.5).reshape(MAP_SIZE, MAP_SIZE)
    pop_mean = sub_maps.mean(0); del sub_maps
    va = va[:min(len(va), 8000)]                                                     # cap per-epoch validation cost
    te_eval = np.sort(rng.choice(te, size=min(10000, len(te)), replace=False))     # cap held-out metric memory
    print(f'dev {len(tr):,} (fit {len(tr_fit):,} / val {len(va):,}) | held-out test {len(te):,} scans, '
          f'{len(np.unique(cache.pids[te])):,} patients (metrics on {len(te_eval):,})')

    G_dev, best_ep, hist_dev = train_generator(cache, tr_fit, va, x_mu, x_sd, disc_mask, device, EPOCHS, 'dev')

    # ---- held-out metrics ----
    te = te_eval
    pred_te = predict_maps(G_dev, cache, te, x_mu, x_sd, device)
    tgt_te = np.asarray(cache.maps[te], np.float32)                                 # te is sorted -> same order as pred_te
    dm = disc_mask.ravel()
    rmse = masked_rmse_um(pred_te, tgt_te, dm)
    pop_rmse = masked_rmse_um(np.broadcast_to(pop_mean, tgt_te.shape), tgt_te, dm)
    try:
        from skimage.metrics import structural_similarity as ssim
        ss = np.mean([ssim(tgt_te[i].reshape(MAP_SIZE, MAP_SIZE), pred_te[i].reshape(MAP_SIZE, MAP_SIZE), data_range=250.0)
                      for i in range(min(2000, len(te)))])
        ss_pop = np.mean([ssim(tgt_te[i].reshape(MAP_SIZE, MAP_SIZE), pop_mean.reshape(MAP_SIZE, MAP_SIZE), data_range=250.0)
                          for i in range(min(2000, len(te)))])
    except Exception:
        ss = ss_pop = float('nan')
    sec_err = np.nanmean([np.abs(sectors_to_clock_order(sector_means(pred_te[i].reshape(MAP_SIZE, MAP_SIZE))) - cache.clocks[te[i]])
                          for i in range(min(2000, len(te)))])
    print('\n' + '=' * 70 + f'\nHELD-OUT ({len(te):,} scans, patient-disjoint)\n' + '=' * 70)
    print(f'masked RMSE : generative {rmse:.2f} um | mean-map baseline {pop_rmse:.2f} um')
    print(f'SSIM        : generative {ss:.3f} | mean-map baseline {ss_pop:.3f}')
    print(f'ring-sector |generated - input clocks| : {sec_err:.1f} um (mean over 12 sectors)')

    # ---- example figure (held-out, no identifiers) ----
    def clock_color(v): return '#d94a43' if v < 70 else ('#f5e642' if v < 90 else '#5fbf5f')
    def draw_pie(ax, clocks, title):
        order = [12] + list(range(1, 12)); v = [clocks[h - 1] for h in order]
        ax.pie([1] * 12, colors=[clock_color(x) for x in v], labels=[f'{x:.0f}' for x in v], startangle=105,
               counterclock=False, labeldistance=1.12, wedgeprops=dict(edgecolor='black', linewidth=1.0), textprops=dict(fontsize=10))
        ax.set_title(title)
    sel = np.random.default_rng(SEED).choice(len(te), size=min(3, len(te)), replace=False)
    fig, axes = plt.subplots(len(sel), 3, figsize=(14, 4.5 * len(sel)), squeeze=False)
    for row, i in enumerate(sel):
        real, gen = tgt_te[i].reshape(MAP_SIZE, MAP_SIZE).astype(float), pred_te[i].reshape(MAP_SIZE, MAP_SIZE).astype(float)
        gen[disc_mask] = 0.0
        e = masked_rmse_um(gen.ravel()[None], real.ravel()[None], dm)
        rw, rr = map_means(real); gw, gr = map_means(gen)
        axes[row, 0].imshow(convert_array_to_pil(real)); axes[row, 0].set_title(f'real map — held-out scan {row+1}\nmean {rw:.0f} um | ring {rr:.0f} um'); axes[row, 0].axis('off')
        draw_pie(axes[row, 1], cache.clocks[te[i]], f'12 clock hours (um)\nmean {cache.clocks[te[i]].mean():.0f}')
        axes[row, 2].imshow(convert_array_to_pil(gen)); axes[row, 2].set_title(f'generative decoder\nmean {gw:.0f} um | ring {gr:.0f} um | RMSE {e:.1f} um'); axes[row, 2].axis('off')
    plt.suptitle(f'generative clock->map decoder — held-out patients (RMSE {rmse:.1f} um vs mean-map {pop_rmse:.1f} um)', fontsize=13)
    plt.tight_layout(); fig_path = os.path.join(OUT_DIR, 'generative_examples_heldout.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white'); plt.close(); print(f'figure -> {fig_path}')
    del G_dev; torch.cuda.empty_cache() if device.type == 'cuda' else None

    # ---- final refit on ALL scans for best_ep epochs ----
    n_ep = max(1, best_ep)
    print('\n' + '=' * 70 + f'\nFINAL DECODER: refit on all {N:,} scans for {n_ep} epochs\n' + '=' * 70)
    x_mu_all, x_sd_all = cache.clocks.mean(0), cache.clocks.std(0) + 1e-6
    sub_all = np.sort(np.random.default_rng(SEED).choice(N, size=min(5000, N), replace=False))
    disc_all = ((np.asarray(cache.maps[sub_all], np.float32) == 0).mean(0) > 0.5).reshape(MAP_SIZE, MAP_SIZE)
    G_all, _, hist_all = train_generator(cache, np.arange(N), np.array([], int), x_mu_all, x_sd_all, disc_all, device, n_ep, 'all', use_val=False)

    w_name = 'clock2map_generative_all_available.pth'; w_path = os.path.join(OUT_DIR, w_name)
    torch.save(G_all.state_dict(), w_path)
    w_sha = hashlib.sha256(open(w_path, 'rb').read()).hexdigest()
    model_cfg = {'base_ch': BASE_CH, 'map_size': MAP_SIZE, 'grad_weight': GRAD_WEIGHT, 'adv_weight': ADV_WEIGHT,
                 'register_cup': REGISTER_CUP, 'epochs': n_ep, 'batch': BATCH, 'lr_g': LR_G, 'lr_d': LR_D}
    npz_path = os.path.join(OUT_DIR, 'decoder_generative_clock2map_all_available.npz')
    np.savez_compressed(npz_path, kind='generative', clock_mean=x_mu_all.astype(np.float32), clock_sd=x_sd_all.astype(np.float32),
                        disc_mask=disc_all, k=0, map_size=MAP_SIZE, mirror_os=MIRROR_OS, units='um', um_per_axial_px=UM_PER_AXIAL_PX,
                        n_train=N, weights_file=w_name, weights_sha256=w_sha, model_cfg=json.dumps(model_cfg),
                        cfg=json.dumps({'feature': 'std', 'k': 0, 'weighting': 'none', 'rescale': False, 'texture': False}))
    report = {'scans_loaded': N, 'unique_patients': int(len(np.unique(cache.pids))), 'register_cup': REGISTER_CUP,
              'dev_scans': int(len(tr)), 'heldout_scans': int(len(te)), 'heldout_patients': int(len(np.unique(cache.pids[te]))),
              'best_epoch_dev': int(best_ep), 'heldout_masked_rmse_um': rmse, 'mean_map_baseline_rmse_um': pop_rmse,
              'heldout_ssim': ss, 'mean_map_baseline_ssim': ss_pop, 'heldout_sector_abs_error_um': float(sec_err),
              'adv_weight': ADV_WEIGHT, 'grad_weight': GRAD_WEIGHT, 'final_epochs': n_ep,
              'weights_file': w_name, 'weights_sha256': w_sha, 'history_dev': hist_dev, 'history_all': hist_all,
              'privacy_note': 'artifact contains network weights, clock scaler and a population disc mask only; no individual maps'}
    json.dump(report, open(os.path.join(OUT_DIR, 'decoder_generative_report_all_available.json'), 'w'), indent=2)
    print(f'\nsaved: {w_path}\n       {npz_path}\nDONE')

if __name__ == '__main__':
    main()
