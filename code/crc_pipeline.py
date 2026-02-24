"""
Conformal Risk Control (CRC) Pipeline for Multi-Speaker Localization
using SRP Spatial Spectrum.

NEW (2026-02):
- Calibration is now WATER-FILL / FLOOD-FILL REGION based on an ABSOLUTE threshold lambda in [0,1]
- We normalize each spatial spectrum S -> S_norm in [0,1]
- For each validated peak (matched to GT within ae_TH_deg), we grow/shrink a connected region:
      region(lambda) = connected component containing the peak within { S_norm >= lambda }
  (optionally restricted to the peak's Voronoi cell to prevent overlaps)
- For each matched GT, define lambda_star = largest lambda for which GT is still inside region(lambda)
- Choose lambda_hat as the finite-sample alpha-quantile of lambda_star such that:
      P(GT inside region(lambda_hat)) ≈ 1 - alpha

This file is meant to fully replace the old delta-based calibration.
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Optional, Any

# Optional import for plotting - gracefully handle missing matplotlib
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ============================================================
# Index <-> Angle conversion (FIXED azimuth grid)
# ============================================================

def idx_to_angles(ele_idx: int, azi_idx: int, nele: int, nazi: int) -> Tuple[float, float]:
    """
    Convert grid indices -> degrees.

    Elevation is assumed to be a closed grid [0, 180] with nele bins (includes endpoints).
    Azimuth is assumed to be a circular grid [-180, 180) with nazi bins (endpoint excluded).
    This matches: np.linspace(-180, 180, nazi, endpoint=False)
    """
    ele_deg = (ele_idx / (nele - 1)) * 180.0
    azi_deg = (azi_idx / nazi) * 360.0 - 180.0  # divide by nazi (not nazi-1)
    return float(ele_deg), float(azi_deg)


def angles_to_idx(ele_deg: float, azi_deg: float, nele: int, nazi: int) -> Tuple[int, int]:
    """
    Convert degrees -> grid indices.

    Elevation: clamp to [0,180], mapped to [0, nele-1]
    Azimuth: wrap to [-180,180), mapped to [0, nazi-1] circularly.
    """
    # Elevation: closed interval
    ele_deg = float(np.clip(ele_deg, 0.0, 180.0))
    ele_idx = int(np.round((ele_deg / 180.0) * (nele - 1)))
    ele_idx = int(np.clip(ele_idx, 0, nele - 1))

    # Azimuth: wrap to [-180, 180)
    azi_deg = float(((azi_deg + 180.0) % 360.0) - 180.0)
    azi_idx = int(np.round(((azi_deg + 180.0) / 360.0) * nazi)) % nazi
    return ele_idx, azi_idx


def wrap_azi_diff(a: float, b: float) -> float:
    """Minimal circular absolute azimuth diff in degrees."""
    return float(abs((a - b + 180.0) % 360.0 - 180.0))


# ============================================================
# Normalization (NEW)
# ============================================================

def normalize_map(S: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize spectrum to [0,1] per-frame.

    NOTE: If S is constant, returns all zeros.
    """
    S = np.asarray(S, dtype=np.float32)
    mn = float(np.min(S))
    mx = float(np.max(S))
    denom = (mx - mn)
    if denom < eps:
        return np.zeros_like(S, dtype=np.float32)
    return (S - mn) / (denom + eps)


# ============================================================
# Peak detection (same logic)
# ============================================================

def find_top2_peaks(S: np.ndarray, suppress_radius: Tuple[int, int] = (3, 3)) -> List[Tuple[int, int]]:
    """
    Find the top 2 peaks in the spatial spectrum using 8-neighbor strict peak detection.
    Returns list of (ele_idx, azi_idx), length <= 2.
    """
    nele, nazi = S.shape
    r_ele, r_azi = suppress_radius

    def azi_distance(a1, a2, n_azi):
        diff = abs(a1 - a2)
        return min(diff, n_azi - diff)

    def get_neighbor_values(S_, d_ele, d_azi):
        nele_, nazi_ = S_.shape
        neighbors = np.zeros_like(S_, dtype=np.float32)
        for e in range(nele_):
            for a in range(nazi_):
                ne = e + d_ele
                na = (a + d_azi) % nazi_
                if 0 <= ne < nele_:
                    neighbors[e, a] = S_[ne, na]
                else:
                    neighbors[e, a] = -np.inf
        return neighbors

    neighbors = [
        get_neighbor_values(S, -1, -1),
        get_neighbor_values(S, -1,  0),
        get_neighbor_values(S, -1,  1),
        get_neighbor_values(S,  0, -1),
        get_neighbor_values(S,  0,  1),
        get_neighbor_values(S,  1, -1),
        get_neighbor_values(S,  1,  0),
        get_neighbor_values(S,  1,  1),
    ]

    is_peak = np.ones((nele, nazi), dtype=bool)
    for neighbor in neighbors:
        is_peak &= (S > neighbor)

    peak_indices = np.where(is_peak)

    if len(peak_indices[0]) == 0:
        # fallback to global argmax
        flat_idx = int(np.argmax(S))
        peak1 = np.unravel_index(flat_idx, S.shape)
        peak_indices = (np.array([peak1[0]]), np.array([peak1[1]]))

    peak_values = S[peak_indices]
    sorted_idx = np.argsort(peak_values)[::-1]
    sorted_peaks = [(int(peak_indices[0][i]), int(peak_indices[1][i])) for i in sorted_idx]

    if len(sorted_peaks) == 0:
        return []

    peak1 = sorted_peaks[0]
    result = [peak1]
    peak1_ele, peak1_azi = peak1

    for peak_candidate in sorted_peaks[1:]:
        peak2_ele, peak2_azi = peak_candidate
        ele_dist = abs(peak2_ele - peak1_ele)
        azi_dist = azi_distance(peak2_azi, peak1_azi, nazi)
        if ele_dist > r_ele or azi_dist > r_azi:
            result.append(peak_candidate)
            break
    else:
        # fallback: suppress around peak1 and take next argmax
        S_suppressed = S.copy()
        for e in range(max(0, peak1_ele - r_ele), min(nele, peak1_ele + r_ele + 1)):
            for a_offset in range(-r_azi, r_azi + 1):
                a = (peak1_azi + a_offset) % nazi
                S_suppressed[e, a] = -np.inf
        flat_idx2 = int(np.argmax(S_suppressed))
        peak2 = np.unravel_index(flat_idx2, S_suppressed.shape)
        if S_suppressed[peak2] > -np.inf:
            result.append((int(peak2[0]), int(peak2[1])))

    return result[:2]


# ============================================================
# Voronoi masks (optional overlap prevention)
# ============================================================

def create_voronoi_masks(peaks: List[Tuple[int, int]], spectrum_shape: Tuple[int, int]) -> List[np.ndarray]:
    """
    Very simple Voronoi in INDEX space (ele,azi grid distances),
    with circular wrap in azimuth index.
    """
    if len(peaks) == 0:
        return []

    nele, nazi = spectrum_shape
    masks: List[np.ndarray] = []

    def azi_distance(a1, a2, n_azi):
        diff = abs(a1 - a2)
        return min(diff, n_azi - diff)

    for peak_idx, _ in enumerate(peaks):
        mask = np.zeros((nele, nazi), dtype=bool)
        for e in range(nele):
            for a in range(nazi):
                min_dist = np.inf
                closest_peak_idx = -1
                for other_idx, other_pos in enumerate(peaks):
                    ele_dist = abs(e - other_pos[0])
                    azi_dist = azi_distance(a, other_pos[1], nazi)
                    total_dist = np.sqrt(ele_dist**2 + azi_dist**2)
                    if total_dist < min_dist:
                        min_dist = total_dist
                        closest_peak_idx = other_idx
                if closest_peak_idx == peak_idx:
                    mask[e, a] = True
        masks.append(mask)

    return masks


# ============================================================
# Flood fill (NEW absolute-threshold semantics)
# ============================================================

def flood_fill_region_threshold(
    S_norm: np.ndarray,
    peak_idx: Tuple[int, int],
    thr: float,
    connectivity: int = 8,
    voronoi_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Connected component containing peak_idx within the set { S_norm >= thr }.
    - S_norm must be in [0,1]
    - thr must be in [0,1] (values outside are clipped)
    - If S_norm[peak] < thr -> returns empty mask (all False)
    """
    S_norm = np.asarray(S_norm, dtype=np.float32)
    nele, nazi = S_norm.shape
    pe, pa = peak_idx

    if not (0 <= pe < nele and 0 <= pa < nazi):
        raise ValueError(f"Peak {peak_idx} outside S shape {S_norm.shape}")

    thr = float(np.clip(thr, 0.0, 1.0))

    # If seed itself is below threshold, region is empty
    if float(S_norm[pe, pa]) < thr:
        return np.zeros((nele, nazi), dtype=bool)

    region_mask = np.zeros((nele, nazi), dtype=bool)
    visited = np.zeros((nele, nazi), dtype=bool)

    if connectivity == 4:
        neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                            (0, -1),           (0, 1),
                            (1, -1),  (1, 0),  (1, 1)]
    else:
        raise ValueError(f"Connectivity must be 4 or 8, got {connectivity}")

    q = deque([(pe, pa)])
    visited[pe, pa] = True
    region_mask[pe, pa] = True

    while q:
        ce, ca = q.popleft()
        for de, da in neighbor_offsets:
            ne = ce + de
            na = (ca + da) % nazi  # wrap in azimuth index

            if not (0 <= ne < nele):
                continue
            if visited[ne, na]:
                continue
            if voronoi_mask is not None and not voronoi_mask[ne, na]:
                visited[ne, na] = True
                continue

            visited[ne, na] = True
            if float(S_norm[ne, na]) >= thr:
                region_mask[ne, na] = True
                q.append((ne, na))

    return region_mask


# ============================================================
# Matching peaks to GT in DEGREES (validation gate A)
# ============================================================

def match_peaks_to_gt_degrees(
    pred_peaks: List[Tuple[int, int]],
    gt_angles_deg: List[Tuple[float, float]],
    nele: int,
    nazi: int,
    ae_TH_deg: float = 30.0,
    use_elevation: bool = False
) -> List[Optional[int]]:
    """
    Match GT speakers (given in degrees) to predicted peaks (indices),
    using small Hungarian/bruteforce (<=2x2) in degrees.
    Returns list length = len(gt_angles_deg), each entry is peak index or None (invalid).

    Cost:
      - default: azimuth wrap diff
      - if use_elevation: sqrt(az^2 + el^2)
    """
    if len(pred_peaks) == 0 or len(gt_angles_deg) == 0:
        return [None] * len(gt_angles_deg)

    n_pred = min(len(pred_peaks), 2)
    n_gt = min(len(gt_angles_deg), 2)

    pred_peaks = pred_peaks[:n_pred]
    gt_angles_deg = gt_angles_deg[:n_gt]

    pred_ang = [idx_to_angles(pe, pa, nele, nazi) for (pe, pa) in pred_peaks]  # (ele,azi) deg
    gt_ang = [(float(el), float(az)) for (el, az) in gt_angles_deg]

    C = np.zeros((n_gt, n_pred), dtype=np.float32)
    for gi in range(n_gt):
        gt_el, gt_az = gt_ang[gi]
        for pj in range(n_pred):
            pr_el, pr_az = pred_ang[pj]
            az = wrap_azi_diff(pr_az, gt_az)
            if use_elevation:
                el = abs(pr_el - gt_el)
                C[gi, pj] = float(np.sqrt(az**2 + el**2))
            else:
                C[gi, pj] = float(az)

    # brute force assignments for <=2
    assignment: List[Optional[int]] = [None] * n_gt
    if n_gt == 1 and n_pred == 1:
        assignment = [0]
    elif n_gt == 2 and n_pred == 2:
        cost1 = C[0, 0] + C[1, 1]
        cost2 = C[0, 1] + C[1, 0]
        assignment = [0, 1] if cost1 <= cost2 else [1, 0]
    elif n_gt == 1 and n_pred == 2:
        assignment = [0 if C[0, 0] <= C[0, 1] else 1]
    elif n_gt == 2 and n_pred == 1:
        assignment = [0, None] if C[0, 0] <= C[1, 0] else [None, 0]

    # validation gate by ae_TH_deg
    for gi in range(n_gt):
        pj = assignment[gi]
        if pj is None:
            continue
        if float(C[gi, pj]) > float(ae_TH_deg):
            assignment[gi] = None

    # extend to original len(gt_angles_deg) if >2 later
    result = assignment.copy()
    while len(result) < len(gt_angles_deg):
        result.append(None)
    return result


# ============================================================
# Calibration (NEW): lambda_star + alpha-quantile => lambda_hat
# ============================================================

def _lambda_star_for_one_gt(
    S_norm: np.ndarray,
    peak_idx: Tuple[int, int],
    gt_idx: Tuple[int, int],
    lambda_grid: np.ndarray,
    connectivity: int = 8,
    voronoi_mask: Optional[np.ndarray] = None
) -> float:
    """
    Scan lambda in increasing order. Region shrinks as lambda increases.
    Return lambda_star = largest lambda for which GT is inside region(lambda).
    If GT not even inside at lambda=0 -> return 0.0
    If GT inside for all lambdas up to 1 -> return 1.0
    """
    ge, ga = gt_idx
    nele, nazi = S_norm.shape
    if not (0 <= ge < nele and 0 <= ga < nazi):
        return 0.0

    last_ok = None
    for lam in lambda_grid:
        mask = flood_fill_region_threshold(S_norm, peak_idx, float(lam), connectivity, voronoi_mask)
        if mask[ge, ga]:
            last_ok = float(lam)
        else:
            # since lambda increases => region only shrinks, once it fails, it will fail for larger lambda
            break

    if last_ok is None:
        return 0.0
    return float(last_ok)


def calibrate_lambda_waterfill(
    spectra_cal: List[np.ndarray],
    doa_gt_deg_cal: List[np.ndarray],
    vad_gt_cal: Optional[List[np.ndarray]] = None,
    alpha: float = 0.1,
    ae_TH_deg: float = 30.0,
    use_elevation_in_cost: bool = False,
    prevent_overlap: bool = True,
    connectivity: int = 8,
    suppress_radius: Tuple[int, int] = (3, 3),
    lambda_grid: Optional[np.ndarray] = None,
    plot_calibration: bool = False
) -> Dict[str, Any]:
    """
    Calibrate lambda_hat for the WATER-FILL region method.

    Inputs per frame k:
      - spectra_cal[k]: (nele,nazi) spectrum
      - doa_gt_deg_cal[k]: (2,ns) degrees, where [0,:]=ele, [1,:]=azi
      - vad_gt_cal[k] (optional): (ns,) 0/1 or bool; if provided, only active GT are used.
        If vad_gt_cal is None -> uses all GT sources in doa_gt_deg_cal.

    Procedure:
      1) Normalize S -> S_norm in [0,1]
      2) Detect top2 peaks
      3) Match peaks <-> active GT in degrees with ae_TH_deg gate (validated peaks only)
      4) For each matched (GT,peak), compute lambda_star = largest lambda such that GT in region(lambda)
      5) lambda_hat = finite-sample alpha-quantile of lambda_star:
           k = ceil((n+1)*alpha) - 1
           lambda_hat = sorted(lambda_star)[k]
         so that P(GT inside region(lambda_hat)) ≈ 1-alpha

    Returns dict with:
      - lambda_hat
      - lambda_stars (np.ndarray)
      - n_frames, n_gt_used, n_matched
      - (optional) coverage_curve: (lambda_grid, coverage)
    """
    if lambda_grid is None:
        # Dense grid (hard calculation, no tricks)
        lambda_grid = np.linspace(0.0, 1.0, 1001, dtype=np.float32)
    else:
        lambda_grid = np.asarray(lambda_grid, dtype=np.float32)
        lambda_grid = np.clip(lambda_grid, 0.0, 1.0)

    lambda_stars: List[float] = []
    n_frames = 0
    n_gt_used = 0
    n_matched = 0

    for k, S in enumerate(spectra_cal):
        n_frames += 1
        S = np.asarray(S)
        nele, nazi = S.shape

        # normalize
        S_norm = normalize_map(S)

        # peaks on normalized map (recommended)
        peaks = find_top2_peaks(S_norm, suppress_radius=suppress_radius)
        if len(peaks) == 0:
            continue

        # active GT angles (degrees)
        gt_angles_all = np.asarray(doa_gt_deg_cal[k], dtype=np.float32)
        if gt_angles_all.ndim != 2 or gt_angles_all.shape[0] != 2:
            raise ValueError(f"doa_gt_deg_cal[{k}] must have shape (2,ns), got {gt_angles_all.shape}")
        ns = gt_angles_all.shape[1]

        if vad_gt_cal is not None:
            vad = np.asarray(vad_gt_cal[k]).astype(bool).reshape(-1)
            if vad.shape[0] != ns:
                raise ValueError(f"vad_gt_cal[{k}] length {vad.shape[0]} != ns {ns}")
            active_idx = np.where(vad)[0].tolist()
        else:
            active_idx = list(range(ns))

        if len(active_idx) == 0:
            continue

        gt_angles = [(float(gt_angles_all[0, i]), float(gt_angles_all[1, i])) for i in active_idx]
        n_gt_used += len(gt_angles)

        # match peaks to GT in degrees (validated peaks only)
        matched = match_peaks_to_gt_degrees(
            pred_peaks=peaks,
            gt_angles_deg=gt_angles,
            nele=nele,
            nazi=nazi,
            ae_TH_deg=ae_TH_deg,
            use_elevation=use_elevation_in_cost
        )

        # optional Voronoi masks
        voronoi_masks = None
        if prevent_overlap and len(peaks) > 1:
            voronoi_masks = create_voronoi_masks(peaks, (nele, nazi))

        # compute lambda_star per matched GT
        for gi, pj in enumerate(matched):
            if pj is None:
                continue  # A) only validated/matched peaks enter calibration (miss handling later)
            n_matched += 1

            gt_el, gt_az = gt_angles[gi]
            ge, ga = angles_to_idx(gt_el, gt_az, nele, nazi)

            peak_idx = peaks[int(pj)]
            vm = None
            if voronoi_masks is not None and int(pj) < len(voronoi_masks):
                vm = voronoi_masks[int(pj)]

            lam_star = _lambda_star_for_one_gt(
                S_norm=S_norm,
                peak_idx=peak_idx,
                gt_idx=(ge, ga),
                lambda_grid=lambda_grid,
                connectivity=connectivity,
                voronoi_mask=vm
            )
            lambda_stars.append(float(lam_star))

    lambda_stars_arr = np.asarray(lambda_stars, dtype=np.float64)

    if lambda_stars_arr.size == 0:
        # no matched samples -> cannot calibrate
        return {
            "lambda_hat": 0.0,
            "lambda_stars": lambda_stars_arr,
            "n_frames": n_frames,
            "n_gt_used": n_gt_used,
            "n_matched": n_matched,
            "alpha": float(alpha),
            "note": "No matched GT-peaks in calibration; returning 0.0"
        }

    # Finite-sample conformal quantile in LOWER tail (alpha-quantile)
    # lambda_hat = sort(lambda_star)[ceil((n+1)*alpha)-1]
    n = int(lambda_stars_arr.size)
    idx = int(np.ceil((n + 1) * float(alpha)) - 1)
    idx = max(0, min(idx, n - 1))

    srt = np.sort(lambda_stars_arr)
    lambda_hat = float(srt[idx])

    # sanity: empirical coverage on matched set:
    # GT in region(lambda_hat) <=> lambda_star >= lambda_hat
    emp_cov = float(np.mean(lambda_stars_arr >= lambda_hat) * 100.0)
    target_cov = float((1.0 - float(alpha)) * 100.0)

    out: Dict[str, Any] = {
        "lambda_hat": lambda_hat,
        "lambda_stars": lambda_stars_arr,
        "n_frames": n_frames,
        "n_gt_used": n_gt_used,
        "n_matched": n_matched,
        "alpha": float(alpha),
        "empirical_coverage_matched_percent": emp_cov,
        "target_coverage_percent": target_cov,
        "quantile_index": idx,
    }

    # optional coverage curve (cheap, no BFS: use lambda_star thresholding)
    # coverage(lambda) = P(lambda_star >= lambda)
    cov_curve = np.array([np.mean(lambda_stars_arr >= lam) for lam in lambda_grid], dtype=np.float32)
    out["coverage_curve"] = {
        "lambda_grid": lambda_grid,
        "coverage": cov_curve
    }

    if plot_calibration and MATPLOTLIB_AVAILABLE:
        plt.figure(figsize=(10, 5))
        plt.hist(lambda_stars_arr, bins=40, alpha=0.8, edgecolor='black', density=True)
        plt.axvline(lambda_hat, linestyle="--", linewidth=2)
        plt.xlabel("lambda_star (largest λ with GT inside)")
        plt.ylabel("density")
        plt.title(f"Waterfill calibration: alpha={alpha}  |  lambda_hat={lambda_hat:.3f}\n"
                  f"matched={n_matched}  empirical cov≈{emp_cov:.1f}% (target {target_cov:.1f}%)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(lambda_grid, cov_curve)
        plt.axhline(1.0 - float(alpha), linestyle="--")
        plt.axvline(lambda_hat, linestyle="--")
        plt.xlabel("lambda (absolute threshold on S_norm)")
        plt.ylabel("coverage on matched set")
        plt.title("Coverage curve (matched GT only)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return out


# ============================================================
# Test-time region extraction (UPDATED to absolute-threshold lambda on S_norm)
# ============================================================

def extract_regions_test(
    S: np.ndarray,
    lambda_hat: float,
    connectivity: int = 8,
    prevent_overlap: bool = True,
    suppress_radius: Tuple[int, int] = (3, 3),
    return_normalized: bool = False
) -> Dict[str, Any]:
    """
    Extract CP regions at test time.

    Returns dict:
      {
        'S_norm': (nele,nazi) optional,
        'peaks': [(pe,pa),...],
        'regions': [ region_info, ... ],
        'lambda_used': lambda_hat
      }

    Region definition:
      region = connected component containing peak within { S_norm >= lambda_hat }
      (optionally restricted to Voronoi cell)
    """
    S = np.asarray(S)
    nele, nazi = S.shape
    S_norm = normalize_map(S)

    peaks = find_top2_peaks(S_norm, suppress_radius=suppress_radius)
    if len(peaks) == 0:
        return {"peaks": [], "regions": [], "lambda_used": float(lambda_hat), "note": "no peaks"}

    voronoi_masks = None
    if prevent_overlap and len(peaks) > 1:
        voronoi_masks = create_voronoi_masks(peaks, (nele, nazi))

    regions: List[Dict[str, Any]] = []
    lambda_hat = float(np.clip(lambda_hat, 0.0, 1.0))

    # grids for visualization / bounds conversion
    ele_grid = np.linspace(0.0, 180.0, nele, endpoint=True)
    azi_grid = np.linspace(-180.0, 180.0, nazi, endpoint=False)

    for j, peak in enumerate(peaks):
        vm = voronoi_masks[j] if voronoi_masks is not None else None
        region_mask = flood_fill_region_threshold(S_norm, peak, lambda_hat, connectivity, vm)
        region_size = int(np.sum(region_mask))

        pe, pa = peak
        peak_value = float(S_norm[pe, pa])

        info: Dict[str, Any] = {
            "peak_position_idx": (int(pe), int(pa)),
            "peak_position_deg": idx_to_angles(int(pe), int(pa), nele, nazi),
            "peak_value": peak_value,          # normalized peak value
            "lambda_used": lambda_hat,         # absolute threshold
            "threshold": lambda_hat,           # same as lambda_used (clarity)
            "region_mask": region_mask,
            "region_size": region_size,
        }

        if region_size > 0:
            coords = np.where(region_mask)
            info["region_bounds"] = {
                "ele_min": int(np.min(coords[0])),
                "ele_max": int(np.max(coords[0])),
                "azi_min": int(np.min(coords[1])),
                "azi_max": int(np.max(coords[1])),
            }

            # weighted centroid (circular mean in azimuth index)
            weights = np.maximum(S_norm - lambda_hat, 0.0) * region_mask
            if float(np.sum(weights)) > 0.0:
                w = weights[coords]
                ele_w = float(np.sum(coords[0] * w) / np.sum(w))

                azi_idx = coords[1].astype(np.float64)
                azi_w = w.astype(np.float64)

                azi_angles = 2 * np.pi * azi_idx / float(nazi)
                cplx = np.exp(1j * azi_angles)
                mean_cplx = np.sum(cplx * azi_w) / (np.sum(azi_w) + 1e-12)
                mean_angle = np.angle(mean_cplx)
                azi_w_idx = float((mean_angle * float(nazi) / (2 * np.pi)) % float(nazi))

                info["weighted_centroid"] = (ele_w, azi_w_idx)
                # also centroid in degrees for convenience
                ce = int(np.clip(round(ele_w), 0, nele - 1))
                ca = int(np.clip(round(azi_w_idx), 0, nazi - 1))
                info["weighted_centroid_deg"] = (float(ele_grid[ce]), float(azi_grid[ca]))
            else:
                # fallback to mean indices
                info["weighted_centroid"] = (float(np.mean(coords[0])), float(np.mean(coords[1])))
        else:
            info["region_bounds"] = {"ele_min": int(pe), "ele_max": int(pe), "azi_min": int(pa), "azi_max": int(pa)}
            info["weighted_centroid"] = (float(pe), float(pa))

        regions.append(info)

    out: Dict[str, Any] = {
        "peaks": peaks,
        "regions": regions,
        "lambda_used": lambda_hat,
    }
    if return_normalized:
        out["S_norm"] = S_norm
    return out


# ============================================================
# Visualization stub (optional)
# ============================================================

def visualize_regions(
    S: np.ndarray,
    regions_out: Dict[str, Any],
    gt_angles_deg: Optional[List[Tuple[float, float]]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Simple visualization helper (optional).
    - Draw S_norm heatmap
    - Plot peaks and rectangle bounds of regions
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    S = np.asarray(S)
    nele, nazi = S.shape
    S_norm = normalize_map(S)

    ele_grid = np.linspace(0.0, 180.0, nele, endpoint=True)
    azi_grid = np.linspace(-180.0, 180.0, nazi, endpoint=False)

    plt.figure(figsize=(8, 6))
    plt.imshow(S_norm, origin="lower", aspect="auto", extent=[azi_grid[0], azi_grid[-1], ele_grid[0], ele_grid[-1]])
    plt.colorbar(label="S_norm")

    peaks = regions_out.get("peaks", [])
    for i, (pe, pa) in enumerate(peaks):
        el, az = idx_to_angles(pe, pa, nele, nazi)
        plt.scatter(az, el, marker="o", s=70, facecolors="none", edgecolors="cyan", linewidths=2,
                    label="peaks" if i == 0 else None)

    for i, reg in enumerate(regions_out.get("regions", [])):
        b = reg.get("region_bounds", None)
        if b is None:
            continue
        ele_min = ele_grid[b["ele_min"]]
        ele_max = ele_grid[b["ele_max"]]
        azi_min = azi_grid[b["azi_min"]]
        azi_max = azi_grid[b["azi_max"]]
        rect_azi = [azi_min, azi_max, azi_max, azi_min, azi_min]
        rect_ele = [ele_min, ele_min, ele_max, ele_max, ele_min]
        plt.plot(rect_azi, rect_ele, linewidth=2, label=f"region{i+1}")

    if gt_angles_deg is not None:
        for i, (el, az) in enumerate(gt_angles_deg):
            az = ((az + 180.0) % 360.0) - 180.0
            el = float(np.clip(el, 0.0, 180.0))
            plt.scatter(az, el, marker="x", s=90, linewidths=2.5, c="red",
                        label="GT" if i == 0 else None)

    plt.xlabel("Azimuth [deg]")
    plt.ylabel("Elevation [deg]")
    plt.title(f"Regions (lambda={regions_out.get('lambda_used', None)})")
    plt.legend(loc="upper right")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    print("=== CRC Pipeline Self-Test (Waterfill calibration) ===")

    nele, nazi = 37, 73
    rng = np.random.default_rng(0)

    # create a toy spectrum with two blobs
    S = rng.normal(0.0, 0.03, (nele, nazi)).astype(np.float32)

    blob1_idx = (10, 12)
    blob2_idx = (25, 50)

    for e in range(nele):
        for a in range(nazi):
            d1 = np.sqrt((e - blob1_idx[0])**2 + (a - blob1_idx[1])**2)
            d2 = np.sqrt((e - blob2_idx[0])**2 + (a - blob2_idx[1])**2)
            S[e, a] += 1.0 * np.exp(-(d1**2) / 10.0)
            S[e, a] += 0.7 * np.exp(-(d2**2) / 12.0)

    # GT in degrees (close to blobs)
    gt1_deg = idx_to_angles(blob1_idx[0], blob1_idx[1], nele, nazi)
    gt2_deg = idx_to_angles(blob2_idx[0], blob2_idx[1], nele, nazi)

    # build calibration list
    spectra_cal = [S.copy() for _ in range(20)]
    doa_gt_deg_cal = [np.array([[gt1_deg[0], gt2_deg[0]], [gt1_deg[1], gt2_deg[1]]], dtype=np.float32) for _ in range(20)]
    vad_gt_cal = [np.array([1, 1], dtype=np.float32) for _ in range(20)]

    cal = calibrate_lambda_waterfill(
        spectra_cal=spectra_cal,
        doa_gt_deg_cal=doa_gt_deg_cal,
        vad_gt_cal=vad_gt_cal,
        alpha=0.1,
        ae_TH_deg=30.0,
        prevent_overlap=True,
        connectivity=8,
        plot_calibration=False
    )

    print("Calibration output keys:", list(cal.keys()))
    print("lambda_hat:", cal["lambda_hat"])
    print("matched:", cal["n_matched"], "emp_cov_matched%:", cal["empirical_coverage_matched_percent"])

    # test extraction
    out = extract_regions_test(S, lambda_hat=cal["lambda_hat"], return_normalized=False)
    print("test peaks:", out["peaks"])
    print("region sizes:", [r["region_size"] for r in out["regions"]])

    if MATPLOTLIB_AVAILABLE:
        visualize_regions(S, out, gt_angles_deg=[gt1_deg, gt2_deg], save_path=None)
def flood_fill_region_threshold(S: np.ndarray,
                                peak_idx: Tuple[int, int],
                                lambda_thr: float,
                                connectivity: int = 8,
                                voronoi_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    NEW: region = connected component around peak in the set { S >= lambda_thr }.
    S is assumed normalized to [0,1]. lambda_thr in [0,1].
    """
    nele, nazi = S.shape
    pe, pa = peak_idx
    if not (0 <= pe < nele and 0 <= pa < nazi):
        raise ValueError("peak out of bounds")

    # if peak itself is below threshold -> empty region (or singleton False)
    if S[pe, pa] < lambda_thr:
        region = np.zeros((nele, nazi), dtype=bool)
        return region

    region_mask = np.zeros((nele, nazi), dtype=bool)
    visited = np.zeros((nele, nazi), dtype=bool)

    from collections import deque
    q = deque([(pe, pa)])
    visited[pe, pa] = True
    region_mask[pe, pa] = True

    if connectivity == 4:
        neigh = [(-1,0),(1,0),(0,-1),(0,1)]
    else:
        neigh = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    while q:
        ce, ca = q.popleft()
        for de, da in neigh:
            ne = ce + de
            na = (ca + da) % nazi
            if not (0 <= ne < nele):
                continue
            if visited[ne, na]:
                continue
            if voronoi_mask is not None and not voronoi_mask[ne, na]:
                continue
            visited[ne, na] = True
            if S[ne, na] >= lambda_thr:
                region_mask[ne, na] = True
                q.append((ne, na))

    return region_mask
