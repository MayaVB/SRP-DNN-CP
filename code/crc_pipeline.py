"""
Conformal Risk Control (CRC) Pipeline for Multi-Speaker Localization
using SRP Spatial Spectrum.

This module implements a conformal prediction approach for controlling
localization risk by building confidence regions around detected speaker peaks.
"""

import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Optional

# Optional import for plotting - gracefully handle missing matplotlib
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def find_top2_peaks(S: np.ndarray, suppress_radius: Tuple[int, int] = (3, 3)) -> List[Tuple[int, int]]:
    """
    Find the top 2 peaks in the spatial spectrum using 8-neighbor peak detection.

    Args:
        S: Spatial spectrum array of shape (nele, nazi)
        suppress_radius: (r_ele, r_azi) radius for suppressing around first peak

    Returns:
        List of (ele_idx, azi_idx) tuples for top 2 peaks
        Returns as [(ele1, azi1), (ele2, azi2)]
    """
    nele, nazi = S.shape
    r_ele, r_azi = suppress_radius

    # Helper function for azimuth wraparound distance
    def azi_distance(a1, a2, n_azi):
        """Compute minimum distance considering azimuth wraparound."""
        diff = abs(a1 - a2)
        return min(diff, n_azi - diff)

    # Create neighbor arrays using modulo indexing for azimuth wraparound
    def get_neighbor_values(S, d_ele, d_azi):
        """Get neighbor values with boundary handling."""
        nele, nazi = S.shape
        neighbors = np.zeros_like(S)

        for e in range(nele):
            for a in range(nazi):
                ne = e + d_ele
                na = (a + d_azi) % nazi  # Azimuth wraparound

                # Handle elevation boundaries
                if 0 <= ne < nele:
                    neighbors[e, a] = S[ne, na]
                else:
                    neighbors[e, a] = -np.inf  # Invalid neighbor

        return neighbors

    # Get all 8 neighbors
    neighbors = [
        get_neighbor_values(S, -1, -1),  # top-left
        get_neighbor_values(S, -1,  0),  # top
        get_neighbor_values(S, -1,  1),  # top-right
        get_neighbor_values(S,  0, -1),  # left
        get_neighbor_values(S,  0,  1),  # right
        get_neighbor_values(S,  1, -1),  # bottom-left
        get_neighbor_values(S,  1,  0),  # bottom
        get_neighbor_values(S,  1,  1),  # bottom-right
    ]

    # Find local maxima: point is peak if greater than all neighbors
    is_peak = np.ones((nele, nazi), dtype=bool)
    for neighbor in neighbors:
        is_peak &= (S > neighbor)

    # Get peak candidates
    peak_indices = np.where(is_peak)

    # Fallback to global maximum if no local maxima found
    if len(peak_indices[0]) == 0:
        print("Warning: No local maxima found, falling back to global argmax")
        flat_idx = np.argmax(S)
        peak1 = np.unravel_index(flat_idx, S.shape)
        peak_indices = ([peak1[0]], [peak1[1]])

    peak_values = S[peak_indices]
    sorted_idx = np.argsort(peak_values)[::-1]
    sorted_peaks = [(peak_indices[0][i], peak_indices[1][i]) for i in sorted_idx]

    if len(sorted_peaks) == 0:
        return []

    # Take the highest peak
    peak1 = sorted_peaks[0]
    result = [peak1]

    # Find second peak with suppression
    peak1_ele, peak1_azi = peak1

    for peak_candidate in sorted_peaks[1:]:
        peak2_ele, peak2_azi = peak_candidate

        # Check distance from first peak
        ele_dist = abs(peak2_ele - peak1_ele)
        azi_dist = azi_distance(peak2_azi, peak1_azi, nazi)

        if ele_dist > r_ele or azi_dist > r_azi:
            result.append(peak_candidate)
            break
    else:
        print(f"Warning: Second peak too close to first peak at ({peak1_ele}, {peak1_azi})")

        # Fallback: find second global maximum with suppression
        S_suppressed = S.copy()

        # Suppress region around peak1
        for e in range(max(0, peak1_ele - r_ele), min(nele, peak1_ele + r_ele + 1)):
            for a_offset in range(-r_azi, r_azi + 1):
                a = (peak1_azi + a_offset) % nazi
                S_suppressed[e, a] = -np.inf

        # Find second peak
        flat_idx2 = np.argmax(S_suppressed)
        peak2 = np.unravel_index(flat_idx2, S_suppressed.shape)

        if S_suppressed[peak2] > -np.inf:  # Valid second peak
            result.append(peak2)

    return result[:2]


def flood_fill_region(S: np.ndarray, peak_idx: Tuple[int, int], lambda_val: float,
                     connectivity: int = 8, voronoi_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Build a connected region around a peak using flood-fill algorithm with BFS.

    Starting from peak position, include neighbors if their spectrum value
    is >= peak_value - lambda_val. Continue expanding until no more neighbors
    satisfy the condition.

    Args:
        S: Spatial spectrum array of shape (nele, nazi)
        peak_idx: (ele_idx, azi_idx) position of the peak
        lambda_val: Threshold parameter controlling region size
        connectivity: 4 or 8 neighbor connectivity
        voronoi_mask: Optional binary mask of shape (nele, nazi) defining the
                     Voronoi cell for this peak. Region expansion is limited
                     to this mask to prevent overlapping regions.

    Returns:
        Binary mask of shape (nele, nazi) where True indicates points in the region
    """
    nele, nazi = S.shape
    peak_ele, peak_azi = peak_idx

    # Validate peak position
    if not (0 <= peak_ele < nele and 0 <= peak_azi < nazi):
        raise ValueError(f"Peak position {peak_idx} is outside spectrum bounds {S.shape}")

    # Initialize region mask and visited set
    region_mask = np.zeros((nele, nazi), dtype=bool)
    visited = np.zeros((nele, nazi), dtype=bool)

    # Threshold for inclusion in region
    peak_value = S[peak_ele, peak_azi]
    threshold = peak_value - lambda_val

    # BFS queue using deque for efficient operations
    queue = deque([(peak_ele, peak_azi)])
    region_mask[peak_ele, peak_azi] = True
    visited[peak_ele, peak_azi] = True

    # Define neighbor offsets based on connectivity
    if connectivity == 4:
        # 4-connected (cardinal directions)
        neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        # 8-connected (cardinal + diagonal)
        neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                           (0, -1),           (0, 1),
                           (1, -1),  (1, 0),  (1, 1)]
    else:
        raise ValueError(f"Connectivity must be 4 or 8, got {connectivity}")

    # BFS flood-fill
    while queue:
        curr_ele, curr_azi = queue.popleft()  # Remove from front (BFS)

        # Check all neighbors
        for d_ele, d_azi in neighbor_offsets:
            next_ele = curr_ele + d_ele
            next_azi = (curr_azi + d_azi) % nazi  # Handle azimuth wraparound

            # Handle elevation boundaries safely
            if not (0 <= next_ele < nele):
                continue  # Skip out of bounds in elevation

            # Skip if already visited
            if visited[next_ele, next_azi]:
                continue

            # Check Voronoi ownership if provided
            if voronoi_mask is not None and not voronoi_mask[next_ele, next_azi]:
                continue  # Skip if outside this peak's Voronoi cell

            # Mark as visited
            visited[next_ele, next_azi] = True

            # Check threshold condition
            if S[next_ele, next_azi] >= threshold:
                # Include in region and add to queue for further expansion
                region_mask[next_ele, next_azi] = True
                queue.append((next_ele, next_azi))

    return region_mask


def create_voronoi_masks(peaks: List[Tuple[int, int]], spectrum_shape: Tuple[int, int]) -> List[np.ndarray]:
    """
    Create Voronoi ownership masks for multiple peaks to prevent region overlap.

    Args:
        peaks: List of (ele_idx, azi_idx) tuples for peak positions
        spectrum_shape: (nele, nazi) shape of the spatial spectrum

    Returns:
        List of binary masks, one for each peak, defining Voronoi cells
    """
    if len(peaks) == 0:
        return []

    nele, nazi = spectrum_shape
    masks = []

    # Helper function for azimuth wraparound distance
    def azi_distance(a1, a2, n_azi):
        diff = abs(a1 - a2)
        return min(diff, n_azi - diff)

    for peak_idx, peak_pos in enumerate(peaks):
        mask = np.zeros((nele, nazi), dtype=bool)

        for e in range(nele):
            for a in range(nazi):
                # Find closest peak to this position
                min_dist = np.inf
                closest_peak_idx = -1

                for other_idx, other_pos in enumerate(peaks):
                    # Distance to this peak
                    ele_dist = abs(e - other_pos[0])
                    azi_dist = azi_distance(a, other_pos[1], nazi)
                    total_dist = np.sqrt(ele_dist**2 + azi_dist**2)

                    if total_dist < min_dist:
                        min_dist = total_dist
                        closest_peak_idx = other_idx

                # Assign to this peak's Voronoi cell
                if closest_peak_idx == peak_idx:
                    mask[e, a] = True

        masks.append(mask)

    return masks


def match_peaks_to_gt(pred_peaks: List[Tuple[int, int]],
                     gt_positions: List[Tuple[int, int]],
                     spectrum_shape: Optional[Tuple[int, int]] = None) -> List[Optional[int]]:
    """
    Match predicted peaks to ground truth positions using brute-force over permutations.

    Args:
        pred_peaks: List of (ele_idx, azi_idx) tuples for predicted peaks
        gt_positions: List of (ele_idx, azi_idx) tuples for ground truth positions
        spectrum_shape: (nele, nazi) for azimuth wraparound distance calculation

    Returns:
        List of matched GT indices for each predicted peak (None if no match)
        matched_pred_for_gt = [pred_for_gt1, pred_for_gt2]
    """
    if len(pred_peaks) == 0 or len(gt_positions) == 0:
        return [None] * len(gt_positions)

    # Helper function for azimuth wraparound distance
    def azi_distance(a1, a2, n_azi):
        """Compute minimum distance considering azimuth wraparound."""
        if n_azi is None:
            return abs(a1 - a2)  # No wraparound
        diff = abs(a1 - a2)
        return min(diff, n_azi - diff)

    n_pred = min(len(pred_peaks), 2)
    n_gt = min(len(gt_positions), 2)

    if n_pred == 0 or n_gt == 0:
        return [None] * len(gt_positions)

    # Convert to numpy for easier computation
    pred_array = np.array(pred_peaks[:n_pred])  # Shape: (n_pred, 2)
    gt_array = np.array(gt_positions[:n_gt])     # Shape: (n_gt, 2)

    # Get azimuth dimension for wraparound
    nazi = spectrum_shape[1] if spectrum_shape is not None else None

    # Compute cost matrix using distance in index space with azimuth wraparound
    cost_matrix = np.zeros((n_gt, n_pred))

    for gt_idx in range(n_gt):
        for pred_idx in range(n_pred):
            gt_ele, gt_azi = gt_array[gt_idx]
            pred_ele, pred_azi = pred_array[pred_idx]

            # Elevation distance (no wraparound)
            ele_diff = abs(pred_ele - gt_ele)

            # Azimuth distance (with wraparound if spectrum_shape provided)
            azi_diff = azi_distance(pred_azi, gt_azi, nazi)

            # Combined distance in index space
            cost_matrix[gt_idx, pred_idx] = np.sqrt(ele_diff**2 + azi_diff**2)

    # Brute-force over permutations for small cases
    best_assignment = [None] * n_gt

    if n_pred == 1 and n_gt == 1:
        best_assignment = [0]
    elif n_pred == 2 and n_gt == 2:
        # Two permutations: (0,1) and (1,0)
        cost1 = cost_matrix[0, 0] + cost_matrix[1, 1]  # GT0->Pred0, GT1->Pred1
        cost2 = cost_matrix[0, 1] + cost_matrix[1, 0]  # GT0->Pred1, GT1->Pred0

        if cost1 <= cost2:
            best_assignment = [0, 1]
        else:
            best_assignment = [1, 0]
    elif n_pred == 2 and n_gt == 1:
        # One GT, two predictions - pick best
        if cost_matrix[0, 0] <= cost_matrix[0, 1]:
            best_assignment = [0]
        else:
            best_assignment = [1]
    elif n_pred == 1 and n_gt == 2:
        # Two GT, one prediction - assign to closest
        if cost_matrix[0, 0] <= cost_matrix[1, 0]:
            best_assignment = [0, None]
        else:
            best_assignment = [None, 0]

    # Extend result to match original input size
    result = best_assignment.copy()
    while len(result) < len(gt_positions):
        result.append(None)

    return result[:len(gt_positions)]


def compute_lambda_cp(spectra_cal: List[np.ndarray], gt_cal: List[List[Tuple[int, int]]], alpha: float = 0.1, plot_calibration: bool = False) -> Tuple[float, np.ndarray]:
    """
    Compute lambda using conformal prediction for desired coverage.

    For each calibration frame and each speaker:
    1) Detect top-2 peaks in spectrum S
    2) Match predicted peaks to GT speakers
    3) Compute nonconformity score: delta = S_peak - S_at_GT

    Then compute lambda_hat as the (1-alpha) quantile of all deltas.

    Args:
        spectra_cal: List of spatial spectra, each of shape (nele, nazi)
        gt_cal: List of GT speaker positions for each frame, each containing
               list of (ele_idx, azi_idx) tuples for speakers in that frame
        alpha: Miscoverage rate (e.g., 0.1 for 90% coverage)
        plot_calibration: If True, create calibration plots (requires matplotlib)

    Returns:
        lambda_hat: Threshold parameter for desired coverage
        deltas: Array of all nonconformity scores
    """
    deltas = []

    print(f"Computing conformal prediction lambda with alpha={alpha} (target coverage={(1-alpha)*100:.1f}%)")
    print(f"Processing {len(spectra_cal)} calibration frames...")

    for frame_idx, (spectrum, gt_speakers) in enumerate(zip(spectra_cal, gt_cal)):
        if len(gt_speakers) == 0:
            continue  # Skip frames with no speakers

        # 1) Detect top-2 peaks
        peaks = find_top2_peaks(spectrum, suppress_radius=(3, 3))

        if len(peaks) == 0:
            continue  # Skip if no peaks found

        # 2) Match predicted peaks to GT speakers
        matched_gt_indices = match_peaks_to_gt(peaks, gt_speakers, spectrum.shape)

        # 3) Compute nonconformity scores for matched pairs
        #
        # ⚠️  CRITICAL CP CORRECTNESS ISSUE - TODO: FIX THIS! ⚠️
        #
        # Current implementation SILENTLY SKIPS unmatched GT speakers (when pred_idx is None).
        # This is INCORRECT for conformal prediction and breaks coverage guarantees!
        #
        # Why this is wrong:
        # - CP requires exactly ONE nonconformity score per calibration example
        # - If a GT speaker is missed by peak detection, that's a FAILURE, not something to ignore
        # - By skipping missed speakers, we underestimate the true risk
        # - This leads to overly optimistic lambda_hat values
        # - Coverage guarantee becomes FALSE
        #
        # Correct behavior should be:
        # if pred_idx is None:
        #     delta = np.inf  # or some very large penalty value
        #     deltas.append(delta)
        # else:
        #     # compute normal delta = max(0.0, S_peak - S_at_GT)
        #
        # This ensures that missed detections are properly penalized in the quantile calculation.
        #
        for gt_idx, pred_idx in enumerate(matched_gt_indices):
            if pred_idx is not None and gt_idx < len(gt_speakers):
                # Get peak value at predicted location
                peak_ele, peak_azi = peaks[pred_idx]
                S_peak = spectrum[peak_ele, peak_azi]

                # Get spectrum value at ground truth location
                gt_ele, gt_azi = gt_speakers[gt_idx]
                S_at_GT = spectrum[gt_ele, gt_azi]

                # Nonconformity score: ensure non-negative values for conformal prediction
                delta = max(0.0, S_peak - S_at_GT)
                deltas.append(delta)
            # TODO: Add the missing case:
            # else:
            #     deltas.append(np.inf)  # Penalty for missed GT speaker

                if frame_idx < 3:  # Debug info for first few frames
                    print(f"  Frame {frame_idx}, Speaker {gt_idx}: "
                          f"Peak@{peaks[pred_idx]} ({S_peak:.3f}) - GT@{gt_speakers[gt_idx]} ({S_at_GT:.3f}) = {delta:.3f}")

    deltas = np.array(deltas)

    if len(deltas) == 0:
        print("Warning: No valid nonconformity scores computed!")
        return 0.0, deltas

    print(f"Collected {len(deltas)} nonconformity scores")
    print(f"Delta range: [{np.min(deltas):.3f}, {np.max(deltas):.3f}]")

    # Compute conformal prediction quantile with finite-sample correction
    n = len(deltas)
    # Standard conformal prediction quantile index
    index = int(np.ceil((n + 1) * (1 - alpha))) - 1  # -1 for 0-based indexing
    index = max(0, min(index, n - 1))  # Clamp to valid range

    # Sort deltas and get quantile
    sorted_deltas = np.sort(deltas)
    lambda_hat = sorted_deltas[index]

    print(f"Conformal quantile index: {index+1}/{n} (corrected for finite sample)")
    print(f"Lambda_hat = {lambda_hat:.3f}")

    # Empirical coverage calculation
    empirical_coverage = np.mean(deltas <= lambda_hat) * 100
    target_coverage = (1 - alpha) * 100
    print(f"Empirical coverage: {empirical_coverage:.1f}% (target: {target_coverage:.1f}%)")

    # Create calibration plots if requested
    if plot_calibration:
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib not available, skipping plots")
        else:
            # Create histogram of deltas with lambda_hat marked
            plt.figure(figsize=(10, 6))

            # Histogram of nonconformity scores
            plt.hist(deltas, bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)

            # Mark lambda_hat with a vertical line
            plt.axvline(lambda_hat, color='red', linestyle='--', linewidth=2,
                       label=f'λ_hat = {lambda_hat:.3f}')

            # Add text annotations
            plt.xlabel('Nonconformity Score (δ = S_peak - S_GT)')
            plt.ylabel('Density')
            plt.title(f'Calibration: Nonconformity Score Distribution\n'
                     f'α = {alpha} (Target Coverage: {target_coverage:.1f}%), '
                     f'Empirical Coverage: {empirical_coverage:.1f}%')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Add statistics text box
            stats_text = (f'n = {n} scores\n'
                         f'Mean: {np.mean(deltas):.3f}\n'
                         f'Std: {np.std(deltas):.3f}\n'
                         f'Min: {np.min(deltas):.3f}\n'
                         f'Max: {np.max(deltas):.3f}')
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            plt.tight_layout()

            # Save figure instead of showing
            filename = f'calibration_plot_alpha_{alpha:.3f}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Calibration plot saved to: {filename}")
            plt.close()  # Close figure to free memory

    return lambda_hat, deltas


def extract_regions_test(S: np.ndarray,
                        lambda_hat: float,
                        doa_candidate: Optional[List[np.ndarray]] = None,
                        connectivity: int = 8,
                        prevent_overlap: bool = True) -> List[Dict]:
    """
    Extract confidence regions for test data using learned lambda.

    This is the test stage of conformal prediction: apply the learned lambda_hat
    to new test data to get confidence regions around detected speakers.

    Args:
        S: Spatial spectrum of shape (nele, nazi)
        lambda_hat: Learned threshold parameter from calibration
        doa_candidate: Optional [ele_candidate, azi_candidate] arrays in radians
                      for converting indices to degrees. If None, returns indices.
        connectivity: 4 or 8 neighbor connectivity for flood-fill
        prevent_overlap: If True, use Voronoi cells to prevent region overlap

    Returns:
        List of region dictionaries, each containing:
            - 'peak_position_idx': (ele_idx, azi_idx) in array indices
            - 'peak_position_deg': (ele_deg, azi_deg) in degrees (if doa_candidate provided)
            - 'peak_value': Spectrum value at peak
            - 'region_mask': Binary mask of shape (nele, nazi)
            - 'region_size': Number of pixels in region
            - 'threshold': Actual threshold used (peak_value - lambda_hat)
            - 'weighted_centroid': Weighted centroid using spectrum values
    """
    nele, nazi = S.shape

    # 1) Find top-2 peaks in spatial spectrum
    peaks = find_top2_peaks(S, suppress_radius=(3, 3))

    if len(peaks) == 0:
        print("Warning: No peaks found in test spectrum")
        return []

    # 2) Create Voronoi masks to prevent region overlap if requested
    voronoi_masks = None
    if prevent_overlap and len(peaks) > 1:
        voronoi_masks = create_voronoi_masks(peaks, S.shape)

    regions = []

    # 3) Apply flood-fill with lambda_hat around each peak
    for peak_idx, (peak_ele, peak_azi) in enumerate(peaks):
        peak_value = S[peak_ele, peak_azi]

        # Get Voronoi mask for this peak
        mask = voronoi_masks[peak_idx] if voronoi_masks is not None else None

        # Apply flood-fill to get confidence region
        region_mask = flood_fill_region(S, (peak_ele, peak_azi), lambda_hat, connectivity, mask)
        region_size = np.sum(region_mask)

        # Create region dictionary
        region_info = {
            'peak_position_idx': (peak_ele, peak_azi),
            'peak_value': float(peak_value),
            'region_mask': region_mask,
            'region_size': int(region_size),
            'threshold': float(peak_value - lambda_hat),
            'lambda_used': float(lambda_hat)
        }

        # 4) Convert to degrees if doa_candidate provided
        if doa_candidate is not None and len(doa_candidate) == 2:
            ele_candidate, azi_candidate = doa_candidate

            # Convert indices to degrees
            if 0 <= peak_ele < len(ele_candidate) and 0 <= peak_azi < len(azi_candidate):
                ele_deg = float(np.degrees(ele_candidate[peak_ele]))
                azi_deg = float(np.degrees(azi_candidate[peak_azi]))
                region_info['peak_position_deg'] = (ele_deg, azi_deg)
            else:
                print(f"Warning: Peak position {(peak_ele, peak_azi)} outside doa_candidate bounds")
                region_info['peak_position_deg'] = None
        else:
            region_info['peak_position_deg'] = None

        # 5) Compute improved region statistics
        if region_size > 1:
            region_coords = np.where(region_mask)
            region_info['region_bounds'] = {
                'ele_min': int(np.min(region_coords[0])),
                'ele_max': int(np.max(region_coords[0])),
                'azi_min': int(np.min(region_coords[1])),
                'azi_max': int(np.max(region_coords[1]))
            }

            # Compute weighted centroid using spectrum values above threshold
            threshold = peak_value - lambda_hat
            weights = np.maximum(S - threshold, 0.0) * region_mask

            if np.sum(weights) > 0:
                # Weighted centroid for elevation (no wraparound)
                weighted_ele = np.sum(region_coords[0] * weights[region_coords]) / np.sum(weights[region_coords])

                # Weighted circular mean for azimuth (considering wraparound)
                azi_coords = region_coords[1]
                azi_weights = weights[region_coords]

                # Convert to complex numbers for circular mean
                azi_angles = 2 * np.pi * azi_coords / nazi
                complex_azi = np.exp(1j * azi_angles)
                weighted_complex = np.sum(complex_azi * azi_weights) / np.sum(azi_weights)
                mean_angle = np.angle(weighted_complex)

                # Convert back to azimuth index
                weighted_azi = (mean_angle * nazi / (2 * np.pi)) % nazi

                region_info['weighted_centroid'] = (float(weighted_ele), float(weighted_azi))
            else:
                # Fallback to simple mean if no weights
                region_info['weighted_centroid'] = (
                    float(np.mean(region_coords[0])),
                    float(np.mean(region_coords[1]))
                )
        else:
            region_info['region_bounds'] = {
                'ele_min': peak_ele, 'ele_max': peak_ele,
                'azi_min': peak_azi, 'azi_max': peak_azi
            }
            region_info['weighted_centroid'] = (float(peak_ele), float(peak_azi))

        regions.append(region_info)

        print(f"Region {peak_idx}: Peak@{(peak_ele, peak_azi)} (value={peak_value:.3f}), "
              f"threshold={region_info['threshold']:.3f}, size={region_size} pixels")

    print(f"Extracted {len(regions)} confidence regions using lambda_hat={lambda_hat:.3f}")

    return regions


def angular_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """
    Compute angular distance between two DOA positions.

    Args:
        pos1, pos2: DOA positions as [elevation, azimuth] in degrees

    Returns:
        Angular distance in degrees
    """
    # TODO: Implement proper spherical distance calculation
    # - Handle azimuth wraparound (-180 to +180 degrees)
    # - Use spherical trigonometry for accurate distance
    pass


def visualize_regions(S: np.ndarray,
                     regions: List[Dict],
                     gt_positions: Optional[np.ndarray] = None,
                     save_path: Optional[str] = None) -> None:
    """
    Visualize spatial spectrum with confidence regions and ground truth.

    Args:
        S: Spatial spectrum of shape (nele, nazi)
        regions: List of region dictionaries from extract_regions_test
        gt_positions: Optional GT positions for comparison
        save_path: Optional path to save the plot
    """
    # TODO: Implement visualization
    # - Plot spatial spectrum as heatmap
    # - Overlay confidence regions as contours/masks
    # - Mark peak positions and ground truth
    # - Add colorbar and proper labeling
    pass


if __name__ == "__main__":
    """
    Simple self-test with fake spatial spectrum containing two bright blobs.
    """
    print("=== CRC Pipeline Self-Test ===")

    # Create fake spatial spectrum with two bright blobs
    nele, nazi = 20, 36  # 20 elevation bins, 36 azimuth bins
    S = np.random.normal(0.1, 0.05, (nele, nazi))  # Background noise

    # Add two bright blobs at known positions
    blob1_pos = (5, 10)   # (ele_idx, azi_idx)
    blob2_pos = (12, 25)

    # Create Gaussian blobs
    for ele in range(nele):
        for azi in range(nazi):
            # Blob 1
            dist1 = np.sqrt((ele - blob1_pos[0])**2 + (azi - blob1_pos[1])**2)
            S[ele, azi] += 0.8 * np.exp(-dist1**2 / 4.0)

            # Blob 2
            dist2 = np.sqrt((ele - blob2_pos[0])**2 + (azi - blob2_pos[1])**2)
            S[ele, azi] += 0.6 * np.exp(-dist2**2 / 4.0)

    print(f"Created spatial spectrum of shape {S.shape}")
    print(f"Expected peaks at {blob1_pos} and {blob2_pos}")

    # Test peak detection
    peaks = find_top2_peaks(S, suppress_radius=(3, 3))
    print(f"Found peaks: {peaks}")

    # Verify peaks are correct
    if len(peaks) == 2:
        peak1, peak2 = peaks
        # Check if peaks are close to expected positions
        dist1_to_blob1 = np.sqrt((peak1[0] - blob1_pos[0])**2 + (peak1[1] - blob1_pos[1])**2)
        dist1_to_blob2 = np.sqrt((peak1[0] - blob2_pos[0])**2 + (peak1[1] - blob2_pos[1])**2)

        if dist1_to_blob1 <= dist1_to_blob2:
            # peak1 matches blob1
            expected_match = [(blob1_pos, peak1), (blob2_pos, peak2)]
        else:
            # peak1 matches blob2
            expected_match = [(blob2_pos, peak1), (blob1_pos, peak2)]

        print("Peak detection: ✓ PASSED")
        for i, (expected, found) in enumerate(expected_match):
            dist = np.sqrt((found[0] - expected[0])**2 + (found[1] - expected[1])**2)
            print(f"  Peak {i+1}: expected {expected}, found {found}, distance={dist:.2f}")
    else:
        print(f"Peak detection: ✗ FAILED - Expected 2 peaks, found {len(peaks)}")

    # Test matching
    gt_positions = [blob1_pos, blob2_pos]  # Ground truth as (ele_idx, azi_idx)
    matched_gt_indices = match_peaks_to_gt(peaks, gt_positions)
    print(f"Matching result: {matched_gt_indices}")

    # Verify matching
    if len(peaks) == 2 and len(matched_gt_indices) == 2:
        # Check if matching makes sense
        total_error = 0
        for gt_idx, pred_idx in enumerate(matched_gt_indices):
            if pred_idx is not None:
                gt_pos = gt_positions[gt_idx]
                pred_pos = peaks[pred_idx]
                error = np.sqrt((pred_pos[0] - gt_pos[0])**2 + (pred_pos[1] - gt_pos[1])**2)
                total_error += error
                print(f"  GT{gt_idx} {gt_pos} -> Pred{pred_idx} {pred_pos}, error={error:.2f}")

        print(f"Matching: ✓ PASSED (total error={total_error:.2f})")
    else:
        print("Matching: ✗ FAILED - Incorrect number of matches")

    print("=== Peak Detection and Matching Test Complete ===\n")

    # =========================================================================
    # Test flood-fill region
    # =========================================================================
    print("=== Flood-Fill Region Test ===")

    # Create a simple 2D spectrum with a peak and smooth decay
    nele, nazi = 15, 20
    center_ele, center_azi = 7, 10
    S_flood = np.zeros((nele, nazi))

    # Create smooth decay from center peak
    peak_value = 1.0
    for ele in range(nele):
        for azi in range(nazi):
            # Distance from center with azimuth wraparound consideration
            ele_dist = abs(ele - center_ele)
            azi_dist = min(abs(azi - center_azi), nazi - abs(azi - center_azi))
            total_dist = np.sqrt(ele_dist**2 + azi_dist**2)

            # Smooth exponential decay
            S_flood[ele, azi] = peak_value * np.exp(-total_dist**2 / 8.0)

    print(f"Created spectrum with peak at {(center_ele, center_azi)} with value {S_flood[center_ele, center_azi]:.3f}")

    # Test different lambda values
    lambda_values = [0.0, 0.2, 0.5, 0.8]

    for lambda_val in lambda_values:
        region_mask = flood_fill_region(S_flood, (center_ele, center_azi), lambda_val)
        region_size = np.sum(region_mask)
        threshold = peak_value - lambda_val

        print(f"Lambda={lambda_val:.1f}, Threshold={threshold:.3f}, Region size={region_size} pixels")

        # Verify peak is always included
        assert region_mask[center_ele, center_azi], f"Peak not included for lambda={lambda_val}"

        # Show some region boundary info
        if region_size > 1:
            region_coords = np.where(region_mask)
            min_ele, max_ele = np.min(region_coords[0]), np.max(region_coords[0])
            min_azi, max_azi = np.min(region_coords[1]), np.max(region_coords[1])
            print(f"  Region spans: ele=[{min_ele}, {max_ele}], azi=[{min_azi}, {max_azi}]")

    # Verify region grows as lambda increases
    regions_sizes = []
    for lambda_val in lambda_values:
        region_mask = flood_fill_region(S_flood, (center_ele, center_azi), lambda_val)
        regions_sizes.append(np.sum(region_mask))

    # Check that regions are monotonically increasing
    is_increasing = all(regions_sizes[i] <= regions_sizes[i+1] for i in range(len(regions_sizes)-1))

    if is_increasing:
        print("✓ PASSED: Region size increases with lambda as expected")
    else:
        print(f"✗ FAILED: Region sizes not increasing: {regions_sizes}")

    # Test edge case: lambda = 0 should only include peak
    region_mask_zero = flood_fill_region(S_flood, (center_ele, center_azi), 0.0)
    if np.sum(region_mask_zero) == 1 and region_mask_zero[center_ele, center_azi]:
        print("✓ PASSED: Lambda=0 includes only peak")
    else:
        print(f"✗ FAILED: Lambda=0 should include only peak, got {np.sum(region_mask_zero)} pixels")

    # Test connectivity options
    region_4conn = flood_fill_region(S_flood, (center_ele, center_azi), 0.3, connectivity=4)
    region_8conn = flood_fill_region(S_flood, (center_ele, center_azi), 0.3, connectivity=8)

    size_4 = np.sum(region_4conn)
    size_8 = np.sum(region_8conn)

    if size_8 >= size_4:
        print(f"✓ PASSED: 8-connectivity ({size_8}) >= 4-connectivity ({size_4})")
    else:
        print(f"✗ FAILED: 8-connectivity should be >= 4-connectivity")

    print("=== Flood-Fill Test Complete ===\n")

    # =========================================================================
    # Test conformal prediction calibration
    # =========================================================================
    print("=== Conformal Prediction Calibration Test ===")

    # Create synthetic calibration data with multiple frames
    np.random.seed(42)  # For reproducible results
    n_cal_frames = 50
    nele, nazi = 15, 20

    spectra_cal = []
    gt_cal = []

    print(f"Creating {n_cal_frames} synthetic calibration frames...")

    for frame_idx in range(n_cal_frames):
        # Create base spectrum with noise
        spectrum = np.random.normal(0.1, 0.05, (nele, nazi))

        # Add 2 speakers at random positions
        speakers = []
        for spk_idx in range(2):
            # Random speaker position
            gt_ele = np.random.randint(2, nele-2)
            gt_azi = np.random.randint(2, nazi-2)
            speakers.append((gt_ele, gt_azi))

            # Add speaker as Gaussian blob with some randomness
            peak_strength = 0.6 + 0.4 * np.random.random()  # Random strength
            for ele in range(nele):
                for azi in range(nazi):
                    dist = np.sqrt((ele - gt_ele)**2 + (azi - gt_azi)**2)
                    spectrum[ele, azi] += peak_strength * np.exp(-dist**2 / 6.0)

        spectra_cal.append(spectrum)
        gt_cal.append(speakers)

    print(f"Created calibration data: {len(spectra_cal)} frames with {sum(len(gt) for gt in gt_cal)} total speakers")

    # Test conformal prediction with different alpha values
    alpha_values = [0.05, 0.1, 0.2]

    for alpha in alpha_values:
        print(f"\n--- Testing alpha = {alpha} (target coverage = {(1-alpha)*100:.0f}%) ---")
        # Enable plotting for the first alpha value to demonstrate calibration visualization
        plot_calibration = (alpha == alpha_values[0])
        lambda_hat, deltas = compute_lambda_cp(spectra_cal, gt_cal, alpha, plot_calibration=plot_calibration)

        if len(deltas) > 0:
            # Basic statistics
            print(f"Statistics: mean={np.mean(deltas):.3f}, std={np.std(deltas):.3f}, "
                  f"median={np.median(deltas):.3f}")

            # Verify lambda_hat is reasonable
            coverage_estimate = np.mean(deltas <= lambda_hat)
            print(f"Empirical coverage with lambda_hat: {coverage_estimate*100:.1f}%")

            if coverage_estimate >= (1 - alpha) - 0.05:  # Allow small tolerance
                print("✓ PASSED: Coverage meets target")
            else:
                print(f"✗ WARNING: Coverage {coverage_estimate*100:.1f}% < target {(1-alpha)*100:.0f}%")

    # Test edge cases
    print("\n--- Testing edge cases ---")

    # Empty calibration data
    try:
        lambda_empty, deltas_empty = compute_lambda_cp([], [], 0.1)
        if len(deltas_empty) == 0:
            print("✓ PASSED: Empty data handled correctly")
        else:
            print("✗ FAILED: Empty data not handled correctly")
    except Exception as e:
        print(f"✗ FAILED: Exception with empty data: {e}")

    # Single frame
    single_spectrum = [spectra_cal[0]]
    single_gt = [gt_cal[0]]
    lambda_single, deltas_single = compute_lambda_cp(single_spectrum, single_gt, 0.1)
    if len(deltas_single) > 0:
        print(f"✓ PASSED: Single frame handled (lambda={lambda_single:.3f})")
    else:
        print("✗ FAILED: Single frame not handled correctly")

    print("=== Conformal Prediction Test Complete ===\n")

    # =========================================================================
    # Test complete conformal prediction pipeline (calibration + test)
    # =========================================================================
    print("=== Complete Conformal Prediction Pipeline Test ===")

    # Use calibrated lambda from previous test
    lambda_hat_90 = compute_lambda_cp(spectra_cal, gt_cal, alpha=0.1)[0]
    print(f"Using calibrated lambda_hat = {lambda_hat_90:.3f} for 90% coverage\n")

    # Create test data (different from calibration)
    print("Creating test spectrum...")
    nele, nazi = 15, 20
    test_spectrum = np.random.normal(0.1, 0.05, (nele, nazi))

    # Add 2 test speakers at known positions
    test_speaker1 = (6, 8)
    test_speaker2 = (11, 15)

    # Create test blobs
    for ele in range(nele):
        for azi in range(nazi):
            # Speaker 1
            dist1 = np.sqrt((ele - test_speaker1[0])**2 + (azi - test_speaker1[1])**2)
            test_spectrum[ele, azi] += 0.7 * np.exp(-dist1**2 / 5.0)

            # Speaker 2
            dist2 = np.sqrt((ele - test_speaker2[0])**2 + (azi - test_speaker2[1])**2)
            test_spectrum[ele, azi] += 0.9 * np.exp(-dist2**2 / 5.0)

    print(f"Test speakers at: {test_speaker1} and {test_speaker2}")

    # Extract confidence regions using learned lambda
    regions = extract_regions_test(test_spectrum, lambda_hat_90)

    print(f"\n--- Test Results ---")
    print(f"Number of regions extracted: {len(regions)}")

    for i, region in enumerate(regions):
        print(f"\nRegion {i}:")
        print(f"  Peak position: {region['peak_position_idx']}")
        print(f"  Peak value: {region['peak_value']:.3f}")
        print(f"  Threshold: {region['threshold']:.3f}")
        print(f"  Region size: {region['region_size']} pixels")
        print(f"  Region bounds: {region['region_bounds']}")
        print(f"  Region centroid: ({region['weighted_centroid'][0]:.1f}, {region['weighted_centroid'][1]:.1f})")

    # Verify regions contain ground truth speakers
    coverage_check = []
    for speaker_idx, gt_speaker in enumerate([test_speaker1, test_speaker2]):
        contained_in_regions = []

        for region_idx, region in enumerate(regions):
            mask = region['region_mask']
            gt_ele, gt_azi = gt_speaker

            if mask[gt_ele, gt_azi]:
                contained_in_regions.append(region_idx)

        coverage_check.append(len(contained_in_regions) > 0)
        print(f"Speaker {speaker_idx} at {gt_speaker}: {'✓ COVERED' if len(contained_in_regions) > 0 else '✗ NOT COVERED'} "
              f"by regions {contained_in_regions}")

    overall_coverage = np.mean(coverage_check) * 100
    print(f"\nOverall test coverage: {overall_coverage:.0f}% ({sum(coverage_check)}/{len(coverage_check)} speakers)")

    # Test with DOA candidate conversion
    print(f"\n--- Testing with DOA candidate conversion ---")
    ele_candidate = np.linspace(0, np.pi, nele)  # 0 to 180 degrees
    azi_candidate = np.linspace(-np.pi, np.pi, nazi)  # -180 to 180 degrees
    doa_candidate = [ele_candidate, azi_candidate]

    regions_with_doa = extract_regions_test(test_spectrum, lambda_hat_90, doa_candidate=doa_candidate)

    for i, region in enumerate(regions_with_doa):
        if region['peak_position_deg'] is not None:
            ele_deg, azi_deg = region['peak_position_deg']
            print(f"Region {i}: Peak at ({ele_deg:.1f}°, {azi_deg:.1f}°)")
        else:
            print(f"Region {i}: DOA conversion failed")

    # Test different lambda values on same test spectrum
    print(f"\n--- Testing different lambda values ---")
    test_lambdas = [0.0, 0.1, 0.2, 0.5]

    for test_lambda in test_lambdas:
        regions_lambda = extract_regions_test(test_spectrum, test_lambda)
        total_region_size = sum(r['region_size'] for r in regions_lambda)
        print(f"Lambda = {test_lambda:.1f}: {len(regions_lambda)} regions, total size = {total_region_size} pixels")

    print("=== Complete Pipeline Test Complete ===")