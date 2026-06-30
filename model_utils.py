import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from scipy import ndimage
from skimage.measure import label


def crop_pad_image_only(image, target_shape=(256, 256)):
    if (image.shape[0] < image.shape[1]) and (image.shape[0] < image.shape[2]):
        image = np.transpose(image, (1, 2, 0))
    orig_shape = image.shape
    H, W, D = orig_shape
    tH, tW = target_shape
    if H >= tH:
        start = (H - tH) // 2
        image = image[start:start + tH, :, :]
    else:
        pad = tH - H
        image = np.pad(image, ((pad // 2, pad - pad // 2), (0, 0), (0, 0)))
    if W >= tW:
        start = (W - tW) // 2
        image = image[:, start:start + tW, :]
    else:
        pad = tW - W
        image = np.pad(image, ((0, 0), (pad // 2, pad - pad // 2), (0, 0)))
    meta = {"orig_shape": orig_shape}
    return image, meta


def reverse_crop_pad(processed, meta):
    orig_y, orig_x, orig_z = meta["orig_shape"]
    py, px, pz = processed.shape
    reconstructed = np.zeros((orig_y, orig_x, orig_z), dtype=processed.dtype)
    start_y_proc = max((py - orig_y) // 2, 0)
    start_x_proc = max((px - orig_x) // 2, 0)
    start_y_orig = max((orig_y - py) // 2, 0)
    start_x_orig = max((orig_x - px) // 2, 0)
    copy_y = min(py, orig_y)
    copy_x = min(px, orig_x)
    reconstructed[
        start_y_orig:start_y_orig + copy_y,
        start_x_orig:start_x_orig + copy_x,
        :
    ] = processed[
        start_y_proc:start_y_proc + copy_y,
        start_x_proc:start_x_proc + copy_x,
        :
    ]
    return reconstructed


def z_normalise_image(image):
    mean = np.mean(image)
    std = np.std(image)
    image -= mean
    image /= (max(std, 1e-8))
    return image


TARGET_SPACING = 1.3671900033950806 # Median X/Y Pixdims from ACDC/MMS training set


def resample_volume(image, native_spacing, target_spacing=TARGET_SPACING):
    """Resample H and W to target spacing; leave D unchanged. Uses bilinear (order=1)."""
    zoom_factor = native_spacing / target_spacing
    if abs(zoom_factor - 1.0) < 1e-3:
        return image.copy()
    return zoom(image, (zoom_factor, zoom_factor, 1.0), order=1)


def resample_mask(mask, native_spacing, target_spacing=TARGET_SPACING):
    """Resample integer label mask back to native spacing. Uses nearest-neighbour (order=0)."""
    zoom_factor = target_spacing / native_spacing
    if abs(zoom_factor - 1.0) < 1e-3:
        return mask
    return zoom(mask.astype(np.float32), (zoom_factor, zoom_factor, 1.0), order=0).astype(np.uint8)

def resample_prior_mask(mask, native_spacing, target_spacing=TARGET_SPACING):
    """Resamples integer label mask to target spacing. Uses nearest-neighbour (order=0)."""
    zoom_factor =  native_spacing / target_spacing
    if abs(zoom_factor - 1.0) < 1e-3:
        return mask
    return zoom(mask.astype(np.float32), (zoom_factor, zoom_factor, 1.0), order=0).astype(np.uint8)


def _stable_softmax(x, axis=-1):
    x = np.asarray(x)
    shifted_x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted_x)
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / sum_exp_x


def _make_gaussian_importance_map(patch_size, sigma_scale=1/8, value_scaling_factor=1, dtype=np.float32):
    patch_size = tuple(int(p) for p in patch_size)
    tmp = np.zeros(patch_size, dtype=np.float32)
    center_coords = tuple([p // 2 for p in patch_size])
    tmp[center_coords] = 1.0
    sigmas = [p * float(sigma_scale) for p in patch_size]
    g = gaussian_filter(tmp, sigma=sigmas, order=0, mode='constant', cval=0.0)
    maxv = g.max()
    if maxv > 0:
        g = g / (maxv / float(value_scaling_factor))
    zero_mask = (g == 0)
    if np.any(zero_mask):
        nonzero_vals = g[~zero_mask]
        if nonzero_vals.size == 0:
            g[...] = float(value_scaling_factor)
        else:
            g[zero_mask] = np.min(nonzero_vals)
    return g.astype(dtype)


def _compute_steps(im_size, patch_size, overlap):
    ps = np.array(patch_size, dtype=int)
    sz = np.array(im_size, dtype=int)
    stride = np.maximum((ps * (1.0 - overlap)).astype(int), 1)
    steps = []
    for i in range(len(sz)):
        s = []
        pos = 0
        while True:
            s.append(pos)
            if pos + ps[i] >= sz[i]:
                break
            pos += stride[i]
            if pos + ps[i] > sz[i]:
                pos = sz[i] - ps[i]
        steps.append(s)
    return steps


def _pad_to_patch_size(vol, patch_size):
    vol = np.asarray(vol)
    spatial = vol.shape[:3]
    pads = []
    for d in range(3):
        need = max(0, patch_size[d] - spatial[d])
        pads.append((need // 2, need - need // 2))
    pad_width = (pads[0], pads[1], pads[2], (0, 0))
    vol_padded = np.pad(vol, pad_width, mode='reflect')
    return vol_padded, pads


def _crop_from_pad(vol, pads):
    y0, y1 = pads[0]
    x0, x1 = pads[1]
    z0, z1 = pads[2]
    Y, X, Z = vol.shape[:3]
    return vol[y0:Y - y1 if y1 > 0 else Y,
               x0:X - x1 if x1 > 0 else X,
               z0:Z - z1 if z1 > 0 else Z, ...]


def _flip3d(vol, axes_mask):
    vol_f = vol
    for ax, do in enumerate(axes_mask):
        if do:
            vol_f = np.flip(vol_f, axis=ax)
    return vol_f


def _iter_tta_axes(do_tta):
    if not do_tta:
        yield (False, False, False)
        return
    for a in (False, True):
        for b in (False, True):
            for c in (False, True):
                yield (a, b, c)


def get_one_hot(indices, num_classes):
    one_hot_array = np.eye(num_classes)[indices]
    return one_hot_array.astype(np.uint8)


def getLargestCC(segmentation):
    labels = label(segmentation)
    assert labels.max() != 0
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC


def postprocess(mask):
    if mask.ndim == 5:
        labels_int = np.argmax(mask, axis=-1)
    else:
        labels_int = mask.copy()
    return labels_int.astype(np.uint8)   # component logic now in enforce_single_components

def enforce_single_components(labels, n_labels, fg_labels=None, max_iter=3):
    """
    labels: (H, W, D) int label volume (0=bg, 1..K structures).
    For each structure, keep its largest connected component. Each smaller
    island is RELABELLED to the adjacent structure it shares the most border
    with (reabsorbed), preserving the voxels. Islands touching no other
    structure are set to background.
    Returns (H, W, D) uint8.
    """
    from scipy import ndimage
    labels = labels.copy()
    if fg_labels is None:
        fg_labels = list(range(1, n_labels))
    struct = ndimage.generate_binary_structure(3, 1)   # 6-connectivity

    for _ in range(max_iter):
        changed = False
        for L in fg_labels:
            mask_L = labels == L
            if not mask_L.any():
                continue
            cc, ncc = ndimage.label(mask_L, structure=struct)
            if ncc <= 1:
                continue
            sizes = ndimage.sum(np.ones_like(cc), cc, index=range(1, ncc + 1))
            largest = int(np.argmax(sizes)) + 1
            for c in range(1, ncc + 1):
                if c == largest:
                    continue
                island = cc == c
                dil = ndimage.binary_dilation(island, structure=struct)
                border = dil & ~island
                neigh = labels[border]
                neigh = neigh[(neigh != 0) & (neigh != L)]   # other structures only
                if neigh.size > 0:
                    vals, counts = np.unique(neigh, return_counts=True)
                    target = int(vals[np.argmax(counts)])     # most-shared neighbour
                    labels[island] = target
                else:
                    labels[island] = 0                        # isolated -> remove
                changed = True
        if not changed:
            break
    return labels.astype(np.uint8)
