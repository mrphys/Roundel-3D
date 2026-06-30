from dicom_utils import *
from model_utils import *

import os
import glob
import math
import hashlib
import shutil
from pathlib import Path
import io

import nibabel as nib
import numpy as np
from PIL import Image, ImageSequence, ImageDraw, ImageFont
from cv2 import resize, INTER_NEAREST
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from skimage.measure import label as cc_label, regionprops
from scipy.ndimage import (
    binary_fill_holes,
    binary_dilation,
    binary_erosion,
    binary_closing,
    gaussian_filter
) 
from skimage.morphology import disk,convex_hull_image
import pandas as pd
from skimage.measure import find_contours
import cv2
import json
import pydicom
import zipfile
import json
from stqdm import stqdm
import sys

# --- Paths: separate read-only bundled assets from writable per-user data ---
if getattr(sys, 'frozen', False):
    bundle_path = Path(sys._MEIPASS)            # PyInstaller one-file extraction dir
    install_path = Path(sys.executable).parent
else:
    bundle_path = Path(__file__).resolve().parent
    install_path = bundle_path

# Read-only assets shipped with the app (model weights, etc.)
models_path = str(bundle_path / "models")

# Writable per-user data (NEVER next to the exe / inside the bundle)
app_data_root = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Roundel"
data_path    = str(app_data_root / "data")
results_path = str(app_data_root / "results")
cache_dir    = str(app_data_root / "cache") + os.sep
final_dir    = f'{results_path}/results.zip'

# Derived result paths
blank_gif_path        = f'{results_path}/temp/blank'
full_edited_gif_path  = f'{results_path}/temp/edited'
preprocessed_gif_path = f'{results_path}/temp/preprocessed'
edv_esv_gif_path      = f'{results_path}/temp/edv_esv'
edited_gif_path       = f'{results_path}/temp/edited_edv_esv'
raw_curve_path        = f'{results_path}/temp/raw_metrics.png'
edited_curve_path     = f'{results_path}/temp/edited_metrics.png'
dicom_mask_path       = f'{results_path}/masks/dicoms/'
nifti_mask_path       = f'{results_path}/masks/nifti/'

# Create writable dirs once
for _p in [data_path,
           f'{results_path}/temp',
           f'{results_path}/gifs',
           f'{results_path}/edited_sax_df',
           cache_dir]:
    os.makedirs(_p, exist_ok=True)

GIF_W = 150
DISPLAY_W = 400

labels = {
    "background": 0,
    "LV": 1,
    "RV": 2,
    "LA": 3,
    "RA": 4,
    "Ao": 5,
    "PA": 6,
}

# Per-structure index constants (used across roundel_utils, get_overlay, etc.)
bg_idx = labels['background']
lv_idx = labels['LV']
rv_idx = labels['RV']
la_idx = labels['LA']
ra_idx = labels['RA']
ao_idx = labels['Ao']
pa_idx = labels['PA']

# Convenience groupings
chamber_indices       = [lv_idx, rv_idx, la_idx, ra_idx]
great_vessel_indices  = [ao_idx, pa_idx]
foreground_indices    = chamber_indices + great_vessel_indices

# Colours: (R, G, B, A on 0–255), one per structure
LV_COLOR = (200, 30,  30,  100)   # red
RV_COLOR = (30,  90,  220, 100)   # blue
LA_COLOR = (240, 140, 60,  100)   # orange
RA_COLOR = (140, 80,  200, 100)   # purple
AO_COLOR = (220, 200, 40,  100)   # yellow
PA_COLOR = (30,  180, 180, 100)   # teal

# Lookup by label index — used by get_overlay when ventricle='all'
BG_COLOR = (0, 0, 0, 0)   # transparent / black, used as the "erase" colour

OVERLAY_COLORS_BY_IDX = {
    bg_idx: BG_COLOR,
    lv_idx: LV_COLOR,
    rv_idx: RV_COLOR,
    la_idx: LA_COLOR,
    ra_idx: RA_COLOR,
    ao_idx: AO_COLOR,
    pa_idx: PA_COLOR,
}

BRUSH_LABELS = {
    lv_idx: 'LV 🔴',
    rv_idx: 'RV 🔵',
    la_idx: 'LA 🟠',
    ra_idx: 'RA 🟣',
    ao_idx: 'Ao 🟡',
    pa_idx: 'PA 🟢',
}

STRUCTURE_GROUPS = {
    'chambers':      [lv_idx, rv_idx, la_idx, ra_idx],
    'great_vessels': [ao_idx, pa_idx],
}


def load_font(size):
    # Try Linux font
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except:
        pass
    # Try Windows font
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
    except:
        pass
    # Fallback (non scalable)
    return ImageFont.load_default()


def save_cached_mask(mask, save_path):
    np.save(save_path, mask)

def load_cached_mask(save_path):
    return np.load(save_path)

def save_config(config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(config, f, indent=2)

def load_config(path) :
    path = Path(path)
    with path.open("r") as f:
        return json.load(f)

def save_mask(mask, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    nib_mask = nib.Nifti1Image(mask, affine=np.eye(4), dtype='uint8')
    nib.save(nib_mask, save_path)

def save_image(image, save_path):
    nib_image = nib.Nifti1Image(image, affine=np.eye(4), dtype='float32')
    nib.save(nib_image, save_path)

def normalize(image):
    image = (image - np.min(image))/(np.max(image) - np.min(image))
    return image


def cv_zoom(images, zoom, interpolation=cv2.INTER_CUBIC):
    """
    Resize height and width of a 4D or 5D array using OpenCV. Only H and W are scaled.

    Args:
        images (numpy.ndarray): Array of shape (H, W, D, T) or (H, W, D, T, C)
        zoom_factors (list or tuple): Zoom factors for (H, W, D, T, C). Only H and W > 1
        interpolation (int): OpenCV interpolation method (default: cv2.INTER_CUBIC)

    Returns:
        numpy.ndarray: Resized array with height and width scaled, other dimensions unchanged
    """
    h_zoom, w_zoom = zoom[0], zoom[1]

    if images.ndim == 4:
        h, w, d, t = images.shape
        resized = np.zeros((int(h*h_zoom), int(w*w_zoom), d, t), dtype=images.dtype)
        for z in range(d):
            for tau in range(t):
                resized[..., z, tau] = cv2.resize(images[..., z, tau], (int(w*w_zoom), int(h*h_zoom)), interpolation=interpolation)
    elif images.ndim == 5:
        h, w, d, t, c = images.shape
        resized = np.zeros((int(h*h_zoom), int(w*w_zoom), d, t, c), dtype=images.dtype)
        for z in range(d):
            for tau in range(t):
                for ch in range(c):
                    resized[..., z, tau, ch] = cv2.resize(images[..., z, tau, ch], (int(w*w_zoom), int(h*h_zoom)), interpolation=interpolation)
    else:
        raise ValueError("Input must be 4D or 5D array.")

    return resized


def load_nii(nii_path):
    file = nib.load(nii_path)
    data = file.get_fdata(caching='unchanged')
    return data

def cv_zoom_mask(mask, zoom, sigma=2.0, interpolation=cv2.INTER_NEAREST):
    """mask: (H, W, D, T, C) one-hot. Resizes H, W. Returns one-hot."""
    zoomed = cv_zoom(mask.astype(np.float32), zoom, interpolation=interpolation)
    # zoomed is already (near-)one-hot from nearest-neighbour resize.
    # Re-binarise via argmax to guarantee clean one-hot, no smoothing.
    C = zoomed.shape[-1]
    labels_int = np.argmax(zoomed, axis=-1).astype(np.uint8)
    return np.eye(C, dtype=np.uint8)[labels_int]


def cv_zoom_ax12(images, zoom, interpolation=cv2.INTER_CUBIC):
    """
    Resize axes 1 and 2 (leave axis 0 and the trailing axes untouched).
    images: (A0, A1, A2, T) or (A0, A1, A2, T, C)
    zoom: scale factors; only zoom[1], zoom[2] are used.
    """
    z1, z2 = zoom[1], zoom[2]
    if images.ndim == 4:
        A0, A1, A2, T = images.shape
        out = np.zeros((A0, int(A1*z1), int(A2*z2), T), dtype=images.dtype)
        for a0 in range(A0):
            for t in range(T):
                out[a0, :, :, t] = cv2.resize(
                    images[a0, :, :, t],
                    (int(A2*z2), int(A1*z1)),   # cv2 = (width=A2, height=A1)
                    interpolation=interpolation
                )
    elif images.ndim == 5:
        A0, A1, A2, T, C = images.shape
        out = np.zeros((A0, int(A1*z1), int(A2*z2), T, C), dtype=images.dtype)
        for a0 in range(A0):
            for t in range(T):
                for c in range(C):
                    out[a0, :, :, t, c] = cv2.resize(
                        images[a0, :, :, t, c],
                        (int(A2*z2), int(A1*z1)),
                        interpolation=interpolation
                    )
    else:
        raise ValueError("Input must be 4D or 5D.")
    return out

def cv_zoom_mask_ax12(mask, zoom, interpolation=cv2.INTER_NEAREST):
    """mask: (A0, A1, A2, T, C) one-hot. Resizes axes 1,2. Returns one-hot."""
    zoomed = cv_zoom_ax12(mask.astype(np.float32), zoom, interpolation=interpolation)
    C = zoomed.shape[-1]
    labels_int = np.argmax(zoomed, axis=-1).astype(np.uint8)
    return np.eye(C, dtype=np.uint8)[labels_int]

def format_delta(value, raw_value, suffix="", round_digits=None):
    if round_digits is not None:
        value = round(value, round_digits)
        raw_value = round(raw_value, round_digits)
    return None if value == raw_value else f"{value - raw_value:.1f}{suffix}"


def find_crop_box(mask, crop_factor):
    '''
    Calculated a bounding box that contains the masks inside.

    Parameters:
    mask: np.array
        A binary mask array, which should be the flattened 3D multislice mask, where the pixels in the z-dimension are summed
    crop_factor: float
        A scaling factor for the bounding box
    Returns:
    list
        A list containing the coordinates of the bounding box [x_min, y_min, x_max, y_max]. These co-ordinates can be used to crop each slice of the input multislice image.
    '''
    # Check shape of the input is 2D
    if len(mask.shape) != 2:
        raise ValueError("Input mask must be a 2D array")

    if np.max(mask) == 0:
        x_min, x_max = 0, mask.shape[0]
        y_min, y_max = 0, mask.shape[1]
        return [x_min, y_min, x_max, y_max]

    else:
        y = np.sum(mask, axis=1) # sum the masks across columns of array, returns a 1D array of row totals
        x = np.sum(mask, axis=0) # sum the masks across rows of array, returns a 1D array of column totals

        top = np.min(np.nonzero(y)) - 1 # Returns the indices of the elements in 1d row totals array that are non-zero, then finds the minimum value and subtracts 1 (i.e. top extent of mask)
        bottom = np.max(np.nonzero(y)) + 1 # Returns the indices of the elements in 1d row totals array that are non-zero, then finds the maximum value and adds 1 (i.e. bottom extent of mask)

        left = np.min(np.nonzero(x)) - 1 # Returns the indices of the elements in 1d column totals array that are non-zero, then finds the minimum value and subtracts 1 (i.e. left extent of mask)
        right = np.max(np.nonzero(x)) + 1 # Returns the indices of the elements in 1d column totals array that are non-zero, then finds the maximum value and adds 1 (i.e. right extent of mask)
        if abs(right - left) > abs(top - bottom):
            largest_side = abs(right - left) # Find the largest side of the bounding box
        else:
            largest_side = abs(top - bottom)

        
        x_mid = round((left + right) / 2) # Find the mid-point of the x-length of mask
        y_mid = round((top + bottom) / 2) # Find the mid-point of the y-length of mask
        half_largest_side = round(largest_side * crop_factor / 2) # Find half the largest side of the bounding box (crop factor scales the largest side to ensure whole heart and some surrounding is captured)
        x_max, x_min = round(x_mid + half_largest_side), round(x_mid - half_largest_side) # Find the maximum and minimum x-values of the bounding box
        y_max, y_min = round(y_mid + half_largest_side), round(y_mid - half_largest_side) # Find the maximum and minimum y-values of the bounding box
        if x_min < 0:
            x_max -= x_min # if x_min less than zero, expand the x_max value by the absolute value of x_min, to ensure bounding box is same size
            x_min = 0

        if y_min < 0:
            y_max -= y_min # if y_min less than zero, expand the y_max value by the absolute value of y_min, to ensure bounding box is same size
            y_min = 0

        if largest_side < 20:
            x_min, x_max = 0, mask.shape[0]
            y_min, y_max = 0, mask.shape[1]
        return [x_min, y_min, x_max, y_max]


def make_video(image, mask, save_file, group='all', mask_frames='all', scale=1):

    
    N = st.session_state['N']
    if group in STRUCTURE_GROUPS:
        channels = STRUCTURE_GROUPS[group]
    else:                              # 'all' or unknown
        channels = foreground_indices

    if mask.ndim == 4:
        mask = np.eye(N, dtype=np.uint8)[mask]

    position = image.shape[2]
    timesteps = image.shape[3]

    grid_rows = int(np.sqrt(position) + 0.5)
    grid_cols = (position + grid_rows - 1) // grid_rows

    H, W = image.shape[:2]
    GIF_H = H*GIF_W/W
    H_scaled, W_scaled = round(GIF_H * scale), round(GIF_W * scale)

    try:
        font = load_font(int(20 * scale))
    except:
        font = ImageFont.load_default()

    frames = []
    if mask_frames == 'all':
        mask_frames = np.arange(timesteps)

    for t in mask_frames:
        canvas = Image.new(
            "RGBA",
            (grid_cols * W_scaled, grid_rows * H_scaled),
            color=(0, 0, 0, 255)
        )

        draw_canvas = ImageDraw.Draw(canvas)

        for idx in range(position):
            row, col = divmod(idx, grid_cols)

            img_slice = image[:, :, idx, t]

            p1, p99 = np.percentile(img_slice, [0.5, 99.5])
            img_slice = np.clip(img_slice, p1, p99)

            img_slice_norm = ((img_slice - img_slice.min()) /
                              (img_slice.max() - img_slice.min() + 1e-9) * 255).astype(np.uint8)
            img_rgb = np.stack([img_slice_norm] * 3, axis=-1)
            img_pil = Image.fromarray(img_rgb, mode="RGB").convert("RGBA")
            img_pil = img_pil.resize((W_scaled, H_scaled), resample=Image.NEAREST)

            # Build the FULL overlay across all channels first
            mH, mW = mask.shape[:2]
            overlay = np.zeros((mH, mW, 4), dtype=np.uint8)
            for ch in channels:
                ch_mask = mask[:, :, idx, t, ch]

                if np.any(ch_mask):
                    overlay[ch_mask > 0] = np.array(OVERLAY_COLORS_BY_IDX[ch], dtype=np.uint8)
                if np.any(ch_mask):
                    overlay[ch_mask > 0] = np.array(OVERLAY_COLORS_BY_IDX[ch], dtype=np.uint8)

            # Composite ONCE, after all channels drawn
            overlay_pil = Image.fromarray(overlay, mode="RGBA").resize(
                (W_scaled, H_scaled), resample=Image.NEAREST
            )
            img_pil = Image.alpha_composite(img_pil, overlay_pil)

            draw_tile = ImageDraw.Draw(img_pil)
            draw_tile.rectangle([0, 0, int(28*scale), int(22*scale)], fill=(211, 211, 211, 255))
            draw_tile.text((3*scale, 2*scale), f"{idx}", fill=(0, 0, 0, 255), font=font)

            canvas.paste(img_pil, (col * W_scaled, row * H_scaled), img_pil)

        draw_canvas.rectangle(
            [canvas.width - int(60*scale), canvas.height - int(20*scale),
             canvas.width, canvas.height],
            fill=(211,211,211,255)
        )
        draw_canvas.text(
            (canvas.width - int(55*scale), canvas.height - int(20*scale)),
            f"{t:02}/{timesteps - 1:02}",
            fill=(0,0,0,255),
            font=font
        )

        frames.append(canvas.convert("RGB"))

    if len(mask_frames) < 5:
        fps = len(mask_frames)/2
    else:
        fps = np.clip(len(mask_frames) / 2, 8, 15)

    save_file = save_file.replace('.gif','')
    pil_frames = [Image.fromarray(f) if not isinstance(f, Image.Image) else f
                  for f in frames]
    duration_ms = int(1000 / fps) if fps else 100
    pil_frames[0].save(
        f'{save_file}.gif',
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )


def calculate_whole_heart_metrics(mask, pixelspacing, thickness):
    """mask: (H, W, D, T) integer labels. Returns {structure: [volume_mL_per_t]}."""
    voxel_volume_ml = (pixelspacing[0] * pixelspacing[1] * thickness) / 1000.0
    T = mask.shape[-1]
    structures = {"LV": lv_idx, "RV": rv_idx, "LA": la_idx, "RA": ra_idx,
                  "Ao": ao_idx, "PA": pa_idx}
    return {
        name: [round(int((mask[..., t] == idx).sum()) * voxel_volume_ml, 2)
               for t in range(T)]
        for name, idx in structures.items()
    }

def thicken_close_fill_and_smooth(strokes, stroke_width):
    if strokes is None or not strokes.any():
        return strokes

    # Use power-law scaling for dilation
    dilation_factor = max(1, int(10 / (stroke_width ** 2)))

    # Detect contours to check for nested shapes
    dilated = binary_dilation(strokes, iterations=dilation_factor)
    contours = find_contours(dilated, 0.5)

    has_ring = False
    for i, c1 in enumerate(contours):
        for j, c2 in enumerate(contours):
            if i == j:
                continue
            y1, x1 = c1[:, 0], c1[:, 1]
            y2, x2 = c2[:, 0], c2[:, 1]
            if (y2.min() > y1.min() and y2.max() < y1.max() and
                x2.min() > x1.min() and x2.max() < x1.max()):
                has_ring = True
                break
        if has_ring:
            break

    if has_ring:
        # Dilation + fill + erosion
        closed = binary_dilation(strokes, iterations=dilation_factor)
        filled = binary_fill_holes(closed)
        filled = binary_erosion(filled, iterations=dilation_factor)
        return filled.astype('uint8')
    else:
        return strokes.astype('uint8')


def wrap(key, min_val, max_val):
    if st.session_state[key] > max_val:
        st.session_state[key] = min_val
    elif st.session_state[key] < min_val:
        st.session_state[key] = max_val



def frame_index_slider(
    T,
    frames,
    initial_idx,
    label,
    disabled_flag,
    key
):
    idx = st.slider(
        f"{label} | *{initial_idx}*",
        -1,
        T,
        value=initial_idx,
        key = key,
        on_change=wrap,
        args=(key, 0, T-1),
        disabled=disabled_flag
    )
    st.image(frames[idx], use_container_width=True)
    return idx

def copy_frames_channels(mask_name, dia_idx, sys_idx, blood_idx, myo_idx):
    frames = [dia_idx, sys_idx]
    channels = [blood_idx, myo_idx]

    mask = st.session_state[mask_name]
    if st.session_state.get('corrector_mask_edited'):
        source_mask = st.session_state['corrected_prior_mask']
        print("[DEBUG] Using corrected prior mask as source for final result.")
        print(f"[DEBUG] source_mask shape: {source_mask.shape}, mask shape: {mask.shape}")
    else:
        source_mask = st.session_state.preprocessed["smooth_mask"]
        print("[DEBUG] Using intial segmentation mask as source for final result.")
        print(f"[DEBUG] source_mask shape: {source_mask.shape}, mask shape: {mask.shape}")

    for f in frames:
        for c in channels:
            mask[:, :, :, f, c] = source_mask[:, :, :, f, c]

def confirm_selection(*args, **kwargs):
    """In whole-heart 3D mode there's no EDV/ESV to confirm; mark as done."""
    st.session_state['edv_esv_selected'].update({"confirmed": True})
    save_config(st.session_state['edv_esv_selected'], st.session_state['cache_config_path'])
    save_cached_mask(st.session_state['edited_mask'], save_path=st.session_state['cache_mask_path'])

def resize_to_original(edited_mask, raw_mask, crop_box, dia_idx, sys_idx, group='all'):
    x_min, y_min, x_max, y_max = crop_box
    final_mask = np.zeros_like(raw_mask)
    channels = STRUCTURE_GROUPS.get(group, foreground_indices)
    for ch in channels:
        for t_idx in (dia_idx, sys_idx):
            final_mask[y_min:y_max, x_min:x_max, ch, t_idx, :] = \
                edited_mask[:, :, ch, t_idx, :]
    return np.argmax(final_mask, axis=-1)


def make_true_vs_prior_vs_pred_mask_3d_gif(image, true_mask, prior_mask, pred_mask, gif_name = 'example_prediction', prior_image=None):
    ''' 
    Creates a side-by-side gif of the true vs prior vs predicted masks for a given image, with an overlay of the true, prior and predicted masks on the original image. 
    The function takes in the original image, the true mask, the prior mask, the predicted mask, a name for the gif, and optionally clinical data for the patient and a wandb run object to upload the gif. 
    The function checks the shape of the input image and transposes it if necessary to ensure it is in the format [z,y,x]. 
    It then creates a custom color map for the masks, extracts clinical data if provided, and creates a figure with two subplots for the true and predicted mask overlays. 
    It iterates through each slice of the image, creates the overlays, and appends them to a list of frames. 
    Finally, it creates an animation from the frames, saves it as a gif, uploads it to WandB if a run object is provided, and closes the plot to free memory.
    '''
    # print(f"image shape on entering gif code: {image.shape}")
    # Check if image_shape is in shape [z,y,x], and if not, change to this format
    if (image.shape[0] < image.shape[1]) & (image.shape[0] < image.shape[2]): # should be less slices than pixels in either x or y dimensions
            image = image
            true_mask = true_mask
            prior_mask = prior_mask
            pred_mask = pred_mask
            if prior_image is not None:
                prior_image = prior_image
    else:
            image = np.transpose(image, (2, 0, 1))
            true_mask = np.transpose(true_mask, (2, 0, 1))
            prior_mask = np.transpose(prior_mask, (2, 0, 1))
            pred_mask = np.transpose(pred_mask, (2, 0, 1))
            if prior_image is not None:
                prior_image = np.transpose(prior_image, (2, 0, 1))

    # print(f"image shape after transpose: {image.shape}")
    
    # Define the custom color map
    colour_map = {0: "black", 1: "red", 2: "blue", 3: "orange",
              4: "purple", 5: "yellow", 6: "cyan"}
    # Create a ListedColormap using the values from the dictionary
    custom_cmap = ListedColormap([colour_map[key] for key in sorted(colour_map.keys())])

    # Create figure and subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 10))
    ax1, ax2, ax3 = axes  # Unpack the axes for easier referencing

    frames = []
    slice_position = image.shape[0]

    for pos in range(slice_position):
        # True mask overlay
        p1 = ax1.imshow(image[pos, :, :], cmap='gray', vmin=np.min(image), vmax=np.max(image))
        p2 = ax1.imshow(true_mask[pos, :, :], alpha=0.2, cmap=custom_cmap, 
                        vmin=0, vmax=len(colour_map) - 1, interpolation='none')
        ax1.set_title(f"{gif_name}\nTrue Mask Overlay")
        ax1.axis('off')

        # Prior mask overlay
        if prior_image is None:
            p3 = ax2.imshow(image[pos, :, :], cmap='gray', vmin=np.min(image), vmax=np.max(image))
            p4 = ax2.imshow(prior_mask[pos, :, :], alpha=0.2, cmap=custom_cmap, 
                            vmin=0, vmax=len(colour_map) - 1, interpolation='none')
            ax2.set_title(f"{gif_name}\nPrior Mask Overlay")
            ax2.axis('off')
        elif prior_image is not None:
            p3 = ax2.imshow(prior_image[pos, :, :], cmap='gray', vmin=np.min(prior_image), vmax=np.max(prior_image))
            p4 = ax2.imshow(prior_mask[pos, :, :], alpha=0.2, cmap=custom_cmap, 
                            vmin=0, vmax=len(colour_map) - 1, interpolation='none')
            ax2.set_title(f"{gif_name}\nPrior Mask Overlay")
            ax2.axis('off')

        # Predicted mask overlay
        p5 = ax3.imshow(image[pos, :, :], cmap='gray', vmin=np.min(image), vmax=np.max(image))
        p6 = ax3.imshow(pred_mask[pos, :, :], alpha=0.2, cmap=custom_cmap, 
                        vmin=0, vmax=len(colour_map) - 1, interpolation='none')
        ax3.set_title(f"{gif_name}\nPredicted Mask Overlay")
        ax3.axis('off')

    
        # Append the frames for animation
        frames.append([p1, p2, p3, p4, p5, p6])

    # Check there are the correct number of frames
    if len(frames) != slice_position:
        raise ValueError("Number of frames is not equal to the number of slices in the image")

    # Create and upload animation to neptune
    output_dir = "True_vs_Prior_vs_Predicted_Mask_gifs"
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
    gif_path = os.path.join(output_dir, f"{gif_name}.gif")  # Save in the directory with a unique name
    ani = animation.ArtistAnimation(fig, frames, interval=200, blit=True)
    ani.save(gif_path, fps=3)

    plt.close()  # Close the plot to free memory

def plot_image_with_prior_masks(image, gif_name='example_prediction'):
    """Plots a 2d slice of a 3D image with its corresponding prior masks side by side, and an overlay of
    the masks on both the target image (channel 0) and the prior image (channel 1). Saves the plot to disk."""
    # image is (H, W, D, C) where channel 0 = target image, channel 1 = prior image, channels 2+ = prior mask classes
    num_prior_masks = image.shape[3] - 2  # subtract target and prior image channels
    # columns: target image, prior image, each mask channel, overlay on target, overlay on prior
    n_cols = num_prior_masks + 4
    fig, axes = plt.subplots(1, n_cols, figsize=(20 * n_cols, 20))

    mid_slice = image.shape[2] // 2
    colors = plt.cm.tab10.colors  # up to 10 distinct colours

    # Target image
    axes[0].imshow(image[:, :, mid_slice, 0], cmap='gray')
    axes[0].set_title('Target Image')
    axes[0].axis('off')

    # Prior image
    axes[1].imshow(image[:, :, mid_slice, 1], cmap='gray')
    axes[1].set_title('Prior Image')
    axes[1].axis('off')

    # Prior mask channels (one per column)
    for m in range(num_prior_masks):
        axes[2 + m].imshow(image[:, :, mid_slice, 2 + m], cmap='gray')
        axes[2 + m].set_title(f'Prior Mask {m + 1}')
        axes[2 + m].axis('off')

    def _draw_overlay(ax, bg_channel, title):
        ax.imshow(image[:, :, mid_slice, bg_channel], cmap='gray')
        for m in range(num_prior_masks):
            mask = image[:, :, mid_slice, 2 + m]
            color = colors[m % len(colors)]
            colored = np.zeros((*mask.shape, 4))  # RGBA
            colored[..., :3] = color[:3]
            colored[..., 3] = mask * 0.5  # alpha proportional to mask intensity
            ax.imshow(colored)
        ax.set_title(title)
        ax.axis('off')

    # Overlay on target image
    _draw_overlay(axes[-2], bg_channel=0, title='Target + Prior Masks')
    # Overlay on prior image
    _draw_overlay(axes[-1], bg_channel=1, title='Prior + Prior Masks')

    output_dir = "image_with_prior_masks_plots"
    # create output dir if doesnt exist
    os.makedirs(output_dir, exist_ok=True)

    plt.suptitle(gif_name)
    plt.savefig(f"{output_dir}/{gif_name}.png")
    plt.close()

def cv_zoom_iso(images, zoom, interpolation=cv2.INTER_CUBIC):
    """
    Resize axes 0, 1, 2 all by `zoom` (a scalar). Leaves T (and C) untouched.
    images: (A0, A1, A2, T) or (A0, A1, A2, T, C)
    """
    from scipy.ndimage import zoom as ndzoom
    if interpolation == cv2.INTER_NEAREST:
        order = 0
    elif interpolation == cv2.INTER_LINEAR:
        order = 1
    else:
        order = 3
    if images.ndim == 4:
        factors = (zoom, zoom, zoom, 1)
    elif images.ndim == 5:
        factors = (zoom, zoom, zoom, 1, 1)
    else:
        raise ValueError("Input must be 4D or 5D.")
    return ndzoom(images, factors, order=order)


def cv_zoom_mask_iso(mask, zoom, interpolation=cv2.INTER_NEAREST):
    """
    mask: (A0, A1, A2, T, C) one-hot. Resizes axes 0,1,2 isotropically.
    Returns one-hot of the same dtype.
    """
    zoomed = cv_zoom_iso(mask.astype(np.float32), zoom, interpolation=interpolation)
    C = zoomed.shape[-1]
    labels_int = np.argmax(zoomed, axis=-1).astype(np.uint8)
    return np.eye(C, dtype=np.uint8)[labels_int]