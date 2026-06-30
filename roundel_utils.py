from utils import *
import traceback
import torch
from im2sim.models import UNet


_HEART_HTML = """
<div id="c" style="width:100%;height:420px;background:#000;"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const meshes = __MESHES__;
const W = document.getElementById('c').clientWidth, H = 420;
const scene = new THREE.Scene();
const cam = new THREE.PerspectiveCamera(50, W/H, 0.1, 5000);
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(W, H);
document.getElementById('c').appendChild(renderer.domElement);
renderer.domElement.addEventListener('contextmenu', e => e.preventDefault());
scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const dl = new THREE.DirectionalLight(0xffffff, 0.6); dl.position.set(1,1,1); scene.add(dl);

// ---- build geometry ONCE, compute COM ----
const inner = new THREE.Group();
let sx=0, sy=0, sz=0, vcount=0;
meshes.forEach(m => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(m.vertices, 3));
    g.setIndex(m.faces);
    g.computeVertexNormals();
    const op = (m.opacity === undefined) ? 1.0 : m.opacity;
    const mat = new THREE.MeshPhongMaterial({
        color: m.color, side: THREE.DoubleSide, flatShading: false,
        shininess: 20, transparent: op < 1.0, opacity: op, depthWrite: op >= 1.0,
    });
    inner.add(new THREE.Mesh(g, mat));
    const pos = m.vertices;
    for (let i=0;i<pos.length;i+=3){ sx+=pos[i]; sy+=pos[i+1]; sz+=pos[i+2]; }
    vcount += pos.length/3;
});
const cx=vcount?sx/vcount:0, cy=vcount?sy/vcount:0, cz=vcount?sz/vcount:0;
inner.position.set(-cx, -cy, -cz);

const orient = new THREE.Group();
orient.add(inner);
orient.rotation.z = -Math.PI / 2;

const pivot = new THREE.Group();
pivot.add(orient);
pivot.rotation.y = Math.PI / 2;
scene.add(pivot);

// ---- camera framing (compute maxr from COM) ----
let maxr=1;
meshes.forEach(m => {
    const p = m.vertices;
    for (let i=0;i<p.length;i+=3){
        const dx=p[i]-cx, dy=p[i+1]-cy, dz=p[i+2]-cz;
        maxr = Math.max(maxr, Math.sqrt(dx*dx+dy*dy+dz*dz));
    }
});
const target = new THREE.Vector3(0, 0, 0);
cam.position.set(0, 0, maxr*2.5);
cam.lookAt(target);

// ---- handlers ----
let dragging=false, panning=false, px=0, py=0;
renderer.domElement.addEventListener('mousedown', e=>{
    px=e.clientX; py=e.clientY;
    if (e.button===0) dragging=true;
    else if (e.button===2) panning=true;
});
window.addEventListener('mouseup', ()=>{ dragging=false; panning=false; });
window.addEventListener('mousemove', e=>{
    const dx=e.clientX-px, dy=e.clientY-py;
    if (dragging){
        pivot.rotateOnWorldAxis(new THREE.Vector3(0,1,0), dx*0.01);
        pivot.rotateOnWorldAxis(new THREE.Vector3(1,0,0), dy*0.01);
    } else if (panning){
        const dist = cam.position.distanceTo(target);
        const panScale = dist * 0.0015;
        const right = new THREE.Vector3(), up = new THREE.Vector3();
        cam.matrixWorld.extractBasis(right, up, new THREE.Vector3());
        const pan = new THREE.Vector3()
            .addScaledVector(right, -dx * panScale)
            .addScaledVector(up,     dy * panScale);
        cam.position.add(pan); target.add(pan); cam.lookAt(target);
    }
    px=e.clientX; py=e.clientY;
});
renderer.domElement.addEventListener('wheel', e=>{
    e.preventDefault();
    const dir = new THREE.Vector3().subVectors(cam.position, target);
    dir.multiplyScalar(1 + e.deltaY*0.001);
    cam.position.copy(target).add(dir);
    cam.lookAt(target);
}, {passive:false});

(function animate(){ requestAnimationFrame(animate); renderer.render(scene,cam); })();
</script>
"""

def segmentation_view():
    st.session_state['N'] = 7
    st.header("Data Upload")

    if 'disable_upload' not in st.session_state:
        st.session_state['disable_upload'] = False
    
    col1, col2 = st.columns(2)
    with col1:
        zip_file = st.file_uploader(
            "Upload ZIP DICOM directory",
            type=["zip"],
            accept_multiple_files=False,
            disabled = st.session_state['disable_upload']
        )
        if zip_file:
            dcms = extract_dicom_from_zip(zip_file)
            if dcms:
                st.session_state['disable_upload'] = True
                print(f"[DEBUG] DICOM files extracted: {len(dcms)}")
                image, sax_df = Pipeline(dcms)
                st.session_state['sax_df'] = sax_df
                first_dcm = sax_df['dcm'].values[0]

                st.session_state.patient_name = str(first_dcm.PatientName) if hasattr(first_dcm, 'PatientName') and first_dcm.PatientName else 'Anonymised Patient'
                st.session_state.series_date = str(first_dcm.SeriesDate) if hasattr(first_dcm, 'SeriesDate') and first_dcm.SeriesDate else 'Unknown'
                st.session_state.series_description = str(first_dcm.SeriesDescription) if hasattr(first_dcm, 'SeriesDescription') and first_dcm.SeriesDescription else 'Unknown'
                st.session_state.pixelspacing = sax_df.pixelspacing.unique()[0]
                st.session_state.thickness = sax_df.thickness.unique()[0]
                st.session_state['sax_series_uid'] = first_dcm.SeriesInstanceUID

                st.session_state.n_slices = sax_df['slicelocation'].nunique()
                st.session_state.n_phases = sax_df.loc[sax_df['slicelocation'] == sax_df['slicelocation'].values[0]]['triggertime'].nunique()
        

    with col2:
        # Create dataframe
        if st.session_state['disable_upload']:
            dicom_data = {
                "Field": [
                    "Patient Name",
                    "Series Date",
                    "Series Description",
                    "Pixel Size",
                    "Slice Thickness",
                    "Number of Images",
                    "Number of Slices",
                    "Number of Phases",
                    "Slice × Phases"
                ],
                "Value": [
                    st.session_state.patient_name,
                    st.session_state.series_date,
                    st.session_state.series_description,
                    f"{st.session_state.pixelspacing} x {st.session_state.pixelspacing} mm",
                    f"{st.session_state.thickness} mm",
                    len(st.session_state['sax_df'] ),
                    st.session_state.n_slices,
                    st.session_state.n_phases,
                    st.session_state.n_slices * st.session_state.n_phases
                ]
            }

            df_dicom = pd.DataFrame(dicom_data).set_index('Field')

            # Display dataframe in Streamlit
            st.dataframe(df_dicom, use_container_width=True)

    
    if "initialized_all" not in st.session_state and st.session_state['disable_upload']:
        with st.spinner("Segmenting..."):
            segment_image(image)
        
        with st.spinner("Initialising..."):
            initialize_app()

    if "initialized_all" in st.session_state:
        st.success('Segmentation Confirmed! ⭕️')


def segment_image(image):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Resample to isotropic BEFORE inference (model trained isotropic) ---
    ps = float(st.session_state['pixelspacing'])
    th = float(st.session_state['thickness'])
    iso_image_list = []
    for t in range(image.shape[-1]):
        img_t = image[..., t].astype(np.float32)            # (H, W, D)
        img_iso, target_sp, zfac = resample_to_isotropic(
            img_t, in_spacing=(ps, ps, th), target=ps, order=1)
        iso_image_list.append(img_iso)
    image = np.stack(iso_image_list, axis=-1)               # (H', W', D', T) isotropic
    st.session_state['iso_spacing'] = target_sp             # record for everyone downstream
    st.session_state['iso_zoom'] = zfac

    # --- Build / cache the model -------------------------------------------
    if 'model' not in st.session_state:
        print("[DEBUG] Loading 3D model...")
        model = UNet(in_channels = 1,
                    out_channels = 7,
                    filters=[32,64,128,256,512],
                    pool_sizes=((2,2,2),(2,2,2),(2,2,2),(2,2,2)),
                    upsample_sizes=(2,2,2,2),
                    kernel_size=3,
                    conv_blocks_per_level=2,
                    rank=3,
                    activation="leaky_relu",
                    norm_type="BatchNorm",
                    dropout_rate=None,
                    final_activation="softmax")
        weights_path = f'{models_path}/SEG-25.pth'   # TODO: rename
        # If the checkpoint is a dict with extra keys, you may need:
        #   state = state['state_dict']
        loaded = torch.load(weights_path, map_location=device)
        model.load_state_dict(loaded.state_dict())
        st.session_state['model'] = model.eval().to(device)
        print("[DEBUG] Model loaded.")

    model = st.session_state['model']

    # --- Inference ----------------------------------------------------------
    mask = []
    for t in stqdm(range(image.shape[-1])):           # T=1 → runs once
        image_t = image[..., t].astype(np.float32)    # (H, W, D)
        image_norm = z_normalise_image(image_t.copy())
        image_padded, pads = pad_to_multiple(image_norm, multiple=16)

        X = image_padded.transpose(2, 0, 1)[np.newaxis, np.newaxis, ...]
        print(X.shape)
        X_tensor = torch.from_numpy(X).to(device)

        with torch.no_grad():
            logits = model(X_tensor)

        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        pred_padded = pred.transpose(1, 2, 0)               # back to (H, W, D)
        pred_mask = crop_back(pred_padded, pads)

        mask.append(pred_mask)

    mask = np.stack(mask, axis=-1)                    # (H, W, D, T=1)
    mask = postprocess(mask)

    N = st.session_state['N']
    for t in range(mask.shape[-1]):
        mask[..., t] = enforce_single_components(mask[..., t], n_labels=N)

    save_image(image, save_path=f'{data_path}/image___{st.session_state.sax_series_uid}.nii.gz')
    save_mask(mask,  save_path=f'{data_path}/masks___{st.session_state.sax_series_uid}.nii.gz')
    return mask
    

def pad_to_multiple(volume, multiple=16):
    """volume: (H, W, D) → padded (H', W', D'), returns padding tuple."""
    H, W, D = volume.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    pad_d = (multiple - D % multiple) % multiple
    padded = np.pad(volume, ((0, pad_h), (0, pad_w), (0, pad_d)), mode='constant')
    return padded, (pad_h, pad_w, pad_d)

def crop_back(volume_padded, pads):
    pad_h, pad_w, pad_d = pads
    H, W, D = volume_padded.shape
    return volume_padded[:H-pad_h if pad_h else None,
                         :W-pad_w if pad_w else None,
                         :D-pad_d if pad_d else None]

# --------------------------------------------------------------
# Initialization
# --------------------------------------------------------------
def initialize_app():
    stages = 5
    with stqdm(total=stages) as pbar:
        raw_image = load_nii(f'{data_path}/image___{st.session_state.sax_series_uid}.nii.gz')
        raw_mask = load_nii(f'{data_path}/masks___{st.session_state.sax_series_uid}.nii.gz').astype('uint8')

        raw_mask = np.eye(st.session_state['N'], dtype=np.uint8)[raw_mask]
        raw_shape = raw_image.shape

        # -----------------------------
        # Compute raw indices
        # -----------------------------
        lv_volume = np.sum(raw_mask[...,lv_idx], axis=(0,1,2))
        rv_volume = np.sum(raw_mask[...,rv_idx], axis=(0,1,2))

        T = raw_mask.shape[3]
        if T == 1:
            raw_lv_dia_idx = raw_lv_sys_idx = raw_rv_dia_idx = raw_rv_sys_idx = 0
            # Auto-confirm so downstream views don't gate on a meaningless slider
            st.session_state['edv_esv_selected'] = {
                "lv_dia_idx": 0, "lv_sys_idx": 0,
                "rv_dia_idx": 0, "rv_sys_idx": 0,
                "confirmed": True,
            }
        elif np.max(lv_volume) == 0:
            raw_lv_dia_idx, raw_lv_sys_idx = 0, min(15, T - 1)
            raw_rv_dia_idx, raw_rv_sys_idx = 0, min(15, T - 1)

        else:
            raw_lv_dia_idx = int(np.argmax(lv_volume))
            raw_lv_sys_idx = np.where(lv_volume != 0)[0][np.argmin(lv_volume[lv_volume != 0])]

            raw_rv_dia_idx = int(np.argmax(rv_volume))
            raw_rv_sys_idx = np.where(rv_volume != 0)[0][np.argmin(rv_volume[rv_volume != 0])]

        st.session_state.raw = {
            "image": raw_image,
            "mask": raw_mask,
            "shape": raw_shape,
            "raw_lv_dia_idx": raw_lv_dia_idx,
            "raw_lv_sys_idx": raw_lv_sys_idx,
            "raw_rv_dia_idx":raw_rv_dia_idx,
            "raw_rv_sys_idx":raw_rv_sys_idx
        }

        # -----------------------------
        # Initialize EDV/ESV selection
        # -----------------------------
        if "edv_esv_selected" not in st.session_state:
            st.session_state['edv_esv_selected'] = {"lv_dia_idx": None, "lv_sys_idx": None,"rv_dia_idx": None, "rv_sys_idx": None, "confirmed": False}

        # -----------------------------
        # Preprocess / crop if required
        # -----------------------------
        mask_channels = [i for i in range(st.session_state.N) if i != bg_idx]

        x_min, y_min, x_max, y_max = find_crop_box(np.max(raw_mask[...,mask_channels], axis=(-1,-2,-3)), crop_factor=1.5)
        st.session_state['subpixel_resolution'] = 2

        preprocessed_image = raw_image[y_min:y_max, x_min:x_max, :, :]
        preprocessed_mask = raw_mask[y_min:y_max, x_min:x_max, :, :, :].astype('uint8')
        H, W, D, T, N = preprocessed_mask.shape

        pbar.update(1)

        has_masks = np.where(np.sum(preprocessed_mask[...,mask_channels], axis = (0,1,3,-1))>0)[0]
        if len(has_masks) == 0:
            has_masks = np.array([1,2,3,4,5,6])

        mid_slice = len(has_masks)//2
        

        smoothed_image = preprocessed_image          # no upsample — native res
        smoothed_mask = preprocessed_mask            # native one-hot


        st.session_state['cache_config_path'] = f"{cache_dir}/config___{st.session_state.sax_series_uid}.json"
        st.session_state['cache_mask_path'] = f"{cache_dir}/masks___{st.session_state.sax_series_uid}.npy"

        pbar.update(1)

        if os.path.exists(st.session_state['cache_config_path']) and os.path.exists(st.session_state['cache_mask_path']):
            smoothed_mask = load_cached_mask(st.session_state['cache_mask_path']).astype("uint8")
            cached = True
        else:
            smoothed_mask = preprocessed_mask
            cached = False


        make_video(smoothed_image[:,:,has_masks[mid_slice-3:mid_slice+3],:], smoothed_mask[:,:,has_masks[mid_slice-3:mid_slice+3],:, :] * 0, save_file=edv_esv_gif_path)
        pbar.update(1)
        make_video(smoothed_image, smoothed_mask*0, save_file=blank_gif_path)
        pbar.update(1)
        preview_gif_path = f'{results_path}/temp/preview'
        make_video(smoothed_image, smoothed_mask, save_file=preview_gif_path)
        st.session_state['preview_gif_path'] = f'{preview_gif_path}.gif'
        pbar.update(1)


        gif = Image.open(f'{edv_esv_gif_path}.gif')

        st.session_state.preprocessed = {
            "image": preprocessed_image,
            "mask": preprocessed_mask,
            "smooth_image": smoothed_image,
            "smooth_mask": smoothed_mask,
            "H": H,
            "W": W,
            "D": D,
            "T": T,
            "N": N,
            "edv_esv_frames": [frame.copy() for frame in ImageSequence.Iterator(gif)],
            "crop_box": [x_min, y_min, x_max, y_max],
        }


        st.session_state['edited_mask'] = st.session_state.preprocessed["smooth_mask"].copy()

        if cached:
            config = load_config(st.session_state['cache_config_path'])
            confirm_selection(lv_dia_idx=config['lv_dia_idx'], 
                            rv_dia_idx=config['rv_dia_idx'], 
                            lv_sys_idx=config['lv_sys_idx'], 
                            rv_sys_idx=config['rv_sys_idx'])

        # -----------------------------
        # Initialize edited mask
        st.session_state['lv_frames'] = None
        st.session_state['rv_frames'] = None
        st.session_state["view_mode"] = 'Static'
        st.session_state["brush_mode"] = "Paint ✏️"
        st.session_state["stroke_width"] = "thin"
        st.session_state['edit_made'] = False
        st.session_state['cached'] = cached
        st.session_state["saved"] = False
        st.session_state['canvas_gen'] = 0
        st.session_state.initialized_all = True
        st.session_state['seg_baseline'] = st.session_state.raw['mask'].copy()   # one-hot, full
        # per-plane corrected masks start as copies of the baseline crop
        st.session_state['mask_trans']   = st.session_state['edited_mask'].copy()
        st.session_state['mask_coronal'] = st.session_state['edited_mask'].copy()
        st.session_state['edited_trans_slices']   = set()
        st.session_state['edited_coronal_slices'] = set()


def preview_segmentation_view():
    if not st.session_state.get('initialized_all'):
        st.info("Run segmentation first to preview the result.")
        return

    st.header("Preview Segmentation")

    image = st.session_state.preprocessed["smooth_image"]   # (H, W, D, T)
    mask  = st.session_state['edited_mask']                  # (H, W, D, T, N)
    H, W, D, T, N = [st.session_state.preprocessed[k] for k in ["H","W","D","T","N"]]
    idx = 0  # static volume

    # choose how many slices per orientation
    n_per = 8

    def slices_with_mask(axis):
        fg = np.any(np.argmax(mask[..., idx, :], axis=-1) > 0,
                    axis=tuple(a for a in range(3) if a != axis))
        present = np.where(fg)[0]
        if len(present) == 0:
            present = np.arange(mask.shape[axis])
        lo, hi = present[0], present[-1]
        # sample n_per points strictly INSIDE the populated range,
        # excluding the near-empty extreme slices
        picks = np.linspace(lo, hi, n_per + 2).round().astype(int)[1:-1]
        return picks

    def overlay_slice(axis, d):
        if axis == 0:
            img = image[d, :, :, idx];      msk = mask[d, :, :, idx, :]
        elif axis == 1:
            img = image[:, d, :, idx];      msk = mask[:, d, :, idx, :]
        else:
            img = image[:, :, d, idx];      msk = mask[:, :, d, idx, :]
        img8 = (normalize(img) * 255).astype(np.uint8)
        ov = get_overlay(img8, msk, img8.shape[0], img8.shape[1], N,
                         OVERLAY_COLORS_BY_IDX, 'all')
        return ov

    orientations = [(0, "Transverse"), (1, "Coronal"), (2, "Sagittal")]

    for axis, label in orientations:
        st.markdown(f"#### {label}")
        picks = slices_with_mask(axis)
        cols = st.columns(len(picks))
        for c, d in zip(cols, picks):
            with c:
                ov = overlay_slice(axis, int(d))
                st.image(ov, caption=f"{label[:3]} {int(d)}", use_container_width=True)

    st.markdown(
        "<div style='background-color: rgba(76, 175, 80, 0.25); "
        "padding: 8px 12px; border-radius: 6px; border: 1px solid rgba(76, 175, 80, 0.6);'>"
        "✅ Review the segmentation across orientations. If corrections are needed, "
        "proceed to the Mask Editor.</div>",
        unsafe_allow_html=True,
    )

def run_corrector_model(mode="dual15"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if 'corrector_model_15' not in st.session_state:
        print("[DEBUG] Loading 15ch corrector...")
        model = UNet(
            in_channels=15, out_channels=7,
            filters=[32,64,128,256,512],
            pool_sizes=((2,2,2),(2,2,2),(2,2,2),(2,2,2)),
            upsample_sizes=(2,2,2,2),
            kernel_size=3, conv_blocks_per_level=2,
            rank=3, activation="leaky_relu", norm_type="BatchNorm",
            dropout_rate=None, final_activation="softmax",
        )
        loaded = torch.load(f'{models_path}/SEG_COR-31.pth', map_location=device)  # TODO filename
        model.load_state_dict(loaded.state_dict())
        st.session_state['corrector_model_15'] = model.eval().to(device)
    model = st.session_state['corrector_model_15']

    N = st.session_state['N']
    raw_image = st.session_state.raw['image']
    x_min, y_min, x_max, y_max = st.session_state.preprocessed['crop_box']

    baseline_full = np.argmax(st.session_state['seg_baseline'], axis=-1)        # (H,W,D,T)

    trans_crop = np.argmax(st.session_state['mask_trans'],   axis=-1)            # (Hc,Wc,D,T)
    coron_crop = np.argmax(st.session_state['mask_coronal'], axis=-1)

    mixed_trans_full = baseline_full.copy()
    mixed_trans_full[y_min:y_max, x_min:x_max, :, :] = trans_crop
    mixed_coron_full = baseline_full.copy()
    mixed_coron_full[y_min:y_max, x_min:x_max, :, :] = coron_crop

    def onehot_fg(label_vol):
        # label_vol: (H,W,D) ints 0..6 -> (6,H,W,D) one-hot, background dropped
        oh = np.eye(N, dtype=np.float32)[label_vol]        # (H,W,D,N)
        return oh[..., 1:].transpose(3, 0, 1, 2)           # (6,H,W,D)

    new_masks = []
    for t in stqdm(range(raw_image.shape[-1])):
        base_t = baseline_full[..., t]
        mt = mixed_trans_full[..., t]
        mc = mixed_coron_full[..., t]

        ind_t = (base_t != mt).astype(np.float32)          # (H,W,D)
        ind_c = (base_t != mc).astype(np.float32)

        img_norm = z_normalise_image(raw_image[..., t].astype(np.float32).copy())  # (H,W,D)

        # pad everything to multiple of 16 (on H,W,D)
        img_p, pads = pad_to_multiple(img_norm, multiple=16)
        ph, pw, pd = pads
        def pad3(a):  return np.pad(a, ((0,ph),(0,pw),(0,pd)), mode='constant')
        def pad4(a):  return np.pad(a, ((0,0),(0,ph),(0,pw),(0,pd)), mode='constant')  # (6,H,W,D)

        trans_oh = pad4(onehot_fg(mt))                     # (6,H',W',D')
        coron_oh = pad4(onehot_fg(mc))                     # (6,H',W',D')
        img_c    = pad3(img_norm)[np.newaxis, ...]         # (1,H',W',D')
        ind_t_c  = pad3(ind_t)[np.newaxis, ...]            # (1,H',W',D')
        ind_c_c  = pad3(ind_c)[np.newaxis, ...]

        # channel order: [img, trans_oh(6), ind_t, coron_oh(6), ind_c]  -> 15
        X = np.concatenate([img_c, trans_oh, ind_t_c, coron_oh, ind_c_c], axis=0)  # (15,H',W',D')

        # spatial transpose to (D,H,W) on each channel
        X = X.transpose(0, 3, 1, 2)                        # (15, D, H, W)
        X_tensor = torch.from_numpy(X[np.newaxis, ...]).to(device).float()  # (1,15,D,H,W)

        with torch.no_grad():
            logits = model(X_tensor)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)  # (D,H,W)
        new_masks.append(crop_back(pred.transpose(1, 2, 0), pads))             # (H,W,D)

    new_mask = postprocess(np.stack(new_masks, axis=-1))
    N = st.session_state['N']
    for t in range(new_mask.shape[-1]):
        new_mask[..., t] = enforce_single_components(new_mask[..., t], n_labels=N)

    # persist + reset (iterative baseline)
    save_mask(new_mask, save_path=f'{data_path}/masks___{st.session_state.sax_series_uid}.nii.gz')
    new_onehot = np.eye(N, dtype=np.uint8)[new_mask]
    new_crop = new_onehot[y_min:y_max, x_min:x_max, :, :, :]
    st.session_state.raw['mask']     = new_onehot
    st.session_state['seg_baseline'] = new_onehot.copy()
    st.session_state['edited_mask']  = new_crop.copy()
    st.session_state['mask_trans']   = new_crop.copy()
    st.session_state['mask_coronal'] = new_crop.copy()
    st.session_state['edited_trans_slices']   = set()
    st.session_state['edited_coronal_slices'] = set()
    st.session_state.preprocessed['mask']        = new_crop
    st.session_state.preprocessed['smooth_mask'] = new_crop

    preview_gif_path = f'{results_path}/temp/preview'
    make_video(st.session_state.preprocessed['smooth_image'], new_crop, save_file=preview_gif_path)
    st.session_state['preview_gif_path'] = f'{preview_gif_path}.gif'
    st.session_state['canvas_gen'] += 1
    st.session_state['edit_made'] = True
    return new_mask

def corrector_model_view():
    st.header("Corrector Model")
    st.info("Corrector model not yet available for whole-heart 3D segmentation.")
    return


def edv_esv_view():
    """Full EDV/ESV Finder view layout."""
    if not st.session_state['initialized_all']:
        st.error("Select and confirm EDV/ESV first.")
        st.stop()

    if "edv_esv_selected" not in st.session_state:
        st.session_state['edv_esv_selected'] = {"lv_dia_idx": None, "lv_sys_idx": None, "rv_dia_idx": None, "rv_sys_idx": None,"confirmed": False}
    
    H, W, D, T, N = [st.session_state.preprocessed[k] for k in ["H","W","D","T","N"]]
    edv_esv_frames= st.session_state.preprocessed['edv_esv_frames']


    if st.session_state.edv_esv_selected['confirmed']:
        display_lv_dia_idx=st.session_state.edv_esv_selected['lv_dia_idx']
        display_rv_dia_idx=st.session_state.edv_esv_selected['rv_dia_idx']
        display_lv_sys_idx=st.session_state.edv_esv_selected['lv_sys_idx']
        display_rv_sys_idx=st.session_state.edv_esv_selected['rv_sys_idx']
    else:
        display_lv_dia_idx=st.session_state.raw['raw_lv_dia_idx']
        display_rv_dia_idx=st.session_state.raw['raw_rv_dia_idx'] 
        display_lv_sys_idx=st.session_state.raw['raw_lv_sys_idx'] 
        display_rv_sys_idx=st.session_state.raw['raw_rv_sys_idx'] 

    disabled_flag = st.session_state['edv_esv_selected']["confirmed"]

    col_lv, col_rv = st.columns(2)

    with col_lv:
        st.markdown('#### Left Ventricle')
        col_edv, col_esv = st.columns(2)

        with col_edv:
            lv_dia_idx = frame_index_slider(T, edv_esv_frames, display_lv_dia_idx, 'LV End-Diastolic Index', disabled_flag, key = 'lv_edv')

        with col_esv:
            lv_sys_idx = frame_index_slider(T, edv_esv_frames, display_lv_sys_idx, 'LV End-Systolic Index',disabled_flag, key = 'lv_esv')

    with col_rv:
        st.markdown('#### Right Ventricle')
        col_edv, col_esv = st.columns(2)
        with col_edv:
            rv_dia_idx = frame_index_slider(T, edv_esv_frames, display_rv_dia_idx, 'RV End-Diastolic Index', disabled_flag, key = 'rv_edv')

        with col_esv:
            rv_sys_idx = frame_index_slider(T, edv_esv_frames, display_rv_sys_idx, 'RV End-Systolic Index',disabled_flag, key = 'rv_esv')


    st.write('')
    if not disabled_flag:
        st.button(
            "Confirm EDV | ESV",
            on_click=lambda: confirm_selection(lv_dia_idx, lv_sys_idx, rv_dia_idx, rv_sys_idx),
            type="primary",
            use_container_width=True
        )
    else:
        st.success("EDV | ESV Confirmed! 🔍")

    st.write('')
    col1, col2 = st.columns(2)
    with col1:
        if st.button("EDV/ESV Only", use_container_width=True, disabled=not disabled_flag):
            st.session_state.mask_editor_mode = "edv_esv_only"
            st.session_state.current_view = "Mask Editor 🔧"
            st.rerun()
    with col2:
        if st.button("All frames", use_container_width=True, type="primary", disabled=not disabled_flag):
            st.session_state.mask_editor_mode = "all_frames"
            st.session_state.current_view = "Mask Editor 🔧"
            st.rerun()



def slice_navigation(D):
    if "slice_idx" not in st.session_state:
        st.session_state.slice_idx = 0
    if "previous_slice_idx" not in st.session_state:
        st.session_state.previous_slice_idx = st.session_state.slice_idx

    # Store previous slice
    previous_d = st.session_state.previous_slice_idx

    # Slider (updates slice_idx immediately)
    st.slider("Slice Index", 0, D - 1,key="slice_idx")

    col_prev, col_next = st.columns(2)
    with col_prev:
        st.button(
            "Previous",
            on_click=lambda: st.session_state.update(
                slice_idx=max(0, st.session_state.slice_idx - 1)
            ),
            use_container_width=True,
        )
    with col_next:
        st.button(
            "Next",
            on_click=lambda: st.session_state.update(
                slice_idx=min(D - 1, st.session_state.slice_idx + 1)
            ),
            use_container_width=True,
        )

    # Determine if canvas needs reset
    previous_objects = st.session_state.get('canvas', {}).get('previous_objects', [])
    reset_canvas = previous_d != st.session_state.slice_idx and bool(previous_objects)

    # Update previous slice for next rerun
    st.session_state.previous_slice_idx = st.session_state.slice_idx

    return st.session_state.slice_idx, reset_canvas

def slice_navigation_value(D):
    """Read/clamp the slice index WITHOUT rendering the widget."""
    if "slice_idx" not in st.session_state:
        st.session_state.slice_idx = 0
    return max(0, min(st.session_state.slice_idx, D - 1))

def slice_navigation_widget(D):
    """Render the slider + prev/next buttons. Call AFTER the image."""
    previous_d = st.session_state.get("previous_slice_idx", st.session_state.slice_idx)
    st.slider("Slice Index", 0, D - 1, key="slice_idx")
    col_prev, col_next = st.columns(2)
    with col_prev:
        st.button("Previous", on_click=lambda: st.session_state.update(
            slice_idx=max(0, st.session_state.slice_idx - 1)), use_container_width=True)
    with col_next:
        st.button("Next", on_click=lambda: st.session_state.update(
            slice_idx=min(D - 1, st.session_state.slice_idx + 1)), use_container_width=True)
    previous_objects = st.session_state.get('canvas', {}).get('previous_objects', [])
    reset_canvas = previous_d != st.session_state.slice_idx and bool(previous_objects)
    st.session_state.previous_slice_idx = st.session_state.slice_idx
    return st.session_state.slice_idx, reset_canvas

def get_overlay(image_slice, mask_state, H, W, N, OVERLAY_COLORS, group):
    if group in STRUCTURE_GROUPS:
        channels = STRUCTURE_GROUPS[group]
    else:                                          # 'all' or anything else
        channels = foreground_indices   

    overlay = Image.fromarray(np.stack([image_slice]*3, axis=-1)).convert("RGBA")
    for i in channels:
        ch_mask = mask_state[:, :, i]
        if np.any(ch_mask):
            mask_img = np.zeros((*mask_state.shape[:2], 4), dtype=np.uint8)
            mask_img[ch_mask > 0] = OVERLAY_COLORS[i]
            overlay = Image.alpha_composite(overlay, Image.fromarray(mask_img))
    return overlay



def select_brush(N, group):
    if group in STRUCTURE_GROUPS:
        valid_channels = STRUCTURE_GROUPS[group]
    else:
        valid_channels = list(BRUSH_LABELS.keys())   # all 6
    options = {idx: BRUSH_LABELS[idx] for idx in valid_channels}
    """Brush selection UI for channel, action, and stroke width."""
    action = st.radio("Brush Stroke Selection", 
                      options=["Paint ✏️", "Erase ✂️"],  
                      index=["Paint ✏️", "Erase ✂️"].index(st.session_state.brush_mode),
                      horizontal=True)
    st.session_state['brush_mode'] = action
    
    stroke_width_map = {"thin":6,"medium":20,"thick":40}
    stroke_width_sel = st.radio("Stroke Width", 
                                options=list(stroke_width_map.keys()),  
                                index= list(stroke_width_map.keys()).index(st.session_state["stroke_width"]), 
                                horizontal=True)
    st.session_state['stroke_width'] = stroke_width_sel

    if action == "Paint ✏️":
        channel = st.radio(
            "Mask",
            options=valid_channels,
            format_func=lambda x: BRUSH_LABELS[x],
            index=0,
            horizontal=True
        )
    else:
        channel = 0
    stroke_width = stroke_width_map[stroke_width_sel]
    return channel, action, stroke_width
def build_heart_geometry(mask_onehot, spacing=(1.0, 1.0, 1.0)):
    from skimage import measure
    from scipy.ndimage import gaussian_filter
    structures = {
        lv_idx: ("LV", "#c71f1f"), rv_idx: ("RV", "#1f59db"),
        la_idx: ("LA", "#f08c3d"), ra_idx: ("RA", "#8c4fc7"),
        ao_idx: ("Ao", "#dbc629"), pa_idx: ("PA", "#1fb3b3"),
    }
    mask = mask_onehot[..., 0, :]
    meshes = []
    for ch, (name, color) in structures.items():
        vol = mask[..., ch].astype(np.float32)
        if vol.sum() == 0:
            continue
        # Smooth the binary volume → marching cubes on the smoothed field
        # gives rounded surfaces instead of voxel stairs.
        vol_s = gaussian_filter(vol, sigma=1.0)
        try:
            verts, faces, _, _ = measure.marching_cubes(
                vol_s, level=0.5, spacing=spacing, step_size=1)
        except (ValueError, RuntimeError):
            continue
        meshes.append({
            "name": name, "color": color,
            "vertices": verts.astype(np.float32).ravel().tolist(),
            "faces": faces.astype(np.uint32).ravel().tolist(),
        })
    return meshes

def _nii_to_bytes(arr, affine=None):
    """Serialise a numpy array to gzipped NIfTI bytes (in-memory)."""
    import gzip, io
    if affine is None:
        affine = np.eye(4)
    img = nib.Nifti1Image(arr.astype(np.uint8), affine=affine, dtype='uint8')
    raw = img.to_bytes()                 # uncompressed .nii bytes
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode='wb') as gz:
        gz.write(raw)
    return out.getvalue()

def mask_editor_view():
    if not st.session_state['edv_esv_selected']["confirmed"]:
        st.error("Select and confirm EDV/ESV first.")
        st.stop()

    if 'canvas_gen' not in st.session_state:
        st.session_state['canvas_gen'] = 0

    # Prime the media store on first entry (before any layout)
    if not st.session_state.get('_editor_primed'):
        st.session_state['_editor_primed'] = True
        st.rerun()

    sub = st.session_state['subpixel_resolution']
    col1, col2, col3 = st.columns([1, 1.5, 1.5])

    H, W, D, T, N = [st.session_state.preprocessed[k] for k in ["H", "W", "D", "T", "N"]]
    image = st.session_state.preprocessed["smooth_image"]
    edited_mask = st.session_state['edited_mask']

    idx = 0 if T == 1 else st.session_state.get("frame_idx", 0)
    if "edit_plane" not in st.session_state:
        st.session_state["edit_plane"] = "Transverse"
    plane = st.session_state["edit_plane"]

    if plane == "Transverse":
        n_slices = image.shape[0]
        def get_img(d):  return image[d, :, :, idx]
        def get_msk(d):  return edited_mask[d, :, :, idx, :]
        def write_msk(d, ch, painted):
            edited_mask[d, :, :, idx, :][painted > 0] = 0
            edited_mask[d, :, :, idx, ch][painted > 0] = 1
        def clear_slice(d):  edited_mask[d, :, :, idx, :] = 0
        def dest_shape(d, ch):  return edited_mask[d, :, :, idx, ch].shape
        def plane_mask(): return st.session_state['mask_trans']
        def write_msk_to(m, d, ch, painted):
            m[d, :, :, idx, :][painted > 0] = 0
            m[d, :, :, idx, ch][painted > 0] = 1
        def clear_slice_to(m, d):  m[d, :, :, idx, :] = 0
    else:  # Coronal
        n_slices = image.shape[1]
        def get_img(d):  return image[:, d, :, idx]
        def get_msk(d):  return edited_mask[:, d, :, idx, :]
        def write_msk(d, ch, painted):
            edited_mask[:, d, :, idx, :][painted > 0] = 0
            edited_mask[:, d, :, idx, ch][painted > 0] = 1
        def clear_slice(d):  edited_mask[:, d, :, idx, :] = 0
        def dest_shape(d, ch):  return edited_mask[:, d, :, idx, ch].shape
        def plane_mask(): return st.session_state['mask_coronal']
        def write_msk_to(m, d, ch, painted):
            m[:, d, :, idx, :][painted > 0] = 0
            m[:, d, :, idx, ch][painted > 0] = 1
        def clear_slice_to(m, d):  m[:, d, :, idx, :] = 0

    slice_key = f"slice_idx_{plane}"
    if slice_key not in st.session_state:
        st.session_state[slice_key] = n_slices // 2
    d = max(0, min(st.session_state[slice_key], n_slices - 1))


    # ---------- col1: brush + mode ----------
    with col1:
        edit_mode = st.radio("Mode", ["Editor", "Viewer"], horizontal=True)
        channel, action, stroke_width = select_brush(N, 'all')
        if T > 1:
            idx = st.slider("Frame Index", 0, T - 1, value=0, key="frame_idx")

    image_slice = (normalize(get_img(d)) * 255).astype(np.uint8)
    mask_slice = get_msk(d)           # (A1, A2, C)

    # ---------- col2: axial editor ----------
    with col2:        
        st.radio("Editing plane", ["Transverse", "Coronal"],
                 horizontal=True, key="edit_plane")
        stroke_color = (
            f"rgba{OVERLAY_COLORS_BY_IDX[bg_idx][:3] + (0.7,)}"
            if action == "Erase ✂️"
            else f"rgba{OVERLAY_COLORS_BY_IDX[channel][:3] + (0.65,)}"
        )

        slice_h, slice_w = image_slice.shape
        canvas_h = DISPLAY_W * slice_h / slice_w

        if edit_mode == 'Viewer':
            ov = get_overlay(image_slice, mask_slice, H, W, N, OVERLAY_COLORS_BY_IDX, 'all')
            st.image(ov, width=DISPLAY_W)
        else:
            # Key changes on slice change AND on save (canvas_gen) → always fresh
            # Stable key: does NOT include d, so the canvas does not remount per slice.
            canvas_key = f"editor_{plane}"
            background_image = get_overlay(image_slice, mask_slice, H, W, N,
                                           OVERLAY_COLORS_BY_IDX, 'all')
            
            if not st.session_state.get('_canvas_warmed'):
                st.image(background_image, width=DISPLAY_W)
                st.caption("Loading editor…")
                st.session_state['_canvas_warmed'] = True
                st.rerun()

            # Clear strokes whenever slice/plane/gen changes (since the canvas persists)
            empty_sig = (plane, d, st.session_state['canvas_gen'])
            force_empty = st.session_state.get('_canvas_empty_sig') != empty_sig
            st.session_state['_canvas_empty_sig'] = empty_sig

            canvas_kwargs = dict(
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_image=background_image,
                update_streamlit=True,
                height=canvas_h,
                width=DISPLAY_W,
                drawing_mode='freedraw',
                key=canvas_key,
            )
            if force_empty:
                canvas_kwargs['initial_drawing'] = {"version": "4.4.0", "objects": []}

            canvas_result = st_canvas(**canvas_kwargs)
            current_objects = []
            if canvas_result and canvas_result.json_data:
                current_objects = canvas_result.json_data.get("objects", [])

            st.slider("Slice Index", 0, n_slices - 1, key=slice_key)
            col_prev, col_next = st.columns(2)
            with col_prev:
                st.button("Previous", use_container_width=True,
                          on_click=lambda: st.session_state.update(
                              **{slice_key: max(0, st.session_state[slice_key] - 1)}))
            with col_next:
                st.button("Next", use_container_width=True,
                          on_click=lambda: st.session_state.update(
                              **{slice_key: min(n_slices - 1, st.session_state[slice_key] + 1)}))
        # Save / clear
        if edit_mode != 'Viewer':
            col_save, col_clear = st.columns([1, 0.3])
            with col_save:
                save_contour = st.button('Save Contour', type='primary', use_container_width=True)
                if save_contour and canvas_result and canvas_result.image_data is not None and current_objects:
                    brush_data = np.array(canvas_result.image_data)
                    rgb = brush_data[:, :, :3].astype(np.float32)
                    alpha = brush_data[:, :, 3].astype(np.float32) / 255.0

                    overlay_colors_list = np.array(
                        [c[:3] for c in OVERLAY_COLORS_BY_IDX.values()], dtype=np.float32)
                    overlay_channels = list(OVERLAY_COLORS_BY_IDX.keys())

                    h, w, _ = rgb.shape
                    distances = np.linalg.norm(
                        rgb.reshape(-1, 3)[:, None, :] - overlay_colors_list[None, :, :], axis=-1)
                    closest_idx = np.argmin(distances, axis=1)
                    alpha_flat = alpha.flatten()

                    for ci, ch in enumerate(overlay_channels):
                        painted = ((closest_idx == ci) & (alpha_flat > 0)).reshape(h, w).astype(np.uint8)
                        painted = thicken_close_fill_and_smooth(painted, stroke_width)
                        if not np.any(painted): continue
                        ds = dest_shape(d, ch)
                        resized = np.array(Image.fromarray(painted).resize(
                            (ds[1], ds[0]), resample=Image.NEAREST))
                        # write to combined display mask
                        write_msk(d, ch, resized)
                        # write to the current plane's pure mask too
                        write_msk_to(plane_mask(), d, ch, resized)

                    if plane == "Transverse":
                        st.session_state['edited_trans_slices'].add(d)
                    else:
                        st.session_state['edited_coronal_slices'].add(d)

                    st.session_state['edit_made'] = True
                    save_cached_mask(edited_mask, save_path=st.session_state['cache_mask_path'])
                    st.session_state['canvas_gen'] += 1
                    st.rerun()

            with col_clear:
                if st.button('❌', use_container_width=True):
                    clear_slice(d)
                    clear_slice_to(plane_mask(), d) 
                    if plane == "Transverse":
                        st.session_state['edited_trans_slices'].add(d)
                    else:
                        st.session_state['edited_coronal_slices'].add(d)
                    save_cached_mask(edited_mask, save_path=st.session_state['cache_mask_path'])
                    st.session_state['edit_made'] = True
                    st.session_state['canvas_gen'] += 1
                    st.rerun()

    # ---------- col3: coronal read-only ----------
    with col3:
        st.caption("3D Heart Render")

        need_render = ('heart_geom' not in st.session_state
                       or st.button("🔄 Update 3D", use_container_width=True))
        if need_render:
            with st.spinner("Building 3D surfaces..."):
                st.session_state['heart_geom'] = build_heart_geometry(edited_mask)
                st.session_state['heart_render_gen'] = st.session_state.get('canvas_gen', 0)

        meshes = st.session_state.get('heart_geom', [])

        # Per-structure visibility toggles
        st.caption("Structure opacity")
        struct_names = [m["name"] for m in meshes]
        cols = st.columns(len(struct_names)) if struct_names else []
        opacities = {}
        for i, name in enumerate(struct_names):
            with cols[i]:
                solid = st.checkbox(name, value=True, key=f"vis_{name}")
                opacities[name] = 1.0 if solid else 0.1

        import json
        meshes_with_alpha = [
            {**m, "opacity": opacities.get(m["name"], 1.0)} for m in meshes
        ]
        meshes_json = json.dumps(meshes_with_alpha)

        html = _HEART_HTML.replace("__MESHES__", meshes_json)
        import streamlit.components.v1 as components
        components.html(html, height=480, scrolling=False)

        if st.session_state.get('heart_render_gen') != st.session_state.get('canvas_gen'):
            st.caption("⚠️ Edits since last render — click Update 3D.")

    st.divider()
    if st.button("🪄 Run Corrector Model", type="primary", use_container_width=True):
            with st.spinner("Running corrector..."):
                run_corrector_model(mode="dual15")
                # in run_corrector_model or on corrector completion:
                st.session_state['_editor_primed'] = False
                st.rerun()

    # ---- Download masks as a zipped {patient}_masks folder ----
    import io, zipfile
    patient = str(st.session_state.get('patient_name', 'patient'))
    safe = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_'
                   for c in patient).strip().replace(' ', '_')

    raw_shape = st.session_state.raw['mask'].shape[:4]
    x_min, y_min, x_max, y_max = st.session_state.preprocessed['crop_box']
    full_onehot = np.zeros(raw_shape + (N,), dtype=np.uint8)
    full_onehot[y_min:y_max, x_min:x_max, :, :, :] = st.session_state['edited_mask']
    full_labels = np.argmax(full_onehot, axis=-1).astype(np.uint8)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{safe}_masks/combined.nii.gz", _nii_to_bytes(full_labels))
        struct_names = {lv_idx:"LV", rv_idx:"RV", la_idx:"LA",
                        ra_idx:"RA", ao_idx:"Ao", pa_idx:"PA"}
        for ch, nm in struct_names.items():
            binary = (full_labels == ch).astype(np.uint8)
            if binary.sum() == 0:
                continue
            zf.writestr(f"{safe}_masks/{nm}.nii.gz", _nii_to_bytes(binary))
    zip_buf.seek(0)

    st.download_button(
        "💾 Download masks (.nii.gz)",
        data=zip_buf,
        file_name=f"{safe}_masks.zip",
        mime="application/zip",
        use_container_width=True,
    )

def final_result_view():
    st.header("Final Result")
    st.info("Final result view is being updated for whole-heart segmentation. "
            "Use the Mask Editor to inspect the segmentation.")

final_result_view_all = final_result_view

def measure_vessel(mask, spacing=(1.5, 1.5, 1.5), n_dense=200, n_rays=64):
    """
    Centerline + per-point min/max/avg diameter for a binary vessel mask.
    mask: (H, W, D) binary. spacing in mm. Returns dict or None.
    """
    from skimage import measure as skmeasure
    from skimage.morphology import skeletonize
    import networkx as nx
    from scipy.interpolate import splprep, splev

    spacing = np.array(spacing, dtype=float)
    mask = mask.astype(bool)
    if mask.sum() < 10:
        return None


    from scipy.ndimage import gaussian_filter
    mask_f = gaussian_filter(mask.astype(np.float32), sigma=0.8)
    verts, faces, _, _ = skmeasure.marching_cubes(mask_f, level=0.5, spacing=spacing)

    skel_vox = np.argwhere(skeletonize(mask))
    if len(skel_vox) < 3:
        return None
    coords = skel_vox * spacing
    idx_map = {tuple(v): i for i, v in enumerate(skel_vox)}
    G = nx.Graph()
    for i, c in enumerate(coords):
        G.add_node(i, coord=c)
    neighs = [(dz,dy,dx) for dz in (-1,0,1) for dy in (-1,0,1)
              for dx in (-1,0,1) if not (dz==dy==dx==0)]
    for i, v in enumerate(skel_vox):
        for dz,dy,dx in neighs:
            w = (v[0]+dz, v[1]+dy, v[2]+dx)
            j = idx_map.get(w)
            if j is not None:
                G.add_edge(i, j, weight=np.linalg.norm(np.array([dz,dy,dx])*spacing))

    ends = [n for n,dg in G.degree() if dg==1]
    if len(ends) < 2:
        return None
    max_pair, max_d = (ends[0], ends[-1]), 0
    for u in ends:
        L = nx.single_source_dijkstra_path_length(G, u, weight='weight')
        for v,d in L.items():
            if v in ends and d > max_d:
                max_d, max_pair = d, (u,v)
    path = nx.shortest_path(G, max_pair[0], max_pair[1], weight='weight')
    raw_cl = np.array([G.nodes[n]['coord'] for n in path])
    if len(raw_cl) < 4:
        return None

    deltas = np.linalg.norm(np.diff(raw_cl, axis=0), axis=1)
    t = np.hstack(([0], np.cumsum(deltas))); t /= t[-1]
    k = min(3, len(raw_cl)-1)
    tck, _ = splprep(raw_cl.T, u=t, s=spacing.mean()*len(raw_cl)*0.5, k=k)
    cl_dense = np.vstack(splev(np.linspace(0,1,n_dense), tck)).T

    # Orient so point 0 is the inferior (foot-ward) end = aortic root / LV connection.
    # Inferior = larger axis-0 coordinate (matches the -axis0 display flip).
    if cl_dense[0, 0] > cl_dense[-1, 0]:
        cl_dense = cl_dense[::-1]

    def section_min_max(pt_mm, tangent):
        tv = tangent/np.linalg.norm(tangent)
        arb = np.array([1,0,0])
        if abs(np.dot(arb,tv))>0.9: arb = np.array([0,1,0])
        u = np.cross(tv,arb); u/=np.linalg.norm(u)
        v = np.cross(tv,u)
        step = min(spacing)/2
        pt_vox = pt_mm/spacing
        dists=[]
        for th in np.linspace(0,2*np.pi,n_rays,endpoint=False):
            dir_mm = np.cos(th)*u + np.sin(th)*v
            dist=0.0
            while True:
                dist+=step
                s = pt_vox + dist*(dir_mm/spacing)/np.linalg.norm(dir_mm)
                ijk = np.round(s).astype(int)
                if (ijk<0).any() or (ijk>=mask.shape).any() or not mask[ijk[0],ijk[1],ijk[2]]:
                    dists.append(dist); break
        return min(dists), max(dists)

    dmins, dmaxs = [], []
    for i in range(len(cl_dense)):
        if 0<i<len(cl_dense)-1:
            tang = 0.5*(cl_dense[i+1]-cl_dense[i-1])
        elif i==0:
            tang = cl_dense[1]-cl_dense[0]
        else:
            tang = cl_dense[-1]-cl_dense[-2]
        rmin,rmax = section_min_max(cl_dense[i], tang)
        dmins.append(2*rmin); dmaxs.append(2*rmax)
    dmins=np.array(dmins); dmaxs=np.array(dmaxs)

    return {
        "length_mm": float(max_d), "cl_dense": cl_dense,
        "verts": verts, "faces": faces,
        "d_min": dmins, "d_max": dmaxs, "d_avg": 0.5*(dmins+dmaxs),
    }

@st.fragment
def pa_branch_explorer(res):
    import plotly.graph_objects as go
    segs = res['segments']
    # order: MPA first, then RPA, then LPA
    order = ['MPA', 'RPA', 'LPA']
    segs_by_name = {s['name']: s for s in segs}
    sequence = [n for n in order if n in segs_by_name]

    if 'pa_seg_stage' not in st.session_state:
        st.session_state['pa_seg_stage'] = 0
    if 'pa_confirmed' not in st.session_state:
        st.session_state['pa_confirmed'] = {}   # name -> diameter

    stage = st.session_state['pa_seg_stage']
    done = stage >= len(sequence)
    current_name = None if done else sequence[stage]

    col_left, col_right = st.columns([1, 2])

    with col_left:
        # Show already-confirmed measurements
        if st.session_state['pa_confirmed']:
            st.markdown("**Confirmed:**")
            for name in sequence:
                if name in st.session_state['pa_confirmed']:
                    st.metric(name, f"{st.session_state['pa_confirmed'][name]:.1f} mm")

        if not done:
            st.markdown(f"**Measuring: {current_name}**")
            s = segs_by_name[current_name]
            n_pts = len(s['d_avg'])
            point_num = st.slider("Centerline point", 1, n_pts, n_pts // 2,
                                  key=f"pa_point_{current_name}")
            i_point = point_num - 1
            st.metric(f"{current_name} diameter", f"{s['d_avg'][i_point]:.1f} mm")

            if st.button(f"✅ Confirm {current_name}", type="primary",
                         use_container_width=True):
                st.session_state['pa_confirmed'][current_name] = float(s['d_avg'][i_point])
                st.session_state['pa_seg_stage'] += 1
                st.rerun()
        else:
            st.success("All PA segments confirmed ✅")

            ao_val  = st.session_state.get('ao_confirmed', float('nan'))
            mpa_val = st.session_state['pa_confirmed'].get('MPA', float('nan'))
            rpa_val = st.session_state['pa_confirmed'].get('RPA', float('nan'))
            lpa_val = st.session_state['pa_confirmed'].get('LPA', float('nan'))

            df = pd.DataFrame([{
                "patient": st.session_state.get('patient_name', 'unknown'),
                "Ao_diameter_mm":  round(ao_val, 2),
                "MPA_diameter_mm": round(mpa_val, 2),
                "RPA_diameter_mm": round(rpa_val, 2),
                "LPA_diameter_mm": round(lpa_val, 2),
            }])
            csv_bytes = df.to_csv(index=False).encode('utf-8')

            patient = str(st.session_state.get('patient_name', 'patient'))
            # sanitise filename
            safe = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_'
                           for c in patient).strip().replace(' ', '_')

            st.download_button(
                "💾 Download vessel measurements (CSV)",
                data=csv_bytes,
                file_name=f"{safe}_vessel_measurements.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )

    with col_right:
        verts, faces = res['verts'], res['faces']
        fig = go.Figure()
        fig.add_trace(go.Mesh3d(x=verts[:,2], y=verts[:,1], z=-verts[:,0],
                                i=faces[:,0], j=faces[:,1], k=faces[:,2],
                                color='lightgrey', opacity=0.3, name='PA'))
        seg_colors = {'MPA':'#1fb3b3', 'RPA':'#f08c3d', 'LPA':'#8c4fc7'}
        for sg in segs:
            cl = sg['cl']
            is_current = (sg['name'] == current_name)
            fig.add_trace(go.Scatter3d(x=cl[:,2], y=cl[:,1], z=-cl[:,0], mode='markers',
                          marker=dict(size=3 if is_current else 2,
                                      color=seg_colors.get(sg['name'], 'white'),
                                      opacity=1.0 if is_current else 0.35),
                          name=sg['name']))
            
        if not done:
            s = segs_by_name[current_name]
            pt = s['cl'][i_point]
            fig.add_trace(go.Scatter3d(x=[pt[2]], y=[pt[1]], z=[-pt[0]], mode='markers',
                          marker=dict(size=10, color='yellow'), name='Selected'))

        fig.update_layout(scene=dict(aspectmode='data',
                          camera=dict(eye=dict(x=0,y=-2.2,z=2.2), up=dict(x=0,y=0,z=1)),
                          xaxis=dict(visible=False), yaxis=dict(visible=False),
                          zaxis=dict(visible=False)),
                          uirevision='pa_view', height=550,
                          margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key="pa_3d")

def vessel_measurement_view():
    st.header("Vessel Measurements")
    if not st.session_state.get('initialized_all'):
        st.info("Run segmentation first.")
        return

    if 'vessel_stage' not in st.session_state:
        st.session_state['vessel_stage'] = 'ao'

    iso = float(st.session_state.get('iso_spacing', st.session_state.get('pixelspacing', 1.5)))
    spacing = (iso, iso, iso)
    edited = st.session_state['edited_mask']
    gen = st.session_state.get('canvas_gen', 0)

    if st.session_state['vessel_stage'] == 'ao':
        st.subheader("Aorta")
        ao_mask = edited[..., 0, ao_idx].astype(bool)
        if (st.session_state.get('ao_measure') is None
                or st.session_state.get('ao_measure_gen') != gen):
            with st.spinner("Measuring aorta..."):
                st.session_state['ao_measure'] = measure_vessel(ao_mask, spacing=spacing)
                st.session_state['ao_measure_gen'] = gen
        res = st.session_state.get('ao_measure')
        if res is None:
            st.warning("Could not measure the aorta.")
            return
        ao_centerline_explorer(res)

        # store the Ao confirmed value when confirming
        if st.button("✅ Confirm Aorta → Measure PA", type="primary", use_container_width=True):
            # the explorer is a fragment, so read the selected point from its state
            ao_i = st.session_state.get("ao_point", len(res['d_avg'])//2) - 1
            st.session_state['ao_confirmed'] = float(res['d_avg'][ao_i])
            st.session_state['vessel_stage'] = 'pa'
            st.session_state['pa_seg_stage'] = 0
            st.session_state['pa_confirmed'] = {}
            st.rerun()

    else:  # PA stage
        st.subheader("Pulmonary Artery")
        pa_mask = edited[..., 0, pa_idx].astype(bool)
        if (st.session_state.get('pa_measure') is None
                or st.session_state.get('pa_measure_gen') != gen):
            with st.spinner("Measuring PA (branched)..."):
                st.session_state['pa_measure'] = measure_pa_tree(pa_mask, spacing=spacing)
                st.session_state['pa_measure_gen'] = gen
        res = st.session_state.get('pa_measure')
        if res is None:
            st.warning("Could not measure the PA.")
        else:
            pa_branch_explorer(res)

        if st.button("⬅ Back to Aorta", use_container_width=True):
            st.session_state['vessel_stage'] = 'ao'
            st.rerun()


@st.fragment
def ao_centerline_explorer(res):
    import plotly.graph_objects as go
    n_pts = len(res['d_avg'])

    col_left, col_right = st.columns([1, 2])

    with col_left:
        point_num = st.slider("Centerline point", 1, n_pts, 10, key="ao_point")
        i_point = point_num - 1
        st.metric("Point", f"{point_num} / {n_pts}")
        st.metric("Diameter", f"{res['d_avg'][i_point]:.1f} mm")

    with col_right:
        verts, faces, cl = res['verts'], res['faces'], res['cl_dense']
        pt = cl[i_point]
        z_mesh = -verts[:, 0]
        z_cl   = -cl[:, 0]
        pt_z   = -pt[0]

        fig3d = go.Figure()
        fig3d.add_trace(go.Mesh3d(x=verts[:,2], y=verts[:,1], z=z_mesh,
                                  i=faces[:,0], j=faces[:,1], k=faces[:,2],
                                  color='lightgrey', opacity=0.3, name='Aorta'))
        fig3d.add_trace(go.Scatter3d(x=cl[:,2], y=cl[:,1], z=z_cl, mode='markers',
                                            marker=dict(size=3, color='#1f77b4'),
                                            name='Centerline'))
        fig3d.add_trace(go.Scatter3d(x=[pt[2]], y=[pt[1]], z=[pt_z], mode='markers',
                                     marker=dict(size=10, color='yellow'),
                                     name=f'Point {point_num}'))
        fig3d.update_layout(
            scene=dict(aspectmode='data',
                       camera=dict(eye=dict(x=3.2, y=0, z=0), up=dict(x=0,y=0,z=1)),
                        xaxis=dict(visible=False, showgrid=False, showbackground=False),
                        yaxis=dict(visible=False, showgrid=False, showbackground=False),
                        zaxis=dict(visible=False, showgrid=False, showbackground=False),),
            uirevision='ao_view',   # preserve user's camera across reruns
            height=550, margin=dict(l=0,r=0,t=0,b=0), showlegend=False,
        )
        st.plotly_chart(fig3d, use_container_width=True, key="ao_3d")
        
def measure_pa_tree(mask, spacing=(1.5,1.5,1.5), n_dense=300, n_rays=64, cheb_deg=5):
    """
    PA as a tree: shared MPA trunk (root → bifurcation) + two branches
    (bifurcation → each of the two longest leaves). Each segment is
    Chebyshev-smoothed with analytic-tangent ray-fan diameters.
    Returns {verts, faces, segments:[{name, cl, d_avg}, ...]} or None.
    """
    from skimage import measure as skmeasure
    from skimage.morphology import skeletonize
    import networkx as nx
    from numpy.polynomial import Chebyshev
    from scipy.ndimage import gaussian_filter

    spacing = np.array(spacing, float)
    mask = mask.astype(bool)
    if mask.sum() < 20:
        return None

    mask_f = gaussian_filter(mask.astype(np.float32), sigma=1.0)
    verts, faces, *_ = skmeasure.marching_cubes(mask_f, level=0.5, spacing=spacing)

    skel_vox = np.argwhere(skeletonize(mask))
    if len(skel_vox) < 5:
        return None
    coords = skel_vox * spacing
    idx = {tuple(v): i for i,v in enumerate(skel_vox)}
    G = nx.Graph()
    for i,c in enumerate(coords):
        G.add_node(i, coord=c)
    offs = [(dz,dy,dx) for dz in(-1,0,1) for dy in(-1,0,1)
            for dx in(-1,0,1) if not(dz==dy==dx==0)]
    for i,v in enumerate(skel_vox):
        for dz,dy,dx in offs:
            w=(v[0]+dz,v[1]+dy,v[2]+dx); j=idx.get(w)
            if j is not None:
                G.add_edge(i,j,weight=np.linalg.norm(np.array([dz,dy,dx])*spacing))
    if G.number_of_edges() == 0:
        return None
    comp = max(nx.connected_components(G), key=len)
    G = G.subgraph(comp).copy()

    leaves = [n for n,d in G.degree() if d==1]
    if len(leaves) < 2:
        return None
    root = max(leaves, key=lambda n: G.nodes[n]['coord'][0])
    outlets = [n for n in leaves if n != root]

    paths = {o: nx.shortest_path(G, root, o, weight='weight') for o in outlets}
    lengths = {o: nx.shortest_path_length(G, root, o, weight='weight') for o in outlets}
    two = sorted(outlets, key=lambda o: -lengths[o])[:2]
    pA, pB = paths[two[0]], paths[two[1]]

    # bifurcation = last shared node walking from root
    common = 0
    for a,b in zip(pA, pB):
        if a == b: common += 1
        else: break
    bif = common - 1
    trunk_nodes   = pA[:bif+1]
    branchA_nodes = pA[bif:]
    branchB_nodes = pB[bif:]

    def section_radii(pt_mm, tangent):
        t = tangent/np.linalg.norm(tangent)
        arb = np.array([1,0,0])
        if abs(np.dot(arb,t))>0.9: arb = np.array([0,1,0])
        u = np.cross(t,arb); u/=np.linalg.norm(u); v = np.cross(t,u)
        step = min(spacing)/2.0; pv = pt_mm/spacing; radii=[]
        for th in np.linspace(0,2*np.pi,n_rays,endpoint=False):
            dm = np.cos(th)*u + np.sin(th)*v; dist=0.0
            while True:
                dist+=step
                s = pv + dist*(dm/spacing)/np.linalg.norm(dm)
                ijk = np.round(s).astype(int)
                if (ijk<0).any() or (ijk>=mask.shape).any() or not mask[ijk[0],ijk[1],ijk[2]]:
                    radii.append(dist); break
        return np.array(radii)

    def fit_segment(nodes, deg):
        raw = np.array([G.nodes[n]['coord'] for n in nodes])
        if len(raw) < 3:
            return None
        d = min(deg, len(raw)-2)
        deltas = np.linalg.norm(np.diff(raw,axis=0),axis=1)
        t_raw = np.hstack(([0],np.cumsum(deltas)))
        if t_raw[-1] == 0:
            return None
        t_raw /= t_raw[-1]
        chs = [Chebyshev.fit(t_raw, raw[:,k], d, domain=[0,1]) for k in range(3)]
        dch = [c.deriv() for c in chs]
        u = np.linspace(0,1,n_dense)
        cl = np.vstack([chs[k](u) for k in range(3)]).T
        davg = np.array([2*section_radii(cl[i],
                         np.array([dch[k](u[i]) for k in range(3)])).mean()
                         for i in range(n_dense)])
        return {"cl": cl, "d_avg": davg}

    segments = []
    for name, nodes in [("MPA", trunk_nodes), ("RPA", branchA_nodes), ("LPA", branchB_nodes)]:
        seg = fit_segment(nodes, cheb_deg)
        if seg:
            seg["name"] = name
            segments.append(seg)
    if not segments:
        return None
    return {"verts": verts, "faces": faces, "segments": segments}

def resample_to_isotropic(image, in_spacing, target=None, order=3):
    """
    image: (H, W, D) array. in_spacing: (sy, sx, sz) mm matching axes (0,1,2).
    target: isotropic target spacing in mm (default = in-plane spacing).
    order: 1=linear (image), 0=nearest (labels).
    Returns (resampled_image, new_spacing_scalar, zoom_factors).
    """
    from scipy.ndimage import zoom
    sy, sx, sz = in_spacing
    if target is None:
        target = sx                      # use in-plane as the isotropic target
    zoom_factors = (sy/target, sx/target, sz/target)
    out = zoom(image, zoom_factors, order=order)
    return out, target, zoom_factors