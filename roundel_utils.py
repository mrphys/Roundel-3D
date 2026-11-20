from utils import *
import traceback
import torch
from im2sim.models import UNet


def _get_threejs_src():
    """Return the bundled three.js source (works in dev and PyInstaller-frozen)."""
    import sys
    if '_threejs_src_cache' not in globals():
        if getattr(sys, 'frozen', False):
            base = Path(sys._MEIPASS)
        else:
            base = Path(__file__).parent
        globals()['_threejs_src_cache'] = (
            base / "assets" / "three.min.js").read_text(encoding='utf-8')
    return globals()['_threejs_src_cache']


_HEART_HTML = """
<div id="c" style="width:100%;height:420px;background:#000;"></div>
<script>__THREEJS__</script>
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

// A renderer left running in a page that has gone keeps its WebGL
// context, and a browser only allows so many of those at once before it
// starts taking the oldest away. Stop, and hand it back, on the way out.
let alive = true;
renderer.domElement.addEventListener('webglcontextlost', e => {
    e.preventDefault(); alive = false;
    console.warn('[3D] WebGL context lost -- too many live contexts?');
});
window.addEventListener('pagehide', () => {
    alive = false;
    try { renderer.forceContextLoss(); } catch (e) {}
    try { renderer.dispose(); } catch (e) {}
});
(function animate(){
    if (!alive) return;
    requestAnimationFrame(animate); renderer.render(scene,cam);
})();
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
            disabled=st.session_state['disable_upload']
        )

        # Read the DICOMs into the Pipeline as soon as a zip is present, but
        # DON'T segment yet — wait for the Process button.
        if zip_file and not st.session_state['disable_upload']:
            dcms = extract_dicom_from_zip(zip_file)
            if dcms:
                print(f"[DEBUG] DICOM files extracted: {len(dcms)}")
                pipe = Pipeline(dcms)                     # keep the object
                image, sax_df = pipe.image, pipe.sax_df
                st.session_state['_pending_image'] = image
                st.session_state['sax_df'] = sax_df
                st.session_state['reoriented_spacing'] = pipe.reoriented_spacing

                first_dcm = sax_df['dcm'].values[0]
                st.session_state.patient_name = str(first_dcm.PatientName) if hasattr(first_dcm, 'PatientName') and first_dcm.PatientName else 'Anonymised Patient'
                st.session_state.series_date = str(first_dcm.SeriesDate) if hasattr(first_dcm, 'SeriesDate') and first_dcm.SeriesDate else 'Unknown'
                st.session_state.series_description = str(first_dcm.SeriesDescription) if hasattr(first_dcm, 'SeriesDescription') and first_dcm.SeriesDescription else 'Unknown'
                st.session_state.pixelspacing = sax_df.pixelspacing.unique()[0]
                st.session_state.thickness = sax_df.thickness.unique()[0]
                st.session_state['sax_series_uid'] = first_dcm.SeriesInstanceUID
                st.session_state.n_slices = sax_df['slicelocation'].nunique()
                st.session_state.n_phases = sax_df.loc[sax_df['slicelocation'] == sax_df['slicelocation'].values[0]]['triggertime'].nunique()

        if (st.session_state['disable_upload']
                and "initialized_all" not in st.session_state):
            st.info("⬇ The crop box is below the series details — scroll "
                    "down to place it, then press **Segment with this crop**.")

        # Process button: only shown once a series is loaded and not yet processed
        can_process = ('_pending_image' in st.session_state
                       and not st.session_state['disable_upload'])
        if can_process:
            if st.button("▶ Process", type="primary", use_container_width=True):
                # the left-right flip lives with the crop box now, where
                # the views show whether it is needed at all
                st.session_state['_process_image'] = st.session_state['_pending_image']
                st.session_state['disable_upload'] = True
                st.rerun()

    with col2:
        if st.session_state['disable_upload']:
            dicom_data = {
                "Field": ["Patient Name", "Series Date", "Series Description",
                          "Pixel Size", "Slice Thickness", "Number of Images",
                          "Number of Slices", "Number of Phases", "Slice × Phases"],
                "Value": [
                    st.session_state.patient_name,
                    st.session_state.series_date,
                    st.session_state.series_description,
                    f"{st.session_state.pixelspacing} x {st.session_state.pixelspacing} mm",
                    f"{st.session_state.thickness} mm",
                    len(st.session_state['sax_df']),
                    st.session_state.n_slices,
                    st.session_state.n_phases,
                    st.session_state.n_slices * st.session_state.n_phases
                ]
            }
            df_dicom = pd.DataFrame(dicom_data).set_index('Field')
            st.dataframe(df_dicom, use_container_width=True)

    if "initialized_all" not in st.session_state and st.session_state['disable_upload']:
        # Rescale once, then let the crop box be placed on the result. The
        # rescaled volume is kept so the box can be moved again without
        # loading the series a second time.
        if 'iso_image' not in st.session_state:
            with st.spinner(f"Rescaling to {TARGET_ISO} mm isotropic..."):
                st.session_state['iso_image'] = prepare_iso_image(
                    st.session_state['_process_image'])
        st.divider()
        if crop_box_picker():
            with st.spinner("Segmenting..."):
                segment_image(st.session_state['iso_image'])
            with st.spinner("Initialising..."):
                initialize_app()
            st.rerun()

    if "initialized_all" in st.session_state:
        st.success('Segmentation Confirmed! ⭕️')
        crop_summary_and_recrop()


# --------------------------------------------------------------
# The model's grid, and where it sits in the volume
# --------------------------------------------------------------
# The volume is rescaled to TARGET_ISO mm and then a TRAIN_SHAPE box is taken
# out of it. The box used to be taken from the centre, which put the heart
# outside it whenever the field of view was off-centre; now its position is
# yours to choose (crop_box_picker).
TARGET_ISO = 1.5                    # mm, the resolution the model was trained on
TRAIN_SHAPE = (160, 128, 128)       # (H, W, D) = (transverse, coronal, sagittal)

# The two views the box is placed in: axis -> (name, the axis drawn down that
# view, the axis drawn across it). Both are drawn across axis 2, which is why
# that one slider serves them both.
CROP_VIEWS = ((0, "Transverse", 1, 2),
              (1, "Coronal", 0, 2))

# The box still has three axes to place, named as they read on screen:
#   axis 0, fixed in a transverse slice, runs up and down the coronal view
#   axis 1, fixed in a coronal slice, runs up and down the transverse view
#   axis 2 runs across both views
CROP_AXIS_NAMES = {0: "Up–down", 1: "AP", 2: "Left–right"}
# What each end of a box bar means, in place of the offsets nobody reads.
# Up, anterior and the left of the picture are all the low end of their axis.
# Left and right are as the views show them, not the patient's.
CROP_AXIS_ENDS = {0: ("Up", "Down"), 1: ("A", "P"), 2: ("L", "R")}
# left to right under the views: the transverse view's vertical, the axis the
# two views share, then the coronal view's vertical
BOX_SLIDER_ORDER = (1, 2, 0)


def crop_offsets_default(shape, target_shape=TRAIN_SHAPE):
    """The centre crop, written as offsets: what the app used to do when
    there was nothing to choose."""
    return [max(0, (int(shape[ax]) - int(target_shape[ax])) // 2) for ax in range(3)]


def crop_at(volume, target_shape=TRAIN_SHAPE, offsets=None):
    """Take a target_shape box out of a (H, W, D) volume, starting at
    `offsets` along each axis. An axis shorter than the target is
    symmetrically zero-padded instead, exactly as before: there is nothing to
    position on an axis that has to grow. Offsets are clamped, so a box can
    never hang off the end.
    """
    if offsets is None:
        offsets = crop_offsets_default(volume.shape, target_shape)
    out = volume
    for ax, target in enumerate(target_shape):
        current = out.shape[ax]
        if current == target:
            continue
        if current > target:                       # crop this axis, where asked
            start = int(min(max(int(offsets[ax]), 0), current - target))
            sl = [slice(None)] * out.ndim
            sl[ax] = slice(start, start + target)
            out = out[tuple(sl)]
        else:                                       # symmetric zero-pad this axis
            total = target - current
            before = total // 2
            after = total - before
            pad = [(0, 0)] * out.ndim
            pad[ax] = (before, after)
            out = np.pad(out, pad, mode='constant', constant_values=0)
    assert out.shape[:3] == tuple(target_shape), f"got {out.shape}, want {target_shape}"
    return out


def center_crop_or_pad(volume, target_shape=TRAIN_SHAPE):
    """The centred box, for anything that wants the old behaviour."""
    return crop_at(volume, target_shape,
                   crop_offsets_default(volume.shape, target_shape))


def prepare_iso_image(image):
    """Rescale an (H, W, D, T) volume to the fixed TARGET_ISO grid.

    Split out of segment_image so the crop box can be placed on the rescaled
    volume -- the grid the box is measured in -- before anything is segmented.
    """
    # per-axis spacing AFTER reorientation (the slice-thickness axis may have moved)
    sx0, sx1, sx2 = st.session_state.get(
        'reoriented_spacing',
        (float(st.session_state['pixelspacing']),
         float(st.session_state['pixelspacing']),
         float(st.session_state['thickness'])))
    iso_image_list = []
    for t in range(image.shape[-1]):
        img_t = image[..., t].astype(np.float32)            # (H, W, D)
        img_iso, target_sp, zfac = resample_to_isotropic(
            img_t, in_spacing=(sx0, sx1, sx2), target=TARGET_ISO, order=1)
        iso_image_list.append(img_iso)
    out = np.stack(iso_image_list, axis=-1)                 # (H', W', D', T)
    st.session_state['iso_spacing'] = target_sp             # = 1.5, for everyone downstream
    st.session_state['iso_zoom'] = zfac
    st.session_state['iso_shape'] = tuple(int(s) for s in out.shape[:3])
    # a new volume: unmirrored as it came in, and the tick box back to
    # whatever this kind of series wants
    st.session_state.pop('_flip_applied', None)   # never mirrored yet
    st.session_state['flip_lr'] = default_lr_mirror()
    st.session_state['lr_flip_box'] = False
    print("[ISO] rescaled volume:", out.shape, flush=True)
    return out


# How big a crop-box view is drawn, in pixels along its longer side. Small
# enough that both slider rows and both views fit on screen at once.
CROP_VIEW_PX = int(os.environ.get("ROUNDEL_CROP_VIEW_PX", 600))

# The row the crop step is laid out in: a column per view, then one for the
# box bars. The bars' column is a little narrower than a view's and still
# twice the width it had when it sat beside the coronal image.
VIEW_COL_WIDTHS = (5, 5, 4)


def _scroll_to_here():
    """Scroll the page down to this point in it, once.

    Streamlit has no scroll call of its own in the versions this app runs
    on. A components iframe sits in the page where it is written, and
    `window.frameElement` is that iframe as the page itself sees it, so the
    script can scroll its own position into view. Everything is wrapped: if
    the browser will not allow it, the line above the DICOM table is what
    tells the user to scroll."""
    try:
        # st.iframe on newer Streamlit, components.v1.html on older ones --
        # that one is deprecated and warns in the page. Both embed the markup
        # as srcdoc, which keeps the frame on the same origin as the page, so
        # the script below can see it.
        render = getattr(st, "iframe", None)
        if render is None:
            import streamlit.components.v1 as _components
            render = _components.html
    except Exception:
        return
    render(
        """
        <script>
        const scroll_here = () => {
          try {
            const frame = window.frameElement;
            if (!frame) return;
            const page = window.parent;
            const top = page.scrollY + frame.getBoundingClientRect().top - 80;
            page.scrollTo({top: Math.max(0, top), behavior: "smooth"});
          } catch (e) { /* a sandboxed iframe: leave the page alone */ }
        };
        setTimeout(scroll_here, 200);
        </script>
        """, height=1)          # 0 is rejected by st.iframe


def series_plane():
    """Which way the slices were acquired, read off the header: 'axial',
    'coronal' or 'sagittal', or None when the header does not say. The same
    test dicom_utils prints on the way in -- whichever patient axis the
    plane's normal lies along."""
    try:
        iop = np.asarray([float(v) for v in
                          st.session_state['sax_df']['orientation'].iloc[0]],
                         dtype=float)
        normal = np.cross(iop[:3], iop[3:])
        return {0: "sagittal", 1: "coronal",
                2: "axial"}[int(np.argmax(np.abs(normal)))]
    except Exception:
        return None


def default_lr_mirror():
    """Whether a series starts out mirrored. Every one of them does.

    The model wants them that way round -- which is what the pipeline's own
    left-right correction used to do, before the slice direction was
    measured from the stacking and that correction stopped firing. It is not
    only axial series that need it, so it is not asked of the header any
    more. The tick box is a change away from this rather than the mirror
    itself, which is why it starts empty whatever was acquired.
    """
    return True
























def apply_lr_mirror(shape):
    """The left-right tick box, and the mirroring it asks for.

    It belongs here rather than at the upload step because this is the first
    place the volume can be seen, and whether a series needs mirroring is not
    something anyone can tell from a file name. Flipping the rescaled volume
    is the same as flipping the series before rescaling -- scipy's zoom lines
    the ends of an axis up, so a flip either side of it gives the same
    samples -- so nothing is read or resampled again.
    """
    # The box is a change away from how the views are already drawn, not the
    # mirror itself, so it starts empty. Seeded from what is applied rather
    # than from a constant, so it comes back the way it was left if
    # Streamlit drops the widget's value while another page is open.
    st.session_state.setdefault(
        'lr_flip_box',
        bool(default_lr_mirror()) != bool(
            st.session_state.get('flip_lr', default_lr_mirror())))
    st.checkbox(
        "Mirror left–right", key='lr_flip_box',
        help="These views are already mirrored, which is the way round the "
             "model expects them. Tick this if the heart still comes out on "
             "the wrong side. The box moves with the image, and the masks "
             "are written back the way the series came in.")

    # what is actually applied to the views, and what the export undoes
    wanted = bool(default_lr_mirror()) != bool(st.session_state['lr_flip_box'])
    st.session_state['flip_lr'] = wanted
    first = '_flip_applied' not in st.session_state      # a volume just read
    if wanted == bool(st.session_state.get('_flip_applied', False)):
        return
    for key in ('iso_image', '_process_image'):
        vol = st.session_state.get(key)
        if vol is not None:
            st.session_state[key] = np.ascontiguousarray(vol[:, :, ::-1, :])
    # The box follows the image, so it stays over the same anatomy. Not when
    # the volume is being mirrored on arrival, though: there is nothing to
    # keep it over yet, and reflecting the centre it starts at would leave
    # it a voxel off centre for no reason.
    room = int(shape[2]) - int(TRAIN_SHAPE[2])
    if room > 0 and not first:
        off = int(st.session_state.get('crop_off_2', room // 2))
        st.session_state['crop_off_2'] = int(min(max(room - off, 0), room))
    st.session_state.pop('_crop_base_cache', None)    # every slice is stale
    st.session_state['_flip_applied'] = wanted
    print(f"[LR] views mirrored: {wanted}", flush=True)


def _crop_frame(shape):
    """The frame both views are drawn in, in voxels: as tall as the taller of
    the two and as wide as the axis they share.

    One frame at one scale is what lets the box line up between the views --
    the left-right slider moves it the same distance in both pictures. Sizing
    the frame from the two views rather than from the whole volume means
    nothing is padded sideways, and only the shorter view gets a band above
    and below."""
    rows = max(int(shape[r]) for _a, _l, r, _c in CROP_VIEWS)
    cols = max(int(shape[c]) for _a, _l, _r, c in CROP_VIEWS)
    return rows, cols


def _crop_base_image(vol, axis, idx, frame=None, px=None):
    """One slice of the volume, windowed, centred in the shared frame and
    scaled to fit `px`, kept in a session cache.

    Returns the image, the scale it was drawn at and where the slice sits in
    the frame -- the box is drawn in those same coordinates.

    Moving a slider should only redraw the box, never re-render the picture:
    on a volume this size, re-rendering every slice on every drag is what
    makes the sliders feel stuck."""
    cache = st.session_state.setdefault('_crop_base_cache', {})
    n = int(vol.shape[axis])
    idx = int(min(max(int(idx), 0), n - 1))
    frame = tuple(frame or _crop_frame(vol.shape))
    px = int(px or CROP_VIEW_PX)
    key = (st.session_state.get('sax_series_uid', ''), int(axis), idx, frame, px)
    hit = cache.get(key)
    if hit is None:
        sl = [slice(None)] * 3
        sl[axis] = idx
        img = np.asarray(vol[tuple(sl)], dtype=np.float32)
        lo, hi = np.percentile(img, (1, 99))      # window on the slice shown
        if float(hi - lo) < 1e-6:                 # a flat slice
            lo, hi = float(img.min()), float(img.min()) + 1.0
        img = np.clip((img - lo) / float(hi - lo), 0.0, 1.0)
        rows, cols = img.shape
        pad_r, pad_c = (frame[0] - rows) // 2, (frame[1] - cols) // 2
        sheet = np.zeros(frame, np.float32)
        sheet[pad_r:pad_r + rows, pad_c:pad_c + cols] = img
        im = Image.fromarray(np.repeat((sheet * 255).astype(np.uint8)[..., None],
                                       3, axis=-1))
        scale = px / float(max(frame))
        im = im.resize((max(1, int(round(frame[1] * scale))),
                        max(1, int(round(frame[0] * scale)))),
                       Image.BILINEAR if scale < 1 else Image.NEAREST)
        if len(cache) > 400:                      # a long session: start again
            cache.clear()
        hit = cache[key] = (im, scale, pad_r, pad_c)
    return hit


def _crop_view_image(vol, offsets, axis, rows_ax, cols_ax, idx,
                     target_shape=TRAIN_SHAPE, frame=None, px=None):
    """The slice at `idx` in one plane, with the crop box drawn over it. The
    slice and the box move independently: panning changes `idx` only."""
    from PIL import ImageDraw as _ImageDraw
    base, scale, pad_r, pad_c = _crop_base_image(vol, axis, idx, frame, px)
    im = base.copy()
    r0 = int(round((pad_r + int(offsets[rows_ax])) * scale))
    r1 = int(round((pad_r + int(offsets[rows_ax]) + int(target_shape[rows_ax])) * scale))
    c0 = int(round((pad_c + int(offsets[cols_ax])) * scale))
    c1 = int(round((pad_c + int(offsets[cols_ax]) + int(target_shape[cols_ax])) * scale))
    draw = _ImageDraw.Draw(im)
    # Columns run across the image, rows down it. On an axis the box is
    # wider than, one side of it falls outside the picture, and PIL draws
    # nothing where there is no canvas -- which is why a box with no bar
    # came out with three sides. Hold the rectangle inside the picture
    # instead: the side that ran off is drawn on the edge it ran off.
    w, h = im.size
    lw = 2
    c0v, c1v = max(0, min(c0, w - lw)), max(0, min(c1 - 1, w - 1))
    r0v, r1v = max(0, min(r0, h - lw)), max(0, min(r1 - 1, h - 1))
    draw.rectangle([min(c0v, c1v), min(r0v, r1v),
                    max(c0v, c1v), max(r0v, r1v)],
                   outline=(255, 205, 0), width=lw)
    return im


def _show_centred(holder, im, caption=None):
    """Put a fixed-size image in the middle of its column. st.image draws it
    hard against the left edge unless it is stretched to the column, and
    stretching is what made these too tall to use.

    `holder` is whatever the picture is written into -- a column, or `st`
    itself. It used to be an st.empty(), which is what made the views
    flicker: a placeholder is cleared on the way to being refilled."""
    import base64
    import io as _io
    buf = _io.BytesIO()
    im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    cap = (f"<div style='text-align:center;font-size:0.8em;opacity:0.7'>{caption}</div>"
           if caption else "")
    # height as well as width: fixing both means a redraw swaps one picture
    # for another the same size instead of reflowing the column around it
    holder.markdown(
        f"<div style='text-align:center'>"
        f"<img src='data:image/png;base64,{b64}' "
        f"width='{im.width}' height='{im.height}'>{cap}</div>",
        unsafe_allow_html=True)


def crop_box_text(offsets, shape=None):
    """One line per axis: where the box starts and ends."""
    shape = shape or st.session_state.get('iso_shape')
    out = []
    for ax in range(3):
        a = int(offsets[ax])
        b = a + int(TRAIN_SHAPE[ax])
        extent = f" of {int(shape[ax])}" if shape else ""
        fixed = ("  (fixed: the box covers this axis)"
                 if shape and int(shape[ax]) <= int(TRAIN_SHAPE[ax]) else "")
        out.append(
            f"{CROP_AXIS_NAMES[ax]:<11} axis {ax}: [{a}:{b}]{extent}{fixed}")
    return "\n".join(out)


@st.fragment
def crop_box_controls():
    """The two views, their sliders and the confirm button.

    All of it is one fragment: dragging a slider redraws this block alone
    instead of rerunning the page, which is what made the sliders feel stuck.
    The button is in here too, so that a click is handled in the same rerun
    scope as the sliders -- split across the two, the first click on it went
    to re-registering the widget instead of to the handler.

    A slice slider sits above each view and pans through the volume without
    moving anything. The box bars stand in a column of their own beside the
    views, reading down AP, left-right, up-down. An axis the box already
    covers has no bar at all, since there is nothing it could do.
    """
    vol = st.session_state['iso_image'][..., 0]
    shape = tuple(int(s) for s in vol.shape[:3])
    frame = _crop_frame(shape)
    # seeded from what is stored, so an axis without a slider of its own
    # keeps its value instead of being reset to zero
    offs = [int(o) for o in st.session_state.get('crop_offsets', [0, 0, 0])]
    slices = [int(s) for s in st.session_state.get('crop_slices', [0, 0, 0])]

    def pan_slider(ax, label):
        """Which slice this view shows. Moves nothing but the picture."""
        key = f"crop_slice_{ax}"
        st.session_state[key] = int(min(max(
            int(st.session_state.get(key, shape[ax] // 2)), 0), shape[ax] - 1))
        return int(st.slider(f"{label} slice", 0, shape[ax] - 1, key=key,
                             help="Pans through the volume. The box does not move."))

    def box_slider(ax):
        """Where the box starts on one axis, or nothing at all if the box
        already covers that axis. The column it would have sat in is still
        laid out, so the other bars keep their places."""
        name = CROP_AXIS_NAMES[ax]
        # "AP" is an acronym and stays as it is; the other two read better
        # in lower case in the middle of a label
        disp = name if name.isupper() else name.lower()
        room = shape[ax] - int(TRAIN_SHAPE[ax])
        key = f"crop_off_{ax}"
        if room <= 0:
            st.session_state[key] = 0
            return 0
        st.session_state[key] = int(min(max(
            int(st.session_state.get(key, room // 2)), 0), room))
        # a select_slider, only so that the ends can read as directions
        # instead of as offsets: a plain slider always prints its own range
        near, far = CROP_AXIS_ENDS[ax]
        return int(st.select_slider(
            f"Box {disp}", options=list(range(int(room) + 1)), key=key,
            format_func=lambda v: (near if v == 0 else
                                   far if v == int(room) else str(int(v))),
            help=f"Moves the box along axis {ax}: {near} to {far}"
                 + (" as the views show them" if ax == 2 else "")
                 + f". {shape[ax]} voxels on this axis, the box takes "
                   f"{TRAIN_SHAPE[ax]}."))

    # One row: a view, a view, then the box bars stacked beside them. The
    # bars column is filled first -- a column renders where it sits, not in
    # the order it is written into -- so every offset is known by the time a
    # view is drawn and the picture goes straight into the page. Drawing it
    # into a placeholder instead is what made the views flicker: a
    # placeholder is cleared on the way to being refilled, and the one filled
    # last stayed blank long enough to see.
    cols = st.columns(VIEW_COL_WIDTHS, vertical_alignment="center")

    with cols[-1]:                    # the tick box, then the bars
        # before the bars, so that mirroring can move the box it sets: a
        # widget's value cannot be written once the widget has been drawn
        apply_lr_mirror(shape)
        for ax in BOX_SLIDER_ORDER:
            offs[ax] = box_slider(ax)

    vol = st.session_state['iso_image'][..., 0]      # mirrored just above?
    for col, (ax, label, r_ax, c_ax) in zip(cols, CROP_VIEWS):
        with col:
            slices[ax] = pan_slider(ax, label)
            # at its own size and centred, not stretched to the column: the
            # point of drawing it this size is that the bars stay on screen
            # beside it
            _show_centred(
                st,
                _crop_view_image(vol, offs, ax, r_ax, c_ax, slices[ax],
                                 frame=frame),
                caption=f"{label.lower()} slice {slices[ax]}")

    st.session_state['crop_offsets'] = offs
    st.session_state['crop_slices'] = slices

    if st.button("✅ Segment with this crop", type="primary",
                 use_container_width=True, key="crop_confirm_btn"):
        st.session_state['crop_confirmed'] = True
        st.rerun()          # the whole page: the segmentation runs outside here


def crop_box_picker():
    """Place the model's box inside the rescaled volume.

    Returns True once the box is confirmed, which is when the segmentation
    runs. The box starts where the centre crop would have been, so accepting
    it unchanged reproduces the old behaviour.
    """
    shape = tuple(int(s) for s in st.session_state['iso_image'].shape[:3])
    if 'crop_offsets' not in st.session_state:
        offs = crop_offsets_default(shape)
        st.session_state['crop_offsets'] = offs
        st.session_state['crop_slices'] = [
            int(min(max(int(offs[ax]) + int(TRAIN_SHAPE[ax]) // 2, 0), shape[ax] - 1))
            for ax in range(3)]
        for ax in range(3):      # what the sliders open on
            st.session_state.setdefault(f"crop_off_{ax}", int(offs[ax]))
            st.session_state.setdefault(
                f"crop_slice_{ax}", int(st.session_state['crop_slices'][ax]))
        # a box to place on a volume nobody has seen yet: show it to them
        st.session_state.pop('crop_scrolled', None)

    if st.session_state.pop('crop_confirmed', False):
        return True

    if not st.session_state.get('crop_scrolled'):
        # once, and from out here rather than from inside the fragment:
        # dragging a slider must not drag the page about as well
        st.session_state['crop_scrolled'] = True
        _scroll_to_here()

    st.subheader("Position the crop box")
    st.caption(
        f"The model works on a fixed {' × '.join(str(s) for s in TRAIN_SHAPE)} "
        f"box at {TARGET_ISO} mm; the rescaled volume is "
        f"{' × '.join(str(s) for s in shape)}. The slider above each view pans "
        "through the volume without moving the box; the bars to the right "
        "move the box itself. An axis the box already covers has no bar. "
        "The box starts where the centre crop would have been. Mirror the "
        "views here if the heart is on the wrong side.")
    crop_box_controls()
    return False


# Everything a new crop invalidates goes; the image, the models and the
# DICOM header stay, so moving the box does not mean uploading again.
_KEEP_ON_RECROP = (
    'model', 'corrector_model_15', 'iso_image', 'iso_shape', 'iso_spacing',
    'iso_zoom', '_pending_image', '_process_image', 'disable_upload',
    'flip_lr', 'lr_flip_box', '_flip_applied', 'crop_offsets', 'crop_slices',
    'sax_df',
    'reoriented_spacing',
    'patient_name',
    'series_date', 'series_description', 'pixelspacing', 'thickness',
    'sax_series_uid', 'n_slices', 'n_phases', 'current_view', 'N',
)


def reset_for_new_crop():
    """Back to the crop step: the segmentation and every edit made on the old
    box are dropped, cached masks included -- they belong to a grid that is
    about to change."""
    for key in ('cache_mask_path', 'cache_config_path'):
        path = st.session_state.get(key)
        if path:
            try:
                os.remove(path)
            except OSError:
                pass
    for k in list(st.session_state.keys()):
        if k not in _KEEP_ON_RECROP and not str(k).startswith(('crop_off_',
                                                               'crop_slice_')):
            del st.session_state[k]


def crop_summary_and_recrop():
    """Where the box ended up, and a way back to move it."""
    offs = st.session_state.get('crop_offsets')
    if offs is None:
        return
    with st.expander("Crop box"):
        st.code(crop_box_text(offs), language=None)
        if st.button("⟲ Move the crop box and segment again",
                     use_container_width=True):
            reset_for_new_crop()
            st.rerun()
        st.caption("The series stays loaded. The segmentation and any edits "
                   "made on the current box are discarded.")


def segment_image(image):
    """`image` is the rescaled volume from prepare_iso_image. The box is cut
    out of it at st.session_state['crop_offsets'] -- where the user put it --
    and that cropped volume is what the model and every step after it use."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    offs = st.session_state.get('crop_offsets')
    if offs is None:                       # no picker ran: the centred box
        offs = crop_offsets_default(image.shape[:3])
    offs = [int(o) for o in offs]
    st.session_state['crop_offsets'] = offs
    cropped_list = []
    for t in range(image.shape[-1]):
        cropped_list.append(crop_at(image[..., t], TRAIN_SHAPE, offs))
    image = np.stack(cropped_list, axis=-1)                 # (160, 128, 128, T)
    print(f"[CROP] box {TRAIN_SHAPE} at offsets {tuple(offs)} "
          f"-> {image.shape}", flush=True)

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

        # No secondary heart-crop: the volume is already the fixed (112,160,96)
        # rescaled+cropped grid, and the whole session works on it directly.
        # Use a full-extent crop box so all downstream crop_box logic still holds.
        H_full, W_full = raw_image.shape[0], raw_image.shape[1]
        x_min, y_min, x_max, y_max = 0, 0, W_full, H_full
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
    # Keyed, and with no index=. A widget with no key is known by its
    # parameters, so an index read from the state the widget itself writes
    # makes it a different widget the moment it is clicked -- and the click
    # that changed it is dropped with the widget it belonged to, which is
    # why these two took two clicks and the keyed ones did not.
    st.session_state.setdefault('brush_mode', "Paint ✏️")
    action = st.radio("Brush Stroke Selection",
                      options=["Paint ✏️", "Erase ✂️"],
                      horizontal=True, key='brush_mode')

    stroke_width_map = {"thin":6,"medium":20,"thick":40}
    st.session_state.setdefault('stroke_width', "thin")
    stroke_width_sel = st.radio("Stroke Width",
                                options=list(stroke_width_map.keys()),
                                horizontal=True, key='stroke_width')

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
# Turning a canvas stroke back into labels. A pixel counts as paint only if
# it carries at least this much of the stroke's own opacity -- the canvas
# antialiases its edges, and a faint pixel's colour is not trustworthy -- and
# only if its colour really is one of the palette's, within this distance in
# RGB (the two closest palette colours are 66 apart).
CANVAS_ALPHA_FRAC = 0.5
CANVAS_COLOR_TOL = 25.0


def canvas_initial_drawing(signature):
    """What to hand the canvas as the drawing it starts from.

    The component reloads the canvas whenever this is not what it was given
    the run before, and empties it in doing so. Handing it nothing on the
    runs in between is not the same as handing it nothing twice: with no
    initial_drawing the wrapper sends its own default, which does not match
    the empty drawing sent on the run the slice changed -- so the next run
    reloads the canvas again and wipes whatever was drawn in the meantime.
    A stroke finished just before a rerun goes with it.

    Hand it the same drawing on every run instead, differing only by a count
    of the times the canvas has been emptied on purpose. Then it reloads
    when the slice, the plane or an edit says to, and at no other time.
    """
    if st.session_state.get('_canvas_empty_sig') != signature:
        st.session_state['_canvas_empty_sig'] = signature
        st.session_state['_canvas_clears'] = \
            st.session_state.get('_canvas_clears', 0) + 1
    return {"version": "4.4.0", "objects": [],
            "clears": st.session_state.get('_canvas_clears', 0)}


def canvas_paint_labels(image_data, colors,
                        alpha_frac=CANVAS_ALPHA_FRAC, tol=CANVAS_COLOR_TOL):
    """Which palette colour each canvas pixel was painted with.

    `image_data` is the canvas RGBA, `colors` the palette as (N, 3). Returns
    an (h, w) array of indices into `colors`, and -1 wherever nothing was
    painted: transparent, too faint to read, or a colour that is not in the
    palette -- a blend of two strokes, or a rim pixel whose colour the round
    trip through premultiplied alpha destroyed.
    """
    data = np.asarray(image_data)
    rgb = data[..., :3].astype(np.float32).reshape(-1, 3)
    alpha = data[..., 3].astype(np.float32).reshape(-1) / 255.0
    colors = np.asarray(colors, dtype=np.float32)

    d = np.linalg.norm(rgb[:, None, :] - colors[None, :, :], axis=-1)
    nearest = np.argmin(d, axis=1)
    peak = float(alpha.max()) if alpha.size else 0.0
    keep = ((alpha > 0) & (alpha >= alpha_frac * peak)
            & (d[np.arange(d.shape[0]), nearest] <= tol))
    return np.where(keep, nearest, -1).reshape(data.shape[:2])


def drop_specks(painted, stroke_width):
    """Drop blobs far too small to have come from the brush.

    A brush of width w lays down about w**2 pixels at a time, so anything
    under an eighth of that was never a mark anyone made."""
    min_px = int(max(1, (int(stroke_width) ** 2) // 8))
    if min_px <= 1 or not painted.any():
        return painted
    from scipy.ndimage import label as _label
    lab, n = _label(painted)
    if n == 0:
        return painted
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return (painted * np.isin(lab, np.flatnonzero(sizes >= min_px))).astype(np.uint8)


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

# How Pipeline reorients a series on the way in. Imported if dicom_utils will
# give them up, since they are its constants, not ours.
try:
    from dicom_utils import MODEL_AXCODES, MODEL_LR_SENSE
except Exception:                                   # pragma: no cover
    MODEL_AXCODES, MODEL_LR_SENSE = ('I', 'P', 'R'), -1


def series_reorientation():
    """What was done to the series' axes on the way in: the nibabel
    orientation transform Pipeline applied, and whether it then mirrored the
    volume left-right.

    Recomputed from the headers rather than remembered, because it is a pure
    function of them -- the same inputs that decided it in the first place.
    It follows whichever dicom_utils is installed beside this file: one that
    measures the slice direction from the stacking lands on MODEL_AXCODES by
    construction and makes no left-right guess, while an older one takes the
    plane's normal for the slice direction and does guess. Returns
    (None, False) for a series with no geometry in its header, which is the
    case Pipeline leaves alone.
    """
    try:
        df = st.session_state['sax_df']
        first = df.iloc[0]
        iop = np.asarray([float(v) for v in first['orientation']], dtype=float)
        row_cos, col_cos = iop[:3], iop[3:]
        normal = np.cross(row_cos, col_cos)
        ps, th = float(first['pixelspacing']), float(first['thickness'])

        # does the installed dicom_utils measure the slice direction?
        step = None
        try:
            import inspect
            import dicom_utils as _du
            if 'slice_vec' in inspect.signature(_du.reorient_volume).parameters:
                step = _du.slice_vector(
                    [df.loc[df['slicelocation'] == loc, 'position'].iloc[0]
                     for loc in df['slicelocation'].unique()[:2]])
        except Exception as exc:
            print(f"[NII] dicom_utils not readable ({exc}); assuming the "
                  "slice direction came from the plane normal", flush=True)

        # the same matrix the series was reoriented with, so the transform
        # below is the one that was actually applied
        aff = np.eye(4)
        aff[:3, 0] = col_cos * ps          # axis 0 is the row index
        aff[:3, 1] = row_cos * ps          # axis 1 is the column index
        aff[:3, 2] = step if step is not None else normal * th
        aff[:3, 3] = np.asarray([float(v) for v in first['position']], dtype=float)
        aff_ras = np.diag([-1., -1., 1., 1.]) @ aff

        transform = nib.orientations.ornt_transform(
            nib.io_orientation(aff_ras),
            nib.orientations.axcodes2ornt(MODEL_AXCODES))

        if step is not None:          # canonical by construction: no guess
            return transform, False
        lr = max([row_cos, col_cos, normal], key=lambda v: abs(v[0]))
        return transform, bool((1 if lr[0] >= 0 else -1) != MODEL_LR_SENSE)
    except Exception as exc:
        print(f"[NII] no header geometry ({exc}); the series was not "
              "reoriented either", flush=True)
        return None, False


def _zoom_back(n_iso, n_src):
    """The step through the TARGET_ISO mm grid that one source voxel is
    worth. scipy.ndimage.zoom lines up the two ends of an axis rather than
    the voxel edges, so coming back is the same arithmetic the other way."""
    return (n_iso - 1) / (n_src - 1) if n_src > 1 and n_iso > 1 else 0.0


def mask_to_series_space(labels, iso_shape, offsets, reoriented_shape,
                         target_shape=TRAIN_SHAPE, flip_lr=False,
                         transform=None, mirrored=False):
    """A label volume on the model's grid, put back on the series' own.

    Undoes, in reverse, everything between the DICOM as it was read and the
    box the model works in: the crop box, the rescale to TARGET_ISO mm, the
    user's left-right flip, the automatic one, and the reorientation.
    Nearest neighbour throughout -- these are labels, and a label halfway
    between two structures is not a thing.
    """
    lab = np.asarray(labels)
    iso_shape = tuple(int(v) for v in iso_shape)
    out_shape = tuple(int(v) for v in reoriented_shape)

    # 4. the crop box: back into the rescaled volume it was cut from
    iso = np.zeros(iso_shape, lab.dtype)
    src, dst = [], []
    for ax in range(3):
        extent, box = iso_shape[ax], int(target_shape[ax])
        if extent >= box:                     # it was cropped, where asked
            start = int(min(max(int(offsets[ax]), 0), extent - box))
            dst.append(slice(start, start + box))
            src.append(slice(0, box))
        else:                                 # it was padded, symmetrically
            before = (box - extent) // 2
            dst.append(slice(0, extent))
            src.append(slice(before, before + extent))
    iso[tuple(dst)] = lab[tuple(src)]

    # 3. the rescale: pick the TARGET_ISO mm voxel each source voxel sits in
    take = [np.clip(np.rint(np.arange(out_shape[ax])
                            * _zoom_back(iso_shape[ax], out_shape[ax])
                            ).astype(int), 0, iso_shape[ax] - 1)
            for ax in range(3)]
    out = iso[np.ix_(*take)]

    # 2. and 1. the two left-right flips, then the reorientation itself
    if flip_lr:
        out = out[:, :, ::-1]
    if mirrored:
        out = out[:, :, ::-1]
    out = np.ascontiguousarray(out)
    if transform is not None:
        # the inverse of a nibabel orientation transform: whichever input
        # axis became output axis i goes back to being axis i, with the same
        # flip. ornt_transform cannot be used for this -- it takes two
        # orientations, and what was kept here is the transform between them.
        tr = np.asarray(transform, dtype=float)
        inverse = np.zeros_like(tr)
        for i in range(tr.shape[0]):
            inverse[int(tr[i, 0])] = (i, tr[i, 1])
        out = nib.orientations.apply_orientation(out, inverse)
    return out


def series_affine():
    """Voxel indices of the series as it was read to patient coordinates, in
    the RAS convention NIfTI wants, so the masks land on the series itself in
    a viewer.

    The slice direction is measured between the first two slices rather than
    taken from the cross product, so its sign follows the order the slices
    were actually stacked in. Falls back to plain spacing, and then to no
    geometry at all, rather than refusing to write a file.
    """
    try:
        df = st.session_state['sax_df']
        iop = np.asarray([float(v) for v in df['orientation'].iloc[0]], dtype=float)
        along_row, along_col = iop[:3], iop[3:]
        ps = float(df['pixelspacing'].iloc[0])
        locs = sorted(df['slicelocation'].unique())
        pos = lambda loc: np.asarray(
            [float(v) for v in df.loc[df['slicelocation'] == loc, 'position'].iloc[0]],
            dtype=float)
        first = pos(locs[0])
        step = (pos(locs[1]) - first if len(locs) > 1 else
                np.cross(along_row, along_col) * float(df['thickness'].iloc[0]))

        # The header rounds its direction cosines to three decimals, which on
        # an oblique acquisition is enough to leave them not quite
        # perpendicular, and the measured step carries that rounding too. ITK,
        # which is what 3D Slicer reads NIfTI with, refuses a direction that
        # is not orthonormal -- "No orthonormal definition found" -- so square
        # them up here. On a real series the correction is ten-thousandths of
        # a millimetre per voxel; on an axis-aligned one it is nothing at all,
        # which is why only oblique series were refused.
        e0 = along_col / np.linalg.norm(along_col)
        e1 = along_row - float(np.dot(along_row, e0)) * e0
        e1 /= np.linalg.norm(e1)
        normal = np.cross(e0, e1)
        gap = float(np.dot(step, normal))      # the slice step along it, signed
        if abs(gap) < 1e-6:                    # a step going nowhere useful
            gap = float(df['thickness'].iloc[0])
            print("[NII] the slice step lies in the plane; falling back to "
                  "the slice thickness", flush=True)

        A = np.eye(4)
        A[:3, 0] = e0 * ps             # axis 0 is the row index: down a column
        A[:3, 1] = e1 * ps             # axis 1 is the column index: along a row
        A[:3, 2] = normal * gap        # axis 2 is the slice
        A[:3, 3] = first
        A[:2, :] *= -1                 # DICOM is LPS, NIfTI is RAS
        if np.isfinite(A).all() and abs(np.linalg.det(A[:3, :3])) > 1e-9:
            return A
        print("[NII] header geometry is degenerate; falling back to spacing",
              flush=True)
    except Exception as exc:
        print(f"[NII] no header geometry ({exc}); falling back to spacing",
              flush=True)
    sp = st.session_state.get('reoriented_spacing')
    if sp is not None and len(sp) == 3:
        return np.diag([float(sp[0]), float(sp[1]), float(sp[2]), 1.0])
    return np.eye(4)


def _nii_to_bytes(arr, affine=None):
    """Serialise a numpy array to gzipped NIfTI bytes (in-memory)."""
    import gzip, io
    if affine is None:
        affine = np.eye(4)
    img = nib.Nifti1Image(np.asarray(arr).astype(np.uint8), affine=affine,
                          dtype='uint8')
    # nibabel writes the sform and leaves the qform unset. ITK-based readers,
    # 3D Slicer among them, would rather have the qform, and a file carrying
    # only one of the two is the kind of thing a reader can refuse. Set both.
    try:
        img.set_qform(img.affine, code='scanner')
        img.set_sform(img.affine, code='scanner')
    except Exception as exc:                 # an affine no quaternion can hold
        print(f"[NII] qform not set ({exc}); the sform alone will have to do",
              flush=True)
    raw = img.to_bytes()                 # uncompressed .nii bytes
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode='wb') as gz:
        gz.write(raw)
    return out.getvalue()

# The surfaces the download writes, and where they sit in the scan.
#
# How finely the label volumes are meshed, as a multiple of the model's own
# 1.5 mm grid, and how much the binary volume is smoothed before the surface
# is found in it. Marching cubes can only put a vertex on an edge of the
# grid it is handed, so the grid is what decides how smooth a surface can
# be: on 1.5 mm voxels the whole heart comes to a few tens of thousands of
# facets. Interpolating the smoothed field onto a finer grid first costs
# memory and seconds rather than accuracy -- it is the same surface,
# triangulated more finely.
#
# Twice is the default: the volume is smoothed over about a voxel before the
# surface is found in it, so there is no detail below a millimetre or so to
# resolve, and a third of a voxel only buys more triangles for the same
# shape -- a megabyte or two against twenty. Raise it with the environment
# variable if a particular structure wants it.
MESH_UPSAMPLE = float(os.environ.get("ROUNDEL_MESH_UPSAMPLE", 2.0))
MESH_SIGMA = float(os.environ.get("ROUNDEL_MESH_SIGMA", 1.0))

# What each structure is called and what colour it is drawn in. The renderer
# keeps its own copy of this; this one also fixes the order the files come
# out in, and gives the .mtl beside an .obj its colours.
MESH_COLORS = {"LV": "#c71f1f", "RV": "#1f59db", "LA": "#f08c3d",
               "RA": "#8c4fc7", "Ao": "#dbc629", "PA": "#1fb3b3"}

# The formats the surfaces can be written in. STL is the one every printer
# and mesher reads; OBJ keeps its vertices shared, carries the normals the
# surface was found with, and takes a colour per structure from a .mtl
# beside it.
MESH_FORMATS = ("STL", "OBJ")


def _shift(d):
    """index -> index + d"""
    m = np.eye(4)
    m[:3, 3] = d
    return m


def _flip_axis(axis, n):
    """index -> n - 1 - index, on one axis"""
    m = np.eye(4)
    m[axis, axis] = -1.0
    m[axis, 3] = float(n) - 1.0
    return m


def editor_to_scan_affine(editor_shape):
    """Where a point in the editor's volume sits in the scan, in millimetres.

    Everything between the two is affine -- the editor's crop, the model's
    box, the rescale to TARGET_ISO mm, the two left-right flips and the
    reorientation -- so the whole chain multiplies out into one 4x4. A
    surface can be carried across with it exactly, without the nearest
    neighbour gather the label volumes go through on their way back.

    The millimetres are LPS, which is what DICOM measures in and what a
    viewer assumes a mesh file is in unless its header says otherwise.

    Returns (affine, where), where `where` is 'scan' for the scan's own
    millimetres and 'box' when the header had no geometry to go back to and
    the surfaces are left on the model's 1.5 mm grid.
    """
    box = np.eye(4) * float(TARGET_ISO)
    box[3, 3] = 1.0
    try:
        crop_box = st.session_state.preprocessed['crop_box']
        iso_shape = tuple(int(v) for v in st.session_state['iso_shape'])
        offsets = st.session_state['crop_offsets']
        series_grid = tuple(
            int(v) for v in np.asarray(st.session_state['_pending_image']).shape[:3])
    except Exception as exc:
        print(f"[MESH] nothing to map the surfaces onto ({exc}); leaving "
              f"them on the model's {TARGET_ISO} mm grid", flush=True)
        return box, 'box'

    # 1. the editor's crop, which takes rows and columns only
    x_min, y_min, _x_max, _y_max = crop_box
    m = _shift((int(y_min), int(x_min), 0))

    # 2. the model's box, into the rescaled volume it was cut from -- or out
    #    of it, on an axis too short to fill the box, which was padded
    d = []
    for ax in range(3):
        extent, want = iso_shape[ax], int(TRAIN_SHAPE[ax])
        if extent >= want:
            d.append(int(min(max(int(offsets[ax]), 0), extent - want)))
        else:
            d.append(-((want - extent) // 2))
    m = _shift(d) @ m

    # 3. the rescale to TARGET_ISO mm. scipy lines the two ends of an axis
    #    up rather than the voxel edges, so coming back is the same
    #    arithmetic the other way round
    s = np.eye(4)
    for ax in range(3):
        n_iso, n_src = iso_shape[ax], series_grid[ax]
        s[ax, ax] = ((n_src - 1) / (n_iso - 1)) if n_iso > 1 and n_src > 1 else 0.0
    m = s @ m

    # 4. the two left-right flips, on the axis they were done on
    transform, mirrored = series_reorientation()
    for flipped in (bool(st.session_state.get('flip_lr', default_lr_mirror())),
                    bool(mirrored)):
        if flipped:
            m = _flip_axis(2, series_grid[2]) @ m

    # 5. the reorientation, undone. nibabel's apply_orientation sends input
    #    axis i to output axis ornt[i, 0], flipping along i first, so the
    #    same ornt written as a matrix is the map these vertices need.
    if transform is not None:
        tr = np.asarray(transform, dtype=float)
        inverse = np.zeros_like(tr)
        for i in range(tr.shape[0]):
            inverse[int(tr[i, 0])] = (i, tr[i, 1])
        p = np.zeros((4, 4))
        p[3, 3] = 1.0
        for i in range(inverse.shape[0]):
            j, f = int(inverse[i, 0]), float(inverse[i, 1])
            p[j, i] = f
            p[j, 3] = 0.0 if f > 0 else float(series_grid[i]) - 1.0
        m = p @ m

    # 6. the scan's own millimetres. series_affine is RAS, which is what
    #    NIfTI wants; a mesh file is read as LPS, which is the other way
    #    round on the first two axes.
    ras = series_affine()
    if not np.isfinite(ras).all() or abs(np.linalg.det(ras[:3, :3])) < 1e-9:
        print("[MESH] no usable scan geometry; leaving the surfaces on the "
              f"model's {TARGET_ISO} mm grid", flush=True)
        return box, 'box'
    return np.diag([-1.0, -1.0, 1.0, 1.0]) @ ras @ m, 'scan'


def build_export_meshes(mask_onehot, upsample=None, sigma=None, frame=0,
                        budget_mb=600.0):
    """A surface per structure, meshed finer than the page draws them.

    Each structure is cut down to its own corner of the volume first, so the
    finer grid is only ever paid for where there is something to mesh. The
    smoothing is done before that interpolation and with background outside
    the cut, so the surface does not know it was cut out.

    Vertices come back in the editor volume's own index coordinates, whole
    numbers on voxel centres, which is what editor_to_scan_affine maps.
    """
    from skimage import measure
    from scipy.ndimage import gaussian_filter, zoom

    up = float(MESH_UPSAMPLE if upsample is None else upsample)
    sg = float(MESH_SIGMA if sigma is None else sigma)
    mask = np.asarray(mask_onehot)[..., int(frame), :]
    pad = int(np.ceil(4.0 * sg)) + 2
    names = {lv_idx: "LV", rv_idx: "RV", la_idx: "LA",
             ra_idx: "RA", ao_idx: "Ao", pa_idx: "PA"}

    meshes = []
    for ch, name in names.items():
        vol = mask[..., ch]
        if not vol.any():
            continue
        here = np.argwhere(vol)
        lo = np.maximum(here.min(0) - pad, 0)
        hi = np.minimum(here.max(0) + pad + 1, vol.shape)
        sub = vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
        sub = gaussian_filter(sub, sigma=sg, mode='constant', cval=0.0)

        # as fine as will fit: a structure in a big box gets less of a lift
        # than one in a small box, rather than the whole export falling over
        f = up
        while f > 1.0 and sub.size * f ** 3 * 4 > budget_mb * 1e6:
            f -= 0.5
        if f > 1.0:
            sub = zoom(sub, f, order=3)
        step = tuple(((hi[a] - lo[a] - 1) / (sub.shape[a] - 1))
                     if sub.shape[a] > 1 else 1.0 for a in range(3))
        try:
            verts, faces, normals, _v = measure.marching_cubes(
                sub, level=0.5, spacing=step, step_size=1)
        except (ValueError, RuntimeError) as exc:
            print(f"[MESH] {name}: no surface ({exc})", flush=True)
            continue
        # skimage returns vertex normals pointing out of the structure and a
        # winding that says the opposite. Turn the winding round so a
        # facet's own normal agrees with them, and nothing is inside out.
        faces = faces[:, ::-1]
        verts = verts + lo                      # back into the whole volume
        meshes.append({"name": name, "verts": verts.astype(np.float64),
                       "faces": faces.astype(np.int64),
                       "normals": np.asarray(normals, dtype=np.float64),
                       "color": MESH_COLORS.get(name, "#cccccc"), "step": f})
        print(f"[MESH] {name}: {len(faces)} triangles, {len(verts)} nodes, "
              f"at {f:g}x ({TARGET_ISO / f:.2f} mm)", flush=True)
    return meshes


def place_mesh(mesh, affine):
    """One surface put where the affine says, ready to write.

    Vertices go through the matrix; normals go through the inverse
    transpose, which is what keeps them perpendicular to a surface that has
    been scaled unevenly. A matrix that mirrors turns a triangle's winding
    inside out as well, so the winding is turned back.
    """
    a = np.asarray(affine, dtype=float)
    verts = (a @ np.c_[mesh["verts"], np.ones(len(mesh["verts"]))].T).T[:, :3]
    faces = np.asarray(mesh["faces"])
    if np.linalg.det(a[:3, :3]) < 0:
        faces = faces[:, ::-1]
    normals = mesh.get("normals")
    if normals is not None and len(normals):
        normals = (np.linalg.inv(a[:3, :3]).T @ np.asarray(normals).T).T
        ln = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(normals, ln, out=np.zeros_like(normals), where=ln > 0)
    return verts, faces, normals


def _stl_bytes(verts, faces, name="surface", space="LPS"):
    """One surface as a binary STL: 84 bytes of preamble and 50 a triangle,
    written out of numpy rather than a triangle at a time.

    The 80-byte header says which way round the coordinates are. Readers
    that look for it, 3D Slicer among them, take it at its word; the rest
    assume LPS, which is what it says.
    """
    v = np.asarray(verts, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int64)
    tri = v[f]                                           # (T, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 0)
    rec = np.zeros(len(f), dtype=np.dtype(
        [('n', '<f4', (3,)), ('v', '<f4', (3, 3)), ('attr', '<u2')]))
    rec['n'] = n.astype(np.float32)
    rec['v'] = tri.astype(np.float32)
    head = f"Roundel 3D {name}. SPACE={space}".encode('ascii', 'replace')
    return b"".join([(head + b" " * 80)[:80],
                     np.uint32(len(f)).tobytes(), rec.tobytes()])


def _obj_bytes(verts, faces, normals=None, name="surface", mtl=None,
               space="LPS"):
    """One surface as a Wavefront OBJ.

    Vertices are shared rather than repeated once per triangle the way STL
    repeats them, and the normals are the ones the surface was found with --
    read off the smoothed volume, so smoother than anything a viewer can
    work out from the triangles afterwards. The header names the coordinate
    system in the comment 3D Slicer looks for.
    """
    import io
    buf = io.StringIO()
    buf.write(f"# Roundel 3D {name}. SPACE={space}\n")
    if mtl:
        buf.write(f"mtllib {mtl}\n")
    buf.write(f"o {name}\n")
    np.savetxt(buf, np.asarray(verts, dtype=float), fmt="v %.4f %.4f %.4f")
    has_n = normals is not None and len(normals) == len(verts)
    if has_n:
        np.savetxt(buf, np.asarray(normals, dtype=float),
                   fmt="vn %.4f %.4f %.4f")
    if mtl:
        buf.write(f"usemtl {name}\n")
    f1 = np.asarray(faces, dtype=np.int64) + 1          # OBJ counts from one
    if has_n:
        np.savetxt(buf, np.c_[f1[:, 0], f1[:, 0], f1[:, 1], f1[:, 1],
                              f1[:, 2], f1[:, 2]],
                   fmt="f %d//%d %d//%d %d//%d")
    else:
        np.savetxt(buf, f1, fmt="f %d %d %d")
    return buf.getvalue().encode("ascii")


def _mtl_bytes(meshes):
    """The colours the structures are drawn in, for the .obj files beside
    it. Written once for all of them, so the names match the files."""
    out = ["# Roundel 3D surface colours"]
    for m in meshes:
        h = str(m.get("color", "#cccccc")).lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        out += [f"newmtl {m['name']}",
                f"Kd {r:.3f} {g:.3f} {b:.3f}",
                "Ka 0.000 0.000 0.000", "Ks 0.000 0.000 0.000",
                "d 1.0", "illum 1", ""]
    return "\n".join(out).encode("ascii")


def _surface_geometry(edited_mask, stamp):
    """The meshes themselves, kept between formats: meshing is the part that
    takes seconds, and writing the same surfaces out as OBJ instead of STL
    is no reason to find them again."""
    if st.session_state.get('_mesh_geom_stamp') != stamp:
        st.session_state['_mesh_geom'] = build_export_meshes(edited_mask)
        st.session_state['_mesh_geom_stamp'] = stamp
    return st.session_state['_mesh_geom']


def _surfaces_zip(edited_mask, safe, fmt="STL", stamp=None):
    """Every structure as its own file, in the scan's millimetres, zipped.

    Seconds of work rather than milliseconds, which is why it is asked for
    rather than kept up to date behind every click.
    """
    import io
    import zipfile
    fmt = str(fmt).upper()
    meshes = _surface_geometry(edited_mask, stamp)
    affine, where = editor_to_scan_affine(np.asarray(edited_mask).shape[:3])
    space = "LPS" if where == 'scan' else "LPS-UNVERIFIED"
    folder = f"{safe}_surfaces"
    mtl_name = f"{safe}_colours.mtl"

    buf = io.BytesIO()
    triangles = 0
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for m in meshes:
            verts, faces, normals = place_mesh(m, affine)
            if fmt == "OBJ":
                zf.writestr(f"{folder}/{m['name']}.obj",
                            _obj_bytes(verts, faces, normals, m["name"],
                                       mtl=mtl_name, space=space))
            else:
                zf.writestr(f"{folder}/{m['name']}.stl",
                            _stl_bytes(verts, faces, m["name"], space))
            triangles += len(faces)
        if fmt == "OBJ" and meshes:
            zf.writestr(f"{folder}/{mtl_name}", _mtl_bytes(meshes))
    print(f"[MESH] {len(meshes)} surfaces, {triangles} triangles, "
          f"{len(buf.getvalue()) / 1e6:.1f} MB of zipped {fmt}, in the "
          f"{'scan' if where == 'scan' else 'model box'}", flush=True)
    return buf.getvalue(), triangles, where


def _masks_zip(edited_mask, N, safe):
    """The download's bytes: every mask on the series' own grid, zipped.

    The better part of a second's work -- a one-hot volume the size of the
    whole box, the inverse of everything done to the series on the way in,
    and then gzip once per structure. A download button wants its bytes
    before anyone clicks it, so this used to run on every rerun of the
    editor, which is what put a wait behind every click of Previous and
    Next. Its caller keeps the result and comes back here only when an edit
    has moved something.

    Returns the zip and how many voxels carry a label, so the caller can say
    when there is nothing in it whether or not it built the file this run.
    """
    import io, zipfile
    raw_shape = st.session_state.raw['mask'].shape[:4]
    x_min, y_min, x_max, y_max = st.session_state.preprocessed['crop_box']
    full_onehot = np.zeros(raw_shape + (N,), dtype=np.uint8)
    full_onehot[y_min:y_max, x_min:x_max, :, :, :] = edited_mask
    full_labels = np.argmax(full_onehot, axis=-1).astype(np.uint8)

    # Back onto the series that was loaded: undo the crop box, the rescale to
    # TARGET_ISO mm, both left-right flips and the reorientation, then write
    # the result with the series' own geometry. No copy of the image is
    # needed -- the masks land on the original DICOM.
    reoriented = st.session_state.get('_pending_image')   # before the user's flip
    iso_shape = st.session_state.get('iso_shape')
    offs = st.session_state.get('crop_offsets')
    in_series_space = (reoriented is not None and iso_shape is not None
                       and offs is not None)
    affine = np.eye(4)
    if in_series_space:
        transform, mirrored = series_reorientation()
        full_labels = np.stack(
            [mask_to_series_space(
                full_labels[..., t], iso_shape, offs,
                np.asarray(reoriented).shape[:3],
                flip_lr=bool(st.session_state.get('flip_lr',
                                                  default_lr_mirror())),
                transform=transform, mirrored=mirrored)
             for t in range(full_labels.shape[-1])], axis=-1)
        affine = series_affine()
        print(f"[NII] masks returned from the {TRAIN_SHAPE} box to the "
              f"series grid {full_labels.shape[:3]}", flush=True)

    labelled = int((full_labels > 0).sum())
    print(f"[NII] {labelled} labelled voxels to write", flush=True)

    # a single frame is a plain 3D volume, which is what a viewer wants
    flat = (lambda a: a[..., 0] if a.shape[-1] == 1 else a)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{safe}_masks/combined.nii.gz",
                    _nii_to_bytes(flat(full_labels), affine))
        struct_names = {lv_idx:"LV", rv_idx:"RV", la_idx:"LA",
                        ra_idx:"RA", ao_idx:"Ao", pa_idx:"PA"}
        for ch, nm in struct_names.items():
            binary = (full_labels == ch).astype(np.uint8)
            if binary.sum() == 0:
                continue
            zf.writestr(f"{safe}_masks/{nm}.nii.gz",
                        _nii_to_bytes(flat(binary), affine))
    return zip_buf.getvalue(), labelled



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
            # the scan on its own: what the masks are covering is the whole
            # reason to look at it in Viewer
            st.image(image_slice, width=DISPLAY_W, clamp=True)
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

            # what the canvas starts from, and when it is emptied
            empty_sig = (plane, d, st.session_state['canvas_gen'])

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
            canvas_kwargs['initial_drawing'] = canvas_initial_drawing(empty_sig)

            canvas_result = st_canvas(**canvas_kwargs)
            current_objects = []
            if canvas_result and canvas_result.json_data:
                current_objects = canvas_result.json_data.get("objects", [])

            st.slider("Slice Index", 0, n_slices - 1, key=slice_key)

            # The slider sends its value back a fifth of a second after the
            # server hands it a new one. Click again inside that window and
            # the browser sends the stale number along with the click;
            # Streamlit writes the widget values in before it runs the
            # callbacks, so stepping from the slider's own key would start
            # from the slice already left and land back on it. Step from
            # what was drawn instead -- nothing arriving from the browser
            # can overwrite that.
            st.session_state[f'_slice_shown_{plane}'] = d

            def _step(by):
                here = int(st.session_state.get(f'_slice_shown_{plane}', d))
                st.session_state[slice_key] = int(
                    min(max(here + by, 0), n_slices - 1))

            col_prev, col_next = st.columns(2)
            with col_prev:
                st.button("Previous", use_container_width=True,
                          on_click=_step, args=(-1,))
            with col_next:
                st.button("Next", use_container_width=True,
                          on_click=_step, args=(1,))
        # Save / clear
        if edit_mode != 'Viewer':
            col_save, col_clear = st.columns([1, 0.3])
            with col_save:
                save_contour = st.button('Save Contour', type='primary', use_container_width=True)
                if save_contour and canvas_result and canvas_result.image_data is not None and current_objects:
                    overlay_colors_list = np.array(
                        [c[:3] for c in OVERLAY_COLORS_BY_IDX.values()], dtype=np.float32)
                    overlay_channels = list(OVERLAY_COLORS_BY_IDX.keys())

                    # solid paint in a palette colour, and nothing else: the
                    # antialiased rim of a stroke comes back with its colour
                    # wrecked, and used to be read as whichever structure it
                    # landed nearest
                    label_px = canvas_paint_labels(canvas_result.image_data,
                                                   overlay_colors_list)

                    for ci, ch in enumerate(overlay_channels):
                        painted = (label_px == ci).astype(np.uint8)
                        painted = drop_specks(painted, stroke_width)
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

        # A couple of megabytes of JSON, serialised twice over and handed to
        # the component again on every rerun -- so on every click of Next
        # too. It can only change when the surfaces or the opacities do, so
        # it is built then and kept. Handing the component the identical
        # string also leaves the iframe where it is, so the render keeps the
        # angle it was turned to rather than reloading.
        html_stamp = (st.session_state.get('heart_render_gen'), id(meshes),
                      len(meshes), tuple(sorted(opacities.items())))
        if st.session_state.get('_heart_html_stamp') != html_stamp:
            html = _HEART_HTML.replace("__MESHES__",
                                       json.dumps(meshes_with_alpha))
            html = html.replace("__THREEJS__", _get_threejs_src())  # offline
            st.session_state['_heart_html'] = html
            st.session_state['_heart_html_stamp'] = html_stamp
        import streamlit.components.v1 as components
        components.html(st.session_state['_heart_html'], height=480,
                        scrolling=False)

        if st.session_state.get('heart_render_gen') != st.session_state.get('canvas_gen'):
            st.caption("⚠️ Edits since last render — click Update 3D.")

    st.divider()
    if st.button("🪄 Run Corrector Model", type="primary", use_container_width=True):
            with st.spinner("Running corrector..."):
                run_corrector_model(mode="dual15")
                st.session_state['_editor_primed'] = False
                st.session_state.pop('heart_geom', None)        # ← force 3D rebuild
                st.rerun()

    # ---- Download masks as a zipped {patient}_masks folder ----
    patient = str(st.session_state.get('patient_name', 'patient'))
    safe = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_'
                   for c in patient).strip().replace(' ', '_')

    # The file is built again only when something it is made of has moved:
    # an edit (every one bumps canvas_gen), a different series, a different
    # box, the other left-right sense. Clicking Previous or Next moves none
    # of them, so those clicks no longer pay for a file nobody has asked
    # for yet.
    export_stamp = (st.session_state.get('canvas_gen'),
                    str(st.session_state.get('sax_series_uid', '')), safe,
                    bool(st.session_state.get('flip_lr',
                                              default_lr_mirror())),
                    tuple(st.session_state.get('crop_offsets') or ()),
                    tuple(st.session_state.preprocessed.get('crop_box') or ()),
                    id(st.session_state['edited_mask']))
    if st.session_state.get('_export_stamp') != export_stamp:
        with st.spinner("Preparing the masks for download..."):
            zip_bytes, labelled = _masks_zip(edited_mask, N, safe)
        st.session_state['_export_zip'] = zip_bytes
        st.session_state['_export_voxels'] = labelled
        st.session_state['_export_stamp'] = export_stamp

    # said every run, not only the one that built the file
    if not st.session_state.get('_export_voxels', 1):
        st.warning("There is nothing to export: the mask is empty. If the "
                   "segmentation looks right in the editor, the crop box was "
                   "probably outside the volume.")

    st.download_button(
        "💾 Download masks (.nii.gz)",
        data=st.session_state.get('_export_zip', b''),
        file_name=f"{safe}_masks.zip",
        mime="application/zip",
        use_container_width=True,
    )

    # ---- Surfaces of the structures, built when they are asked for ----
    # Meshing finely enough to be worth looking at is seconds of work, so it
    # waits for the button rather than happening behind every rerun the way
    # the masks do. The surfaces themselves are kept between the two
    # formats, so asking for the other one is only the writing again.
    st.session_state.setdefault('surface_fmt', MESH_FORMATS[0])
    fmt = st.radio("Surface format", list(MESH_FORMATS), horizontal=True,
                   key='surface_fmt',
                   help="STL is read by every mesher and printer. OBJ keeps "
                        "the vertices shared, carries the normals the "
                        "surface was found with, and takes a colour per "
                        "structure from the .mtl beside it. Both are "
                        "written in the scan’s millimetres, LPS, which "
                        "is what the header says.")
    suffix = fmt.lower()
    mesh_cols = st.columns(2)
    with mesh_cols[0]:
        build_now = st.button(f"🧊 Build surfaces (.{suffix})",
                              key='build_surfaces', use_container_width=True)
    if build_now:
        with st.spinner("Meshing the structures..."):
            mesh_zip, triangles, where = _surfaces_zip(
                edited_mask, safe, fmt, stamp=export_stamp)
        st.session_state['_mesh_zip'] = mesh_zip
        st.session_state['_mesh_stamp'] = (export_stamp, fmt)
        st.session_state['_mesh_note'] = (
            f"{triangles:,} triangles of {fmt}, in the scan’s own "
            "millimetres" if where == 'scan' else
            f"{triangles:,} triangles of {fmt}, on the model’s "
            f"{TARGET_ISO} mm grid: the header had no geometry to put them "
            "back on")
    have_mesh = bool(st.session_state.get('_mesh_zip'))
    built_fmt = (st.session_state.get('_mesh_stamp') or (None, fmt))[1]
    with mesh_cols[1]:
        st.download_button(
            f"💾 Download surfaces (.{str(built_fmt).lower()})",
            data=st.session_state.get('_mesh_zip') or b'',
            file_name=f"{safe}_surfaces_{str(built_fmt).lower()}.zip",
            mime="application/zip",
            key='download_surfaces',
            disabled=not have_mesh,
            use_container_width=True,
        )
    if have_mesh:
        st.caption(
            st.session_state.get('_mesh_note', '')
            + ("" if st.session_state.get('_mesh_stamp') == (export_stamp, fmt)
               else " — build again for "
                    f"{'the latest edit' if built_fmt == fmt else fmt}"))

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
                st.session_state['pa_measure'] = measure_pa_tree(
                    pa_mask, spacing=spacing,
                    left_sign=chamber_left_sign(edited))
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
        
# --------------------------------------------------- PA tree orientation
# The MPA is the most anterior part of the pulmonary artery, so that is where
# the trunk starts. Anterior is the low end of axis 1: the transverse view is
# drawn with the chest wall at the top of the picture, the spine at the
# bottom. On a volume stored back to front, set PA_ANTERIOR_SIGN = -1 and the
# most anterior end becomes the high end of that axis instead.
PA_ANTERIOR_AXIS = 1
PA_ANTERIOR_SIGN = 1

# A skeleton leaf this close to a branch point is a spur thrown off by the
# skeletonisation, not a vessel end. One of those sitting a few voxels in
# front of the real MPA would otherwise be picked as the root.
PA_SPUR_MM = 10.0

# The left-right axis, for telling the two PA branches apart.
PA_LR_AXIS = 2


def chamber_left_sign(mask, axis=2):
    """Which way along `axis` the patient's left lies: +1 if it is the high
    end of the axis, -1 if the low end, or None if the mask cannot say.

    `mask` is the one-hot segmentation, (H, W, D, T, N). The left-sided
    chambers sit to the patient's left of the right-sided ones, so the sign
    of the gap between the two centroids is the answer, read off this case
    rather than assumed from an axis convention.
    """
    try:
        m = np.asarray(mask)
        if m.ndim == 5:
            m = m[..., 0, :]                      # one frame is plenty
        lab = np.argmax(m, axis=-1)
        here = lambda idxs: np.argwhere(np.isin(lab, idxs))
        for left, right in (((lv_idx, la_idx), (rv_idx, ra_idx)),
                            ((lv_idx,), (rv_idx,)),
                            ((la_idx,), (ra_idx,))):
            l_vox, r_vox = here(left), here(right)
            if len(l_vox) < 20 or len(r_vox) < 20:
                continue
            gap = float(l_vox[:, axis].mean() - r_vox[:, axis].mean())
            if abs(gap) < 1.0:                    # too close to call
                continue
            return 1 if gap > 0 else -1
    except Exception:
        pass
    return None

def measure_pa_tree(mask, spacing=(1.5,1.5,1.5), n_dense=300, n_rays=64,
                    cheb_deg=5, left_sign=None):
    """
    PA as a tree: shared MPA trunk (most anterior end → bifurcation) +
    two branches (bifurcation → each of the two longest leaves), named for
    the side they run to when `left_sign` says which side is which. Each segment is
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
    # The MPA starts at the most anterior end of the tree. This used to take
    # the most inferior leaf, which is a branch end whenever a branch runs
    # further foot-ward than the trunk -- and then the trunk, the
    # bifurcation and both branches are all measured off the wrong end.
    branch_pts = [n for n, d in G.degree() if d > 2]

    def _spur_mm(leaf):
        """How far a leaf sits from the nearest branch point, in mm."""
        if not branch_pts:
            return float('inf')
        dist = nx.single_source_dijkstra_path_length(G, leaf, weight='weight')
        return min(dist.get(b, float('inf')) for b in branch_pts)

    real_ends = [n for n in leaves if _spur_mm(n) >= PA_SPUR_MM]
    root = min(real_ends or leaves,
               key=lambda n: PA_ANTERIOR_SIGN * G.nodes[n]['coord'][PA_ANTERIOR_AXIS])
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

    # Which branch is which side. `left_sign` says which way along axis 2 the
    # patient's left lies; without it this falls back to what it always did,
    # which was to call the longer branch the RPA.
    first, second = "RPA", "LPA"
    if left_sign:
        def leftness(nodes):
            return left_sign * np.mean(
                [G.nodes[n]['coord'][PA_LR_AXIS] for n in nodes])
        if leftness(branchA_nodes) > leftness(branchB_nodes):
            first, second = "LPA", "RPA"

    segments = []
    for name, nodes in [("MPA", trunk_nodes), (first, branchA_nodes),
                        (second, branchB_nodes)]:
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