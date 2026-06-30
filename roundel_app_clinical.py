# --------------------------------------------------------------
# Configure Streamlit page
# # --------------------------------------------------------------

import logging
logging.getLogger("streamlit.runtime.media_file_storage").setLevel(logging.ERROR)
logging.getLogger("streamlit.web.server.media_file_handler").setLevel(logging.ERROR)
logging.getLogger("tornado.application").setLevel(logging.ERROR)

from roundel_utils import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
st.set_page_config(page_title="Roundel", page_icon="⭕️", layout='wide')

# --------------------------------------------------------------
# App
# --------------------------------------------------------------
st.write('# Roundel')

if "current_view" not in st.session_state:
    st.session_state.current_view = "Segmentation ⭕"

view = st.segmented_control(
    "Tab",
    options=["Segmentation ⭕", "Preview Segmentation 👁️", "Mask Editor 🔧", "Vessel Measurements 📏"],
    default=st.session_state.current_view,
    label_visibility='hidden',
)
if view is not None:
    st.session_state.current_view = view
active_view = view or st.session_state.current_view

st.divider()

if active_view == "Segmentation ⭕":
    segmentation_view()

elif active_view == "Preview Segmentation 👁️":
    preview_segmentation_view()

elif active_view == "Mask Editor 🔧":
    mask_editor_view()

elif active_view == "Vessel Measurements 📏":
    vessel_measurement_view()