# --------------------------------------------------------------
# Configure Streamlit page
# --------------------------------------------------------------

import logging
logging.getLogger("streamlit.runtime.media_file_storage").setLevel(logging.ERROR)
logging.getLogger("streamlit.web.server.media_file_handler").setLevel(logging.ERROR)
logging.getLogger("tornado.application").setLevel(logging.ERROR)

from roundel_utils import *
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
st.set_page_config(page_title="Roundel", page_icon="⭕️", layout='wide')

# --------------------------------------------------------------
# App
# --------------------------------------------------------------
st.write('# Roundel')

VIEWS = ["Segmentation ⭕", "Mask Editor 🔧", "Vessel Measurements 📏"]

# Keyed, and with no `default=`. A widget without a key is identified by its
# parameters, so handing this one a default that changes every time the tab
# changes made it a different widget on each click -- and a click then took
# two runs, the first of which still drew the view being left behind. That
# second run is what left one page's elements showing under the next one.
st.session_state.setdefault("view_tabs", VIEWS[0])
view = st.segmented_control("Tab", options=VIEWS, key="view_tabs",
                            label_visibility='hidden')

# Clicking the open tab deselects it and returns None; stay where we are.
if view in VIEWS:
    st.session_state["current_view"] = view
active_view = st.session_state.get("current_view", VIEWS[0])

st.divider()

# The view is drawn straight into the page.
#
# It used to be drawn inside an st.empty(), which is a placeholder: writing
# to one replaces what was in it, so the browser drops that whole subtree
# and builds a new one. With the entire page inside it, every rerun
# re-mounted every element on the page -- both iframes reloaded, and every
# button that sizes to its container was laid out again. That is what made
# the editor look as though it reloaded itself when a slice changed.
#
# It was put there to stop one page's elements showing under the next one.
# The tab control above is what fixes that: one run per tab change, so
# there is no run that draws the page being left. Streamlit removes what
# the last run drew by itself.
if active_view == "Segmentation ⭕":
    segmentation_view()

elif active_view == "Mask Editor 🔧":
    mask_editor_view()

elif active_view == "Vessel Measurements 📏":
    vessel_measurement_view()