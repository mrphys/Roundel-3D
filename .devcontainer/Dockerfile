FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         git ca-certificates libopenexr-dev libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

RUN pip install --no-cache-dir \
    pydicom==2.2.2 \
    h5py==3.11.0 \
    scikit-learn==1.5.2 \
    scikit-image==0.24.0 \
    scipy==1.13.1 \
    pandas==2.2.2 \
    networkx==3.3 \
    plotly==5.24.1 \
    matplotlib==3.9.2 \
    nibabel==5.2.1 \
    "typing_extensions==4.13.2" \
    pylibjpeg==2.1.0 \
    pylibjpeg-libjpeg==2.4.0 \
    pylibjpeg-openjpeg==2.4.0 \
    streamlit==1.40.1 \
    stqdm==0.0.5 \
    opencv-python-headless==4.13.0.92 \
    streamlit_drawable_canvas==0.9.3 \
    monai==1.3.2

RUN git clone https://github.com/mrphys/im2sim.git /tmp/im2sim \
    && cd /tmp/im2sim \
    && git checkout 18ad52d \
    && pip install -r requirements.txt \
    && pip install . \
    && rm -rf /tmp/im2sim