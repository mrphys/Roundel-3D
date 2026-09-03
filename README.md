# Roundel 3D

Deep learning segmentation of 3D MRI volumes, with manual correction tools, a correction-informed refinement model, and vessel measurement.

## Overview

The workflow has three stages:

1. **Automatic segmentation** — a trained model produces an initial segmentation of a 3D MRI volume.
2. **Manual correction** — review the result and edit it where the model got it wrong.
3. **Informed correction** — a second model takes your edits as input and propagates them through the rest of the volume, so you don't have to correct every slice by hand.

Once you have a final segmentation, the tool allows for vessel measurements, enabling the user to select the point of measurement 

## Installation

```bash
git clone github.com/mrphys/Roundel-3D
cd Roundel-3D
pip install -r requirements.txt
```

Requires Python and a CUDA-capable GPU for training. Inference will run on CPU, but slowly.

## Input and output

Input volumes are expected as single .zip file of DICOM data corresponding to a 3D MR dataset acquired sagittal

## License

Licensed under the Elastic License 2.0. See [LICENSE.md](LICENSE.md).

**Not for clinical use.** This is a research tool. Outputs have not been validated for diagnostic or treatment decisions.
