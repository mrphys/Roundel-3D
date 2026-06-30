import numpy as np
import os
import pandas as pd
import glob
import json
import math
import re
from PIL import Image
from pathlib import Path
from scipy import stats
import os
import pickle
from pathlib import Path
import cv2
from skimage.measure import label 
import pydicom
import streamlit as st
import zipfile
import tempfile
import io
st.session_state['N'] = 5

class Pipeline:
    def __init__(self, dcms):
        self.dcms  = dcms
        self.sax_df = self.get_sax_df() # read dicom headers for each file into a dataframe called sax_df
        self.image = self.get_sax_image() # create the sax image

    def get_sax_df(self):
        '''
        puts all the dicom header information for ALL dicoms into a dataframe
        '''
        sax_df = {}
        print(f"Number of dicoms in series: {len(self.dcms)}")
        dicoms_in_series = self.read_dicom_header(self.dcms)
        print(f"len dicoms_in_series: {len(dicoms_in_series)}")
        sax_df.update(dicoms_in_series)
        print(f"len sax_df after update: {len(sax_df)}")
        sax_df = pd.DataFrame.from_dict(sax_df, orient = 'index').reset_index(drop = True) # put dicom info for all images into a dataframe

        sax_df = sax_df[sax_df ['triggertime'].notna()] #remove scans with no triggertimes
        if sax_df.slicelocation.isnull().any():
            main_axis = np.argmax(np.cross(sax_df['orientation'].iloc[0][:3], sax_df['orientation'].iloc[0][3:]))
            sax_df['slicelocation'] = sax_df['position'].apply(lambda x: x[main_axis])
        sax_df = sax_df.sort_values(['slicelocation','triggertime'])
        if self.is_sax_valid(sax_df):
            return sax_df
        else:
            st.error('Not a Valid SAX series')
            st.stop()


    def read_dicom_header(self,dicoms_in_series):
        '''
        read the information we want from the header and assert that the series has to have pixelarray data
        '''
        sax_df = {}
        for dicom_num, dcm in enumerate(dicoms_in_series): # go through dicom in each series
            try: # if dicom doesn't have an associate pixel array (image), ignore dicom
                try:
                    image = dcm.pixel_array
                    image_exists = True
                except:
                    image_exists = False                
            except Exception as e:
                image_exists = False

            if image_exists: # if image exists and is not 3d read all other information
                sax_df[dicom_num] = {}
                dcm.PatientBirthDate = dcm.PatientBirthDate = "19000101"
                sax_df[dicom_num]['dcm'] = dcm
                sax_df[dicom_num]['image'] = dcm.pixel_array
                sax_df[dicom_num]['uid'] = dcm.SOPInstanceUID
                sax_df[dicom_num]['seriesuid'] = dcm.SeriesInstanceUID 

                # have to use try and excepts, if the dicom doesn't the information stored use nan
                try:
                    sax_df[dicom_num]['slicelocation'] = int(dcm.InstanceNumber)
                except Exception:
                    sax_df[dicom_num]['slicelocation'] = np.nan
            
                try:
                    sax_df[dicom_num]['thickness'] = round(dcm.SpacingBetweenSlices,3)
                except:
                    try:
                        sax_df[dicom_num]['thickness'] = round(dcm.SliceThickness,3)
                    except:
                        sax_df[dicom_num]['thickness'] = np.nan
                try:
                    sax_df[dicom_num]['seriesnumber'] = dcm.SeriesNumber
                except:
                    sax_df[dicom_num]['seriesnumber'] = np.nan
                try:
                    sax_df[dicom_num]['triggertime'] = round(dcm.TriggerTime)#int(np.ceil(dcm.TriggerTime / 5) * 5)
                except:
                    sax_df[dicom_num]['triggertime'] = np.nan
                try:
                    sax_df[dicom_num]['N_timesteps'] = int(dcm.CardiacNumberOfImages)
                except:
                    sax_df[dicom_num]['N_timesteps'] = np.nan
                try:
                    sax_df[dicom_num]['orientation'] = [round(val,3) for val in dcm.ImageOrientationPatient]
                except:
                    sax_df[dicom_num]['orientation'] = np.nan
                try:
                    sax_df[dicom_num]['position'] = [round(val,3) for val in dcm.ImagePositionPatient]
                except:
                    sax_df[dicom_num]['position'] = np.nan
                try:
                    sax_df[dicom_num]['pixelspacing'] = round(dcm.PixelSpacing[0],3)
                except:
                    sax_df[dicom_num]['pixelspacing'] = np.nan
                # try:
                #     sax_df[dicom_num]['phase'] = dcm[0x0028, 0x1052].value
                # except:
                #     try:
                #         sax_df[dicom_num]['phase'] = list(dcm.RealWorldValueMappingSequence)[0].RealWorldValueIntercept 
                #     except:
                #         sax_df[dicom_num]['phase'] = 0
        return sax_df

    def get_sax_image(self):
        '''
        makes the 4D sax image image[height, width, slice, time]
        '''
        try:
            image_4D = []
            for uni_slice in self.sax_df.slicelocation.unique():
                image_4D.append(np.stack(self.sax_df.loc[self.sax_df['slicelocation'] == uni_slice].image.values, axis =-1))
            image_4D = np.stack(image_4D, axis = -2)
        except:
            self.status = 'Mismatched timesteps'
            raise ValueError('Mismatched timesteps')
        return image_4D
    
    def calc_N_timesteps(self,sax_df):
        '''
        N timesteps is given in the dicom header as number cardiac images, but it's not always there.
        This calculates the number of timesteps there should be in a series by taking the modal value of the 
        number of trigger times for each series.
        '''
        sax_df = sax_df.drop_duplicates(subset = ['slicelocation','triggertime']) # remove any repeated scans
        # print(f"length of sax_df after dropping duplicates: {len(sax_df)}")
        unique_slices = sax_df.slicelocation.unique()
        # print(f"Unique slices: {unique_slices}")
        possible_N_timesteps = []
        for uni_slice in unique_slices:
            # print(f"Number of images for slice {uni_slice}: {len(sax_df.loc[sax_df['slicelocation'] == uni_slice])}")
            # print(f"Number of timesteps for slice {uni_slice}: {len(sax_df.loc[sax_df['slicelocation'] == uni_slice].triggertime.unique())}")
            # print(f"Trigger times for slice {uni_slice}: {sax_df.loc[sax_df['slicelocation'] == uni_slice].triggertime.unique()}")
            # print(f"Duplicate trigger times for slice {uni_slice}: {sax_df.loc[sax_df['slicelocation'] == uni_slice].triggertime.duplicated().any()}")
            possible_N_timesteps.append(len(sax_df.loc[sax_df['slicelocation'] == uni_slice]))
        images_without_slice_location = sax_df[~sax_df['slicelocation'].isin(unique_slices)]
        # print(f"number of images without slice location: {len(images_without_slice_location)}")

        N_timesteps = np.min(possible_N_timesteps)
        # print(f"Possible N_timesteps: {possible_N_timesteps}, min N_timesteps: {N_timesteps}")
        return int(N_timesteps)
    
    def is_sax_valid(self, sax_df):
        '''
        Checks that the stack is a valid 3D static volume:
        enough slices, and exactly one image per slice location.
        '''
        min_slices = 20

        N_slices = sax_df.slicelocation.nunique()
        N_images = len(sax_df)

        # In a static 3D volume, each slice location should appear exactly once.
        images_per_slice_ok = (N_images == N_slices)

        if N_slices >= min_slices and images_per_slice_ok:
            sax_valid = True
        else:
            print(
                f"Not a valid 3D SAX volume. "
                f"Number of slices: {N_slices}, total number of images: {N_images}. "
                f"Minimum number of slices: {min_slices}. "
                f"Images per slice == 1: {images_per_slice_ok}"
            )
            sax_valid = False
        return sax_valid

    def __iter__(self):
        yield self.image
        yield self.sax_df




def save_mask_as_dicom_series(masks, save_path):
    os.makedirs(f'{save_path}', exist_ok=True)
    sax_df = st.session_state['sax_df']

    if sax_df.seriesuid.nunique() > 1:
        multi_series = True
    else:
        multi_series = False
        series_uid = pydicom.uid.generate_uid()

        
    for slice_num, uni_slice in enumerate(sax_df.slicelocation.unique()):
        slice_df = sax_df.loc[sax_df['slicelocation'] == uni_slice]

        if multi_series:
            series_uid = pydicom.uid.generate_uid()

        for time_num, uni_time in enumerate(slice_df.triggertime.unique()):
            dcm = slice_df.loc[slice_df['triggertime'] == uni_time, 'dcm'].item()
            arr = masks[:, :, slice_num, time_num].astype(dcm.pixel_array.dtype) * 200 # increase the value for DICOM

            dcm.SeriesDescription = 'Roundel'
            dcm.PixelData = arr.tobytes()

            dcm.StudyInstanceUID = dcm.StudyInstanceUID
            dcm.SeriesInstanceUID = series_uid
            dcm.SOPInstanceUID = pydicom.uid.generate_uid()

            try:
                dcm.save_as(f'{save_path}/slice_{slice_num:02}_time_{time_num:02}.dcm')
            except Exception as e:
                dcm.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
                dcm['PixelData'].is_undefined_length = False
                dcm.save_as(f'{save_path}/slice_{slice_num:02}_time_{time_num:02}.dcm')


def extract_dicom_from_zip(zip_file):
    """Return DICOM datasets from a ZIP file without writing to disk."""
    dcms = []

    zip_file.seek(0)
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        for name in zip_ref.namelist():
            if name.endswith("/") or name.startswith("__MACOSX"):
                continue
            basename = os.path.basename(name)
            if basename.startswith("."):
                continue
            with zip_ref.open(name) as f:
                data = f.read()
            try:
                ds = pydicom.dcmread(io.BytesIO(data), force=True)
                if not hasattr(ds, "PixelData"):
                    continue
                dcms.append(ds)
            except Exception:
                continue
    print(f"Extracted {len(dcms)} DICOM files from ZIP.")
    return dcms