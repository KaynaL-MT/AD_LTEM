"""Helper functions for TIE.

An assortment of helper functions that load images, pass data, and generally
are used in the reconstruction. Additionally, a couple of functions used for
displaying images and stacks.

Author: Arthur McCray, ANL, Summer 2019.
"""

import os

import sys
sys.path.append('../../')
import textwrap

from copy import deepcopy
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact
from ncempy.io import dm
from scipy import ndimage
from scipy.ndimage import median_filter
from skimage import io
from tifffile import TiffFile
from routines.colorwheel import color_im
from routines.params import *


# ============================================================= #
#      Functions used for loading and passing the TIE data      #
# ============================================================= #


def load_data(
    path=None, fls_file="", al_file="", flip=None, flip_fls_file=None, filtersize=3
):
    """Load files in a directory (from a .fls file) using ncempy.

    For more information on how to organize the directory and load the data, as
    well as how to setup the .fls file please refer to the README or the
    TIE_template.ipynb notebook.

    Args:
        path (str): Location of data directory.
        fls_file (str): Name of the .fls file which contains the image names and
            defocus values.
        al_file (str): Name of the aligned stack image file.
        flip (Bool): True if using a flip stack, False otherwise. Uniformly
            thick films can be reconstructed without a flip stack. The
            electrostatic phase shift will not be reconstructed.
        flip_fls_file (str): Name of the .fls file for the flip images if they
            are not named the same as the unflip files. Will only be applied to
            the /flip/ directory.
        filtersize (int): (`optional`) The images are processed with a median
            filter to remove hot pixels which occur in experimental data. This
            should be set to 0 for simulated data, though generally one would
            only use this function for experimental data.

    Returns:
        list: List of length 3, containing the following items:

        - imstack: list of numpy arrays
        - flipstack: list of numpy arrays, empty list if flip == False
        - ptie: TIE_params object holding a reference to the imstack and many
          other parameters.

    """
    unflip_files = []
    flip_files = []

    # Finding the unflip fls file
    path = os.path.abspath(path)
    if not fls_file.endswith(".fls"):
        fls_file += ".fls"
    if os.path.isfile(os.path.join(path, fls_file)):
        fls_full = os.path.join(path, fls_file)
    elif os.path.isfile(os.path.join(path, "unflip", fls_file)):
        fls_full = os.path.join(path, "unflip", fls_file)
    elif os.path.isfile(os.path.join(path, "tfs", fls_file)) and not flip:
        fls_full = os.path.join(path, "tfs", fls_file)
    else:
        raise FileNotFoundError("fls file could not be found.")

    if flip_fls_file is None:  # one fls file given
        fls = []
        with open(fls_full) as file:
            for line in file:
                fls.append(line.strip())

        num_files = int(fls[0])
        if flip:
            for line in fls[1 : num_files + 1]:
                unflip_files.append(os.path.join(path, "unflip", line))
            for line in fls[1 : num_files + 1]:
                flip_files.append(os.path.join(path, "flip", line))
        else:
            if os.path.isfile(os.path.join(path, "tfs", fls[2])):
                tfs_dir = "tfs"
            else:
                tfs_dir = "unflip"
            for line in fls[1 : num_files + 1]:
                unflip_files.append(os.path.join(path, tfs_dir, line))

    else:  # there are 2 fls files given
        if not flip:
            print(
                textwrap.dedent(
                    """
                You probably made a mistake.
                You're defining both unflip and flip fls files but have flip=False.
                Proceeding anyways, will only load unflip stack (if it doesnt break).\n"""
                )
            )
        # find the flip fls file
        if not flip_fls_file.endswith(".fls"):
            flip_fls_file += ".fls"
        if os.path.isfile(os.path.join(path, flip_fls_file)):
            flip_fls_full = os.path.join(path, flip_fls_file)
        elif os.path.isfile(os.path.join(path, "flip", flip_fls_file)):
            flip_fls_full = os.path.join(path, "flip", flip_fls_file)

        fls = []
        flip_fls = []
        with open(fls_full) as file:
            for line in file:
                fls.append(line.strip())

        with open(flip_fls_full) as file:
            for line in file:
                flip_fls.append(line.strip())

        assert int(fls[0]) == int(flip_fls[0])
        num_files = int(fls[0])
        for line in fls[1 : num_files + 1]:
            unflip_files.append(os.path.join(path, "unflip", line))
        for line in flip_fls[1 : num_files + 1]:
            flip_files.append(os.path.join(path, "flip", line))

    f_inf = unflip_files[num_files // 2]
    _, scale = read_image(f_inf)

    try:
        al_stack, _ = read_image(os.path.join(path, al_file))
    except FileNotFoundError as e:
        print("Incorrect aligned stack filename given.")
        raise e

    # quick median filter to remove hotpixels, kinda slow
    print("filtering takes a few seconds")
    al_stack = median_filter(al_stack, size=(1, filtersize, filtersize))

    if flip:
        f_inf_flip = flip_files[num_files // 2]
        _, scale_flip = read_image(f_inf_flip)
        if round(scale, 3) != round(scale_flip, 3):
            print("Scale of the two infocus images are different.")
            print(f"Scale of unflip image: {scale:.4f} nm/pix")
            print(f"Scale of flip image: {scale_flip:.4f} nm/pix")
            print("Proceeding with scale from unflip image.")
            print("If this is incorrect, change value with >>ptie.scale = XX #nm/pixel")

        imstack = al_stack[:num_files]
        flipstack = al_stack[num_files:]
    else:
        imstack = al_stack
        print(imstack)
        flipstack = []

    # show_im(imstack[num_files // 2])
    # show_im(flipstack[num_files // 2])

    # read the defocus values
    defvals = fls[-(num_files // 2) :]
    print(defvals)
    assert num_files == 2 * len(defvals) + 1
    defvals = [float(i) for i in defvals]  # defocus values +/-

    # Create a TIE_params object
    ptie = TIE_params(imstack, flipstack, defvals, scale, flip, path)
    print("Data loaded successfully.")
    return (imstack, flipstack, ptie)
