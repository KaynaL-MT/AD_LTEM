import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def show_im(
    image,
    title=None,
    simple=False,
    origin="upper",
    cbar=True,
    cbar_title="",
    scale=None,
    **kwargs,
):
    """Display an image on a new axis.

    Takes a 2D array and displays the image in grayscale with optional title on
    a new axis. In general it's nice to have things on their own axes, but if
    too many are open it's a good idea to close with plt.close('all').

    Args:
        image (2D array): Image to be displayed.
        title (str): (`optional`) Title of plot.
        simple (bool): (`optional`) Default output or additional labels.

            - True, will just show image.
            - False, (default) will show a colorbar with axes labels, and will adjust the
              contrast range for images with a very small range of values (<1e-12).

        origin (str): (`optional`) Control image orientation.

            - 'upper': (default) (0,0) in upper left corner, y-axis goes down.
            - 'lower': (0,0) in lower left corner, y-axis goes up.

        cbar (bool): (`optional`) Choose to display the colorbar or not. Only matters when
            simple = False.
        cbar_title (str): (`optional`) Title attached to the colorbar (indicating the
            units or significance of the values).
        scale (float): Scale of image in nm/pixel. Axis markers will be given in
            units of nanometers.

    Returns:
        None
    """
    fig, ax = plt.subplots()
    if not simple and np.max(image) - np.min(image) < 1e-12:
        # adjust coontrast range
        vmin = np.min(image) - 1e-12
        vmax = np.max(image) + 1e-12
        im = ax.matshow(image, cmap="gray", origin=origin, vmin=vmin, vmax=vmax)
    else:
        im = ax.matshow(image, cmap="gray", origin=origin, **kwargs)

    if title is not None:
        ax.set_title(str(title), pad=0)

    if simple:
        plt.axis("off")
    else:
        plt.tick_params(axis="x", top=False)
        ax.xaxis.tick_bottom()
        ax.tick_params(direction="in")
        if scale is None:
            ticks_label = "pixels"
        else:

            def mjrFormatter(x, pos):
                return f"{scale*x:.3g}"

            fov = scale * max(image.shape[0], image.shape[1])

            if fov < 4e3:  # if fov < 4um use nm scale
                ticks_label = " nm "
            elif fov > 4e6:  # if fov > 4mm use m scale
                ticks_label = "  m  "
                scale /= 1e9
            else:  # if fov between the two, use um
                ticks_label = " $\mu$m "
                scale /= 1e3

            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter))
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter))

        if origin == "lower":
            ax.text(y=0, x=0, s=ticks_label, rotation=-45, va="top", ha="right")
        elif origin == "upper":  # keep label in lower left corner
            ax.text(
                y=image.shape[0], x=0, s=ticks_label, rotation=-45, va="top", ha="right"
            )

        if cbar:
            plt.colorbar(im, ax=ax, pad=0.02, format="%.2g", label=str(cbar_title))

    plt.show()
    return


def get_ifft(fft):
    """
    Get inverse-FFT of 2D image
    """
    return np.fft.ifft2(np.fft.ifftshift(fft))

def get_fft(im):
    """
    Get FFT of 2D image
    """
    return np.fft.fftshift(np.fft.fft2(im))


import numpy as np
from matplotlib import colors
import textwrap
import sys


def color_im(Bx, By, Bz=None, rad=None, hsvwheel=True, background="black"):
    """Make the RGB image from x and y component vector maps.

    The color intensity corresponds to the in-plane vector component. If a
    z-component is given, it will map from black (negative) to white (positive).

    Args:
        Bx (2D array): (M x N) array consisting of the x-component of the vector
            field.
        By (2D array): (M x N) array consisting of the y-component of the vector
            field.
        Bz (2D array): optional (M x N) array consisting of the y-component of
            the vector field.
        rad (int): (`optional`) Radius of color-wheel in pixels. (default None -> height/16)
            Set rad = 0 to remove color-wheel.
        hsvwheel (bool):
            - True  -- (default) use a standard HSV color-wheel (3-fold)
            - False -- use a four-fold color-wheel
        background (str):
            - 'black' -- (default) magnetization magnitude corresponds to value.
            - 'white' -- magnetization magnitude corresponds to saturation.

    Returns:
        ``ndarray``: Numpy array (M x N x 3) containing the color-image.
    """

    if rad is None:
        rad = Bx.shape[0] // 16
        rad = max(rad, 16)

    bmag = np.sqrt(Bx**2 + By**2)

    if rad > 0:
        pad = 10  # padding between edge of image and color-wheel
    else:
        pad = 0
        rad = 0

    dimy = np.shape(By)[0]
    if dimy < 2 * rad:
        rad = dimy // 2
    dimx = np.shape(By)[1] + 2 * rad + pad
    cimage = np.zeros((dimy, dimx, 3))

    if hsvwheel:
        # Here we will proceed with using the standard HSV color-wheel routine.
        # Get the Hue (angle) as By/Bx and scale between [0,1]
        hue = (np.arctan2(By, Bx) + np.pi) / (2 * np.pi)

        if Bz is None:
            z_wheel = False
            # make the color image
            if background == "white":  # value is ones, magnitude -> saturation
                cb = np.dstack(
                    (hue, bmag / np.max(bmag), np.ones([dimy, dimx - 2 * rad - pad]))
                )
            elif background == "black":  # saturation is ones, magnitude -> values
                cb = np.dstack(
                    (hue, np.ones([dimy, dimx - 2 * rad - pad]), bmag / np.max(bmag))
                )
            else:
                print(
                    textwrap.dedent(
                        """
                    An improper argument was given in color_im().
                    Please choose background as 'black' or 'white.
                    'white' -> magnetization magnitude corresponds to saturation.
                    'black' -> magnetization magnitude corresponds to value."""
                    )
                )
                sys.exit(1)

        else:
            z_wheel = True
            theta = np.arctan2(Bz, np.sqrt(Bx**2 + By**2))
            value = np.where(theta < 0, np.cos(2 * theta) / 2 + 1 / 2, 1)
            # value = np.where(theta<0, 1-1/(1+np.exp(10*theta*2/np.pi+5)), 1)#sigmoid
            sat = np.where(theta > 0, np.cos(2 * theta) / 2 + 1 / 2, 1)
            # sat = np.where(theta>0, 1-1/(1+np.exp(-10*theta*2/np.pi+5)), 1)#sigmoid
            cb = np.dstack((hue, sat, value))

        if rad > 0:  # add the color-wheel
            cimage[:, : -2 * rad - pad, :] = cb
            # make the color-wheel and add to image
            wheel = colorwheel_HSV(rad, background=background, z=z_wheel)
            cimage[
                dimy // 2 - rad : dimy // 2 + rad,
                dimx - 2 * rad - pad // 2 : -pad // 2,
                :,
            ] = wheel
        else:
            cimage = cb
        # Convert to RGB image.
        cimage = colors.hsv_to_rgb(cimage)

    else:  # four-fold color wheel
        bmag = np.where(bmag != 0, bmag, 1.0001)
        cang = Bx / bmag  # cosine of the angle
        sang = np.sqrt(1 - cang**2)  # and sin

        # define the 4 color quadrants
        q1 = ((Bx >= 0) * (By <= 0)).astype(int)
        q2 = ((Bx < 0) * (By < 0)).astype(int)
        q3 = ((Bx <= 0) * (By >= 0)).astype(int)
        q4 = ((Bx > 0) * (By > 0)).astype(int)

        # as is By = Bx = 0 -> 1,1,1 , so to correct for that:
        no_B = np.where((Bx == 0) & (By == 0))
        q1[no_B] = 0
        q2[no_B] = 0
        q3[no_B] = 0
        q4[no_B] = 0

        # Apply to green, red, blue
        green = q1 * bmag * np.abs(sang)
        green += q2 * bmag
        green += q3 * bmag * np.abs(cang)

        red = q1 * bmag
        red += q2 * bmag * np.abs(sang)
        red += q4 * bmag * np.abs(cang)

        blue = (q3 + q4) * bmag * np.abs(sang)

        # apply to cimage channels and normalize
        cimage[:, : dimx - 2 * rad - pad, 0] = red
        cimage[:, : dimx - 2 * rad - pad, 1] = green
        cimage[:, : dimx - 2 * rad - pad, 2] = blue
        cimage = cimage / np.max(cimage)

        # add color-wheel
        if rad > 0:
            mid_y = dimy // 2
            cimage[mid_y - rad : mid_y + rad, dimx - 2 * rad :, :] = colorwheel_RGB(rad)

    return cimage