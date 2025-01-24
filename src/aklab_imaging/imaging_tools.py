# Standard library imports
import os
import time
from os.path import join
import hashlib

# Third-party library imports
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import plotly.graph_objects as go
from tqdm import tqdm

# Custom or local library imports
import aklab_imaging.textcolor as tc


class Imaging:
    def __init__(self, Spectrometer):
        self.camera = Spectrometer.camera
        self.grating_motor = Spectrometer.grating_motor
        self.spectrometer = Spectrometer
        self._init_variables()
        self.set_image_name()

    def _init_variables(self):
        """Placeholders for useful parameters"""
        self.image = None
        self.attributes = None
        self.series = None
        self.background = None
        self.basepath = "./"
        self.exposure = None
        self.filename = None
        self.framecounter = 0
        self.filepath = None
        self.imagesaved = False
        self.last_saved_hash = None

    def capture_image(
        self, exposure=100, ex=None,
    ):
        """exposure - ms"""
        camera = self.camera
        if ex is not None:
            exposure = ex
        self.exposure = exposure

        vbin = 1
        self.attributes = {
            "temperature": camera.getTemperature(),
            "device_status": camera.getDeviceStatus(),
            "exposure": exposure,
            "frame_type": "light",
        }
        camera.setExposureTime(exposure)
        camera.setVBin(vbin)
        camera.setImageArea(10, 0, 2058, 512 // vbin)
        camera.exposeFrame()
        time.sleep(0.1)
        image_data = camera.grabFrame(out=np.empty((512 // vbin, 2048), np.uint16))
        self.image = image_data
        self.imagesaved = False

    def capture_series(self, ex=500, frames=20, **kws):
        images = []
        for i in tqdm(range(frames), desc="Capturing"):
            self.capture_image(ex=ex, **kws)
            images.append(self.image)

        self.series = np.array(images)
        self.background = self.series.sum(axis=0) / frames

    def reset_series(self):
        self.series = None

    def convert_to_xarray(self):
        """Convert to xarray to save as netcdf"""
        data = xr.DataArray(
            self.image,
            dims=["y", "x"],
            coords={"image_counter": 0},
            attrs=self.attributes,
        )
        return data

    def set_image_name(self, name="JobinYvonFLframe", suffix=""):
        """Set image name and suffix for saving data"""
        self.name = name
        self.suffix = f"-{suffix}" if suffix else ""

    def make_filename(self,update_counter=True):
        """Generate a unique filename based on the current state.

        If a file with the same name exists in the folder, increment the frame counter.
        """
        count = self.spectrometer.get_position()
        base_filename = f"{self.name}-{count}-{self.exposure}ms{self.suffix}"

        if base_filename == self.filename:
            self.framecounter += 1
        else:
            self.framecounter = 0
            self.filename = base_filename

        while True:
            final_filename = join(
                self.basepath, f"{self.filename}-{self.framecounter:03}.nc"
            )
            if not os.path.exists(final_filename):
                break
            self.framecounter += 1

        return final_filename

    def set_basepath(self, basepath):
        """Update basepath"""
        self.basepath = basepath
        if not os.path.isdir(self.basepath):
            print(
                f"{tc.YELLOW}Basepath does not exist. Creating {self.basepath}.{tc.RESET}"
            )
            os.makedirs(self.basepath)

    
    def _get_image_hash(self):
        """Generate a hash for the current image to track its uniqueness."""
        if self.image is None:
            return None
        return hashlib.sha256(self.image.tobytes()).hexdigest()            

    def save_image(self, overwrite=False):
        """Save the current image to the desired location."""
        if self.basepath is None:
            message = f"{tc.RED}Basepath is not set.{tc.RESET}"
            message += " Use `Imaging.basepath = 'path/to/your/datafolder/'`."
            print(message)
            return

        if self.image is None:
            print(f"{tc.RED}No image to save.{tc.RESET} Capture an image first.")
            return
        
        image_hash = self._get_image_hash()
        if self.imagesaved and image_hash == self.last_saved_hash and not overwrite:
            print(f"{tc.YELLOW}Image already saved as {self.filepath}.{tc.RESET}")
            return        

        self.filepath = self.make_filename()

        if os.path.exists(self.filepath) and not overwrite:
            print(f"{tc.YELLOW}File already exists: {self.filepath}.{tc.RESET}")
            print("To overwrite, use `save_image(overwrite=True)`.")
            return

        data = self.convert_to_xarray()

        try:
            data.to_netcdf(self.filepath)
            self.imagesaved = True
            self.last_saved_hash = image_hash
            print(f"Image saved successfully at {self.filepath}.")
        except Exception as e:
            print(f"{tc.RED}Error saving image: {e}{tc.RESET}")

    def plot_spectrum(self, limits=None, image=None, zero=0.0):
        """Plot intensity vs. pixel.

        Parameters:
        limits: The starting and ending Y-pixel values to sum over,
                converting the image into a 1D spectrum.
        """
        if limits == None:
            limits = [0, self.image.shape[0] - 1]

        if image == None:
            line_data = self.image[slice(*limits), :].sum(axis=0)
        else:
            line_data = image[slice(*limits), :].sum(axis=0)

        line_data = line_data - float(zero)

        fig = go.Figure()

        fig.add_trace(go.Scatter(y=line_data, mode="lines", name="Spectrum"))

        fig.update_layout(
            title="Intensity vs Pixel",
            xaxis_title="pixel",
            yaxis_title="Intensity, counts",
            template="plotly_white",
        )

        fig.show()

    def plot_image(self, **kws):
        image = self.image
        fig, axs = plt.subplots(2, 1)
        fig.set_size_inches([10, 8])

        axs[0].plot(image.sum(axis=0), "k")
        img = axs[1].imshow(image, origin="lower", norm=LogNorm())

        cbar = fig.colorbar(img, ax=axs[1], orientation="horizontal")

        xlim = kws.get("xlim", [0, 2058])
        [ax.set_xlim(*xlim) for ax in axs]
        l1 = kws.get("l1", 400)
        l2 = kws.get("l2", 300)
        plt.axhline(l1, c="w", lw=1)
        plt.axhline(l2, c="w", lw=1)

    def plotly_plot(self):
        """
        Plots an interactive image with adjustable colorbar in Plotly.

        Args:
            image (np.ndarray): 2D image data.
        """
        image = self.image
        fig = go.Figure()

        # Add heatmap
        fig.add_trace(
            go.Heatmap(
                z=image,
                colorscale="Viridis",
                colorbar=dict(title="Intensity",),
                zmin=image.min(),
                zmax=image.max(),
            )
        )

        steps_zmin = [
            {"label": f"{v:.1f}", "method": "restyle", "args": [{"zmin": [v]}]}
            for v in np.linspace(image.min(), image.max(), 20)
        ]

        steps_zmax = [
            {"label": f"{v:.1f}", "method": "restyle", "args": [{"zmax": [v]}]}
            for v in np.linspace(image.min(), image.max(), 20)
        ]

        fig.update_layout(
            sliders=[
                dict(
                    active=0,
                    currentvalue={"prefix": "Min: "},
                    pad={"t": 20},
                    steps=steps_zmin,
                    x=0.0,
                    xanchor="left",
                ),
                dict(
                    active=0,
                    currentvalue={"prefix": "Max: "},
                    pad={"t": 100},  # Additional padding for second slider
                    steps=steps_zmax,
                    x=0,
                    xanchor="left",
                ),
            ],
            title="Interactive Image with Adjustable Colorbar",
            height=600,  # Ensure sufficient height for sliders
            width=800,
        )

        fig.show()
