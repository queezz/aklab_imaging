import plotly.graph_objects as go
import numpy as np
from lmfit.models import GaussianModel


def plot_spectrum(intensity, x=None, zero=0.0):
    """
    Plot intensity vs. pixel.

    Parameters:
    limits: The starting and ending Y-pixel values to sum over,
            converting the image into a 1D spectrum.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=intensity - float(zero),
            mode="markers",
            marker=dict(size=4),
            name="Spectrum",
        )
    )

    fig.update_layout(
        title="Intensity vs Pixel",
        xaxis_title="pixel",
        yaxis_title="Intensity, counts",
        template="plotly_white",
    )

    return fig


def fit_single_gaussian(x, y, borders, fig, name=None):
    if name is None:
        name = f"Fit {borders[0]}-{borders[1]}"
    mask = (x >= borders[0]) & (x <= borders[1])
    x_fit = x[mask]
    y_fit = y[mask]

    model = GaussianModel()
    params = model.make_params(
        amplitude=np.max(y_fit), center=x_fit[np.argmax(y_fit)], sigma=5
    )
    result = model.fit(y_fit, params, x=x_fit)
    x_smooth = np.linspace(borders[0], borders[1], 500)
    y_smooth = result.eval(x=x_smooth)
    label = f"{name} center = {result.values['center']:.3f} fwhm = {result.values['fwhm']:.3f}"
    print(label)
    fig.add_trace(go.Scatter(x=x_smooth, y=y_smooth, mode="lines", name=name))

    label = f"{result.values['center']:.3f}<br>{result.values['fwhm']:.3f}"
    fig.add_annotation(
        x=result.values["center"],
        y=np.max(y_smooth),
        text=label,
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40,
    )
    return result


def image_to_spectrum(image, limits=None):
    """Sum up along vertical pixel to return 1D data
    Intensity vs pixel"""
    if limits == None:
        limits = [0, image.shape[0] - 1]
    line_data = image[slice(*limits), :].sum(axis=0)
    return line_data


def image_interactive(image):
    """
    Plots an interactive image with adjustable colorbar in Plotly.

    Args:
        image (np.ndarray): 2D image data.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=image,
            colorscale="Viridis",
            colorbar=dict(
                title="Intensity",
            ),
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
