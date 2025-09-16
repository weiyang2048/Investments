from matplotlib.colors import LinearSegmentedColormap


custom_cmap = LinearSegmentedColormap.from_list(
    "custom_red_green",
    [
        (0, "red"),
        (0.5, "white"),
        (1, "green"),
    ],
    N=256,
)
