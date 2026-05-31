import os
from ph_robust.data_processing.topology.landscape import process_Landscape
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import gudhi as gd
import gudhi.representations


def _landscape(ax):
    for k in range(5):
        ax.plot(land_h0[k * 100 : (k + 1) * 100], label=f"$\\lambda_{k + 1}$")
    ax.legend(fontsize=8)
    ax.set_title("Landscape")


def _betti(ax):
    ax.plot(betti_h0)
    ax.set_xlabel("filtration (sampled)")
    ax.set_ylabel("Betti number")
    ax.set_title("Betti Curve")


def _silhouette(ax):
    ax.plot(silh_h0)
    ax.set_xlabel("filtration (sampled)")
    ax.set_title("Silhouette")


def _persistent_image(ax):
    ax.imshow(pi_h0, origin="lower")
    ax.set_title("Persistent Image")


def save_plot(plot_fn, fname, figsize=(4, 4)):
    """Render a matplotlib plot to its own tight, transparent PNG."""
    fig, ax = plt.subplots(figsize=figsize)
    plot_fn(ax)
    fig.savefig(
        os.path.join(out_dir, fname), dpi=DPI, bbox_inches="tight", transparent=True
    )
    plt.close(fig)


trainset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    ),
)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)
images, labels = next(iter(trainloader))
img_data = images[0]

rgb_disp = (img_data.permute(1, 2, 0) / 2 + 0.5).clamp(0, 1).numpy()
image_np = (rgb_disp * 255).astype(np.uint8)  # uint8 [0,255], as in process_PI
bw_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

cubical_complex = gd.CubicalComplex(
    dimensions=bw_img.shape, top_dimensional_cells=bw_img.flatten()
)
diag = cubical_complex.persistence()  # both dims, for barcode/diagram
intervals_h0 = cubical_complex.persistence_intervals_in_dimension(0)[:-1]

L = gd.representations.Landscape(resolution=100)
land_h0 = L.fit_transform([intervals_h0])[0]

PI = gd.representations.PersistenceImage(
    bandwidth=5,
    resolution=[64, 64],
    weight=lambda x: (x[0] - x[1]) ** 2,
    im_range=[0, 256, 0, 256],
)
pi_h0 = PI.fit_transform([intervals_h0])[0].reshape(64, 64)

BC = gd.representations.BettiCurve(resolution=100)
betti_h0 = BC.fit_transform([intervals_h0])[0]

SH = gd.representations.Silhouette(resolution=100, weight=lambda x: (x[1] - x[0]) ** 2)
silh_h0 = SH.fit_transform([intervals_h0])[0]

out_dir = "pipeline_figs"
os.makedirs(out_dir, exist_ok=True)
DPI = 200


plt.imsave(f"{out_dir}/01_rgb_input.png", rgb_disp)
plt.imsave(f"{out_dir}/02_grayscale.png", bw_img, cmap="gray", vmin=0, vmax=255)

save_plot(
    lambda ax: gd.plot_persistence_barcode(diag, axes=ax, legend=True), "03_barcode.png"
)
save_plot(
    lambda ax: gd.plot_persistence_diagram(diag, axes=ax, legend=True), "04_diagram.png"
)

save_plot(_landscape, "05_landscape_h0.png")
save_plot(_persistent_image, "06_pi.png")
save_plot(_betti, "07_betti_h0.png")
save_plot(_silhouette, "08_silhouette_h0.png")

print(f"Saved 8 PNGs to ./{out_dir}/")
