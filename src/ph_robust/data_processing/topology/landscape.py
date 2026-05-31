import torch
import numpy as np
import cv2
import gudhi as gd
import gudhi.representations


def process_Landscape(input, cfg):  # Processing to Landscape

    image_np = np.array(input)
    bw_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    cubical_complex = gd.CubicalComplex(
        dimensions=bw_img.shape, top_dimensional_cells=bw_img.flatten()
    )

    cubical_complex.persistence()
    L = gd.representations.Landscape(resolution=100)

    # For the Persistent Images, the concat output gives 2 images - a simple solution
    if cfg.topo.concat:
        L_0 = L.fit_transform(
            [cubical_complex.persistence_intervals_in_dimension(0)[:-1]]
        )
        L_t_0 = torch.tensor(L_0, dtype=torch.float)
        L_1 = L.fit_transform([cubical_complex.persistence_intervals_in_dimension(1)])
        L_t_1 = torch.tensor(L_1, dtype=torch.float)
        L_t = torch.cat([L_t_0, L_t_1], dim=0)

    elif cfg.topo.dim == 0:
        L_0 = L.fit_transform(
            [cubical_complex.persistence_intervals_in_dimension(cfg.topo.dim)[:-1]]
        )
        L_t = torch.tensor(L_0, dtype=torch.float)

    elif cfg.topo.dim == 1:
        L_1 = L.fit_transform(
            [cubical_complex.persistence_intervals_in_dimension(cfg.topo.dim)]
        )
        L_t = torch.tensor(L_1, dtype=torch.float)

    return L_t
