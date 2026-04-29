import torch
import numpy as np
import cv2
import gudhi as gd
import gudhi.representations


def process_PI(input, cfg):  # Processing to Persistant images

    image_np = np.array(input)
    bw_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    cubical_complex = gd.CubicalComplex(
        dimensions=bw_img.shape, top_dimensional_cells=bw_img.flatten()
    )

    cubical_complex.persistence()
    PI = gd.representations.PersistenceImage(
        bandwidth=5,
        resolution=[64, 64],
        weight=lambda x: (x[0] - x[1]) ** 2,
        im_range=[0, 256, 0, 256],
    )

    # For the Persistent Images, the concat output gives 2 images - a simple solution
    if cfg.topo.concat:
        PI_0 = PI.fit_transform(
            [cubical_complex.persistence_intervals_in_dimension(0)[:-1]]
        )
        L_t_0 = torch.tensor(PI_0, dtype=torch.float).reshape([1, 64, 64])
        PI_1 = PI.fit_transform([cubical_complex.persistence_intervals_in_dimension(1)])
        L_t_1 = torch.tensor(PI_1, dtype=torch.float).reshape([1, 64, 64])
        L_t = torch.cat([L_t_0, L_t_1], dim=0)

    elif cfg.topo.dim == 0:
        PI = PI.fit_transform(
            [cubical_complex.persistence_intervals_in_dimension(cfg.topo.dim)[:-1]]
        )
        L_t = torch.tensor(PI, dtype=torch.float).reshape([1, 64, 64])

    elif cfg.topo.dim == 1:
        PI = PI.fit_transform(
            [cubical_complex.persistence_intervals_in_dimension(cfg.topo.dim)]
        )
        L_t = torch.tensor(PI, dtype=torch.float).reshape([1, 64, 64])

    return L_t
