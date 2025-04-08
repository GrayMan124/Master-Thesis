#!/usr/bin/env python

import os
import itertools

# Fixed parameters
fixed_params = {
    "cores": 1,
    "tv": "pi_img"
}

# Variables to combine
models = ["TBR_img", "TR_img"]
dims = [0, 1,'concat']
tbs_options = ["small", "normal", "large"]
lr_options = [0.0001, 0.0002, 0.0003]

# Create output directory if it doesn't exist
os.makedirs("pi_img_config", exist_ok=True)

# Generate all combinations
config_idx = 1
all_combinations = list(itertools.product(models, dims, tbs_options, lr_options))

for model, dim, tbs, lr in all_combinations:
    # Create a meaningful name
    model_short = "TBR" if model == "TBR_img" else "TR"
    concat_str = "_CONCAT" if dim =='concat' else f"_DIM{dim}"
    tbs_short = tbs[0].upper()  # S, N, or L
    
    name = f"LAND_{model_short}{concat_str}_{tbs_short}_LR{lr}"
    
    # Create config content
    config_content = []
    config_content.append(f"--cores={fixed_params['cores']}")
    config_content.append(f"--tv={fixed_params['tv']}")
    config_content.append(f"--name={name}")
    config_content.append(f"--lr={lr}")
    config_content.append(f"--model={model}")
    
    # Add dim only if not using concat
    if dim!='concat':
        config_content.append(f"--topodim={dim}")
    else:
        config_content.append("--topodim_concat")
    config_content.append(f"--tbs={tbs}")
    
    # Write to file
    with open(f"pi_img_config/pi_img_{config_idx}.txt", "w") as f:
        f.write("\n".join(config_content))
    
    print(f"Created config_{config_idx}.txt with name: {name}")
    config_idx += 1

print(f"Total configs created: {len(all_combinations)}")