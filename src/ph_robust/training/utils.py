import torch


def count_parameters(model):
    """Counts the total, trainable, and non-trainable parameters
    and estimates their memory usage."""

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print(f"--- Parameter Count ---")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable params: {non_trainable_params:,}")

    if total_params > 0:
        param_dtype = next(model.parameters()).dtype

        if param_dtype == torch.float32:
            bytes_per_param = 4
        elif param_dtype == torch.float16 or param_dtype == torch.bfloat16:
            bytes_per_param = 2
        elif param_dtype == torch.float64:
            bytes_per_param = 8
        else:
            try:
                bytes_per_param = torch.finfo(param_dtype).bits // 8
            except TypeError:
                try:
                    bytes_per_param = torch.iinfo(param_dtype).bits // 8
                except TypeError:
                    bytes_per_param = 4  # Default assumption

        total_memory_bytes = total_params * bytes_per_param
        total_memory_mb = total_memory_bytes / (1024**2)

        print("\n--- Memory Usage (Model Weights Only) ---")
        print(f"Assuming all params are: {param_dtype}")
        print(f"Bytes per parameter:   {bytes_per_param}")
        print(f"Total memory (MB):     {total_memory_mb:.2f} MB")

    else:
        print("\nModel has no parameters.")
