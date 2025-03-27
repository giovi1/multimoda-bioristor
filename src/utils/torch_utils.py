import torch

def get_device():
    """
    Get the appropriate device for PyTorch operations.
    Returns:
        torch.device: The device to use (MPS for Apple Silicon, CPU otherwise)
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def to_device(data, device=None):
    """
    Move data to the specified device.
    Args:
        data: The data to move (tensor, model, or dict/list of tensors)
        device: The device to move to (if None, uses get_device())
    Returns:
        The data moved to the specified device
    """
    if device is None:
        device = get_device()
    
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(to_device(item, device) for item in data)
    return data

def print_device_info():
    """
    Print information about the available PyTorch devices.
    """
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS (Apple Silicon GPU) available: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        print(f"MPS built: {torch.backends.mps.is_built()}")
    print(f"Current device: {get_device()}")

# Example usage:
if __name__ == "__main__":
    print_device_info()
    
    # Example of creating a tensor and moving it to the appropriate device
    x = torch.randn(3, 3)
    device = get_device()
    x = to_device(x, device)
    print(f"Tensor device: {x.device}") 