import torch

def get_device(device_str=None):
    """
    Get PyTorch device based on input string and available hardware.
    
    Args:
        device_str: Device string ('cuda', 'mps', 'cpu', or None)
        
    Returns:
        device: PyTorch device
    """
    # If no device specified, try to find the best available
    if device_str is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        return device
    
    # If device specified, check if it's available
    if device_str == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            print("CUDA is not available, falling back to CPU.")
            return torch.device('cpu')
    
    elif device_str == 'mps':
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            print("MPS is not available, falling back to CPU.")
            return torch.device('cpu')
    
    else:
        return torch.device('cpu') 