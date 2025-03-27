import torch
from ..utils.torch_utils import get_device, to_device, print_device_info

def main():
    # Print device information
    print_device_info()
    
    # Get the appropriate device
    device = get_device()
    
    # Create some example tensors
    x = torch.randn(1000, 1000)
    y = torch.randn(1000, 1000)
    
    # Move tensors to device
    x = to_device(x, device)
    y = to_device(y, device)
    
    # Perform some operations
    print("\nPerforming matrix multiplication...")
    start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    if start_time:
        start_time.record()
    
    z = torch.matmul(x, y)
    
    if start_time:
        end_time = torch.cuda.Event(enable_timing=True)
        end_time.record()
        torch.cuda.synchronize()
        print(f"Time taken: {start_time.elapsed_time(end_time):.2f} ms")
    
    print(f"Result shape: {z.shape}")
    print(f"Result device: {z.device}")

if __name__ == "__main__":
    main() 