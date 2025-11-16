import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version in Torch: {torch.version.cuda}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
# Quan trọng: Thử tạo một tensor trên GPU xem có crash không
try:
    x = torch.rand(5, 5).cuda()
    print("Test GPU: OK! Tensor created successfully.")
except Exception as e:
    print(f"Test GPU: FAILED. Error: {e}")
    
import torch
print(f"PyTorch version: {torch.__version__}")
print("Architectures supported by this binary:", torch.cuda.get_arch_list())