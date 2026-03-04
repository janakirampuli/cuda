import torch
import time

def run_benchmark():
    M, N, K = 1024, 1024, 1024
    # disabling tf32 as i want apples to apples comparision
    torch.backends.cuda.matmul.allow_tf32 = False

    A = torch.randn((M, K), device='cuda', dtype=torch.float32)
    B = torch.randn((K, N), device='cuda', dtype=torch.float32)

    # warmup
    for _ in range(10):
        C = torch.matmul(A, B)

    torch.cuda.synchronize()

    num_iterations = 20

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iterations):
        C = torch.matmul(A, B)
    end_event.record()

    torch.cuda.synchronize()

    total_ms = start_event.elapsed_time(end_event)
    avg_ms = total_ms / num_iterations

    tflops_total = 2.0 * M * N * K
    achieved_tflops = (tflops_total / (avg_ms * 1e-3)) / 1e12

    print(f'average time: {avg_ms} ms')
    print(f'achieved Compute: {achieved_tflops} TFLOPS')
    print(f'theoretical Max Compute: ~31.2 TFLOPS')

if __name__ == "__main__":
    run_benchmark()

'''
python3 11_matmul_pytorch.py
average time: 0.16465920209884644 ms
achieved Compute: 13.041989883510098 TFLOPS
theoretical Max Compute: ~31.2 TFLOPS

'''