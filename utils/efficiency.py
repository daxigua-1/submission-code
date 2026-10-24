import time
import torch
from thop import profile, clever_format


def benchmark_model_efficiency(model, args, device='cuda'):
    print("\n" + "=" * 60)
    print(" Model Efficiency Benchmark (Strict Mode) ".center(60))
    print("=" * 60)

    model.eval()
    model.to(device)

    B = getattr(args, 'batch_size', 32)
    L = args.seq_len
    N = getattr(args, 'enc_in', 7)

    # (x, x_mark_enc, x_dec, x_mark_dec, mask)
    dummy_x = torch.randn(B, L, N).float().to(device)
    dummy_mark = torch.randn(B, L, 4).float().to(device)

    if 'PEMS' in args.data or 'Solar' in args.data:
        dummy_mark = None

    print(f"  [Config] Batch Size: {B} | Seq Len: {L} | Features: {N}")

    try:
        macs, params = profile(model, inputs=(dummy_x, dummy_mark, None, dummy_mark), verbose=False)
        flops = macs * 2

        flops_str, params_str = clever_format([flops, params], "%.3f")
        print(f"  [Theory] Parameters  : {params_str}")
        print(f"  [Theory] FLOPs       : {flops_str}")
    except Exception as e:
        print(f"  [Theory] FLOPs calculation skipped. Error: {e}")

    if torch.cuda.is_available():
        with torch.no_grad():
            for _ in range(50):
                _ = model(dummy_x, dummy_mark, None, dummy_mark)
        torch.cuda.synchronize()

    num_iterations = 100
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_x, dummy_mark, None, dummy_mark)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_time = end_time = time.time() - start_time
    time_per_batch = (total_time / num_iterations) * 1000
    print(f"  [Speed]  Time/Batch  : {time_per_batch:.2f} ms")

    if torch.cuda.is_available():
        max_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"  [Memory] Peak Mem    : {max_memory:.2f} MB")
        torch.cuda.reset_peak_memory_stats(device)

    print("=" * 60 + "\n")