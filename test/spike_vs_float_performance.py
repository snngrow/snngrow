#!/usr/bin/env python3
"""
Spike vs Float Performance Analysis for py-spy
==============================================

This script is specifically designed for py-spy profiling to identify
performance bottlenecks in spike multiplication vs float multiplication
across different matrix sizes from 1000x1000 to 10000x10000.

Usage with py-spy:
py-spy record --format speedscope -o out_spike.json --native -- python3 test/spike_vs_float_performance.py
"""

import torch
import numpy as np
import time
import sys
import gc

# Try to import the custom backend
try:
    import snngrow_backend
    BACKEND_AVAILABLE = True
    print("✅ SNNGrow backend loaded successfully")
except ImportError as e:
    BACKEND_AVAILABLE = False
    print(f"❌ Failed to import snngrow_backend: {e}")
    sys.exit(1)

def warm_gpu():
    """Warm up GPU to ensure consistent timing"""
    print("🔥 Warming up GPU...")
    for _ in range(10):
        x = torch.randn(1000, 1000, device='cuda')
        y = x * 2.0
        torch.cuda.synchronize()
    print("🔥 GPU warmup completed")

def test_matrix_size(size, iterations=50):
    """
    Test spike multiplication vs float multiplication for given matrix size
    
    Args:
        size: Matrix dimension (size x size)
        iterations: Number of iterations for timing
    """
    elements = size * size
    print(f"\n📊 Testing {size}×{size} ({elements:,} elements)")
    
    # Create test tensors
    bool_matrix = torch.randint(0, 2, (size, size), dtype=torch.bool, device='cuda')
    float_matrix = bool_matrix.float()  # Pre-converted for fair comparison
    scalar_value = 3.14159
    
    # Force memory allocation and GPU cache warm-up for this size
    for _ in range(5):
        _ = snngrow_backend.scalar_mul_cuda(bool_matrix, scalar_value)
        _ = float_matrix * scalar_value
        torch.cuda.synchronize()
    
    print(f"   Running {iterations} iterations for spike multiplication...")
    
    # Test 1: Spike multiplication (our implementation)
    spike_times = []
    torch.cuda.synchronize()
    
    for i in range(iterations):
        start_time = time.perf_counter()
        result_spike = snngrow_backend.scalar_mul_cuda(bool_matrix, scalar_value)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        spike_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Force immediate cleanup to avoid memory accumulation
        del result_spike
        if i % 10 == 0:
            torch.cuda.empty_cache()
    
    print(f"   Running {iterations} iterations for float multiplication...")
    
    # Test 2: Float multiplication (PyTorch native)
    float_times = []
    torch.cuda.synchronize()
    
    for i in range(iterations):
        start_time = time.perf_counter()
        result_float = float_matrix * scalar_value
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        float_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Force immediate cleanup
        del result_float
        if i % 10 == 0:
            torch.cuda.empty_cache()
    
    # Calculate statistics (remove outliers)
    spike_times_clean = sorted(spike_times)[5:-5]  # Remove top/bottom 5
    float_times_clean = sorted(float_times)[5:-5]
    
    spike_mean = np.mean(spike_times_clean)
    spike_std = np.std(spike_times_clean)
    float_mean = np.mean(float_times_clean)
    float_std = np.std(float_times_clean)
    
    speedup = float_mean / spike_mean
    efficiency = (speedup - 1) * 100
    
    # Calculate throughput
    spike_throughput = elements / (spike_mean * 1e6)  # GEPS
    float_throughput = elements / (float_mean * 1e6)  # GEPS
    
    print(f"   Spike Mul: {spike_mean:.3f}ms ± {spike_std:.3f}ms ({spike_throughput:.2f} GEPS)")
    print(f"   Float Mul: {float_mean:.3f}ms ± {float_std:.3f}ms ({float_throughput:.2f} GEPS)")
    print(f"   Speedup: {speedup:.2f}× ({efficiency:+.1f}%)")
    
    # Force garbage collection and GPU memory cleanup
    del bool_matrix, float_matrix
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        'size': size,
        'elements': elements,
        'spike_mean': spike_mean,
        'spike_std': spike_std,
        'float_mean': float_mean,
        'float_std': float_std,
        'speedup': speedup,
        'efficiency': efficiency,
        'spike_throughput': spike_throughput,
        'float_throughput': float_throughput
    }

def intensive_computation_workload():
    """
    Intensive computation workload to create clear profile signatures
    This helps py-spy identify hotspots in the spike multiplication code
    """
    print("\n🔥 Running intensive computation workload for py-spy analysis...")
    
    # Focus on problematic sizes (small matrices)
    small_sizes = [1000, 1500, 2000, 2500]
    large_sizes = [3000, 4000, 5000, 6000, 8000, 10000]
    
    print("\n📉 Small matrices (expected slower performance):")
    small_results = []
    for size in small_sizes:
        result = test_matrix_size(size, iterations=100)  # More iterations for small sizes
        small_results.append(result)
    
    print("\n📈 Large matrices (expected better performance):")
    large_results = []
    for size in large_sizes:
        result = test_matrix_size(size, iterations=50)  # Fewer iterations for large sizes
        large_results.append(result)
    
    return small_results, large_results

def comprehensive_analysis():
    """
    Comprehensive analysis across all matrix sizes
    """
    print("🚀 Comprehensive Spike vs Float Performance Analysis")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print()
    
    # Warm up GPU
    warm_gpu()
    
    # Run intensive workload for py-spy
    small_results, large_results = intensive_computation_workload()
    
    # Summary analysis
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    
    if small_results:
        small_avg_speedup = np.mean([r['speedup'] for r in small_results])
        print(f"Small matrices (1000-2500): Average speedup = {small_avg_speedup:.2f}×")
        
        worst_case = min(small_results, key=lambda x: x['speedup'])
        print(f"Worst case: {worst_case['size']}×{worst_case['size']} = {worst_case['speedup']:.2f}×")
    
    if large_results:
        large_avg_speedup = np.mean([r['speedup'] for r in large_results])
        print(f"Large matrices (3000-10000): Average speedup = {large_avg_speedup:.2f}×")
        
        best_case = max(large_results, key=lambda x: x['speedup'])
        print(f"Best case: {best_case['size']}×{best_case['size']} = {best_case['speedup']:.2f}×")
    
    print(f"\n💡 For py-spy analysis focus:")
    print(f"   - Small matrix performance bottlenecks")
    print(f"   - CUDA kernel launch overhead")
    print(f"   - Memory access patterns")
    
    return small_results + large_results

def main():
    """
    Main function optimized for py-spy profiling
    """
    try:
        # Check CUDA availability
        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            sys.exit(1)
        
        print("🎯 Starting performance analysis for py-spy profiling...")
        print("   Use: py-spy record --format speedscope -o out_spike.json --native -- python3 test/spike_vs_float_performance.py")
        print()
        
        # Run comprehensive analysis
        results = comprehensive_analysis()
        
        print("\n✅ Performance analysis completed!")
        print("   Check py-spy output for detailed profiling data")
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        sys.exit(1)
    finally:
        # Final cleanup
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()
