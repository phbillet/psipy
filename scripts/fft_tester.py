#!/usr/bin/env python3
"""
Benchmark 1D and 2D FFT/IFFT using scipy.fft and pyfftw.

Runs performance tests for a set of array sizes, using multiprocessing to
evaluate multiple configurations concurrently. Results are printed as a table.
"""

import argparse
import multiprocessing as mp
import time
import numpy as np
import pyfftw
from scipy import fft

# ----------------------------------------------------------------------
# Benchmark functions for each package and dimension
# ----------------------------------------------------------------------

def bench_scipy_1d(size, runs):
    """Time forward+inverse 1D FFT using scipy.fft."""
    data = np.random.randn(size).astype(complex)
    # Warm-up (planning/cache)
    _ = fft.fft(data)
    _ = fft.ifft(data)
    start = time.perf_counter()
    for _ in range(runs):
        fft_data = fft.fft(data)
        _ = fft.ifft(fft_data)
    end = time.perf_counter()
    return (end - start) / runs

def bench_scipy_2d(shape, runs):
    """Time forward+inverse 2D FFT using scipy.fft."""
    data = np.random.randn(*shape).astype(complex)
    _ = fft.fft2(data)
    _ = fft.ifft2(data)
    start = time.perf_counter()
    for _ in range(runs):
        fft_data = fft.fft2(data)
        _ = fft.ifft2(fft_data)
    end = time.perf_counter()
    return (end - start) / runs

def bench_pyfftw_1d(size, runs, threads):
    """Time forward+inverse 1D FFT using pyfftw.builders."""
    # Create aligned arrays for best performance
    data = pyfftw.empty_aligned(size, dtype='complex128')
    data[:] = np.random.randn(size) + 1j * np.random.randn(size)
    # Build plans (planning time is included in the first execution)
    fft_plan = pyfftw.builders.fft(data, threads=threads)
    ifft_plan = pyfftw.builders.ifft(data, threads=threads)
    # Warm-up
    fft_data = fft_plan(data)
    _ = ifft_plan(fft_data)
    start = time.perf_counter()
    for _ in range(runs):
        fft_data = fft_plan(data)
        _ = ifft_plan(fft_data)
    end = time.perf_counter()
    return (end - start) / runs

def bench_pyfftw_2d(shape, runs, threads):
    """Time forward+inverse 2D FFT using pyfftw.builders."""
    data = pyfftw.empty_aligned(shape, dtype='complex128')
    data[:] = np.random.randn(*shape) + 1j * np.random.randn(*shape)
    fft_plan = pyfftw.builders.fft2(data, threads=threads)
    ifft_plan = pyfftw.builders.ifft2(data, threads=threads)
    # Warm-up
    fft_data = fft_plan(data)
    _ = ifft_plan(fft_data)
    start = time.perf_counter()
    for _ in range(runs):
        fft_data = fft_plan(data)
        _ = ifft_plan(fft_data)
    end = time.perf_counter()
    return (end - start) / runs

# ----------------------------------------------------------------------
# Worker function for multiprocessing
# ----------------------------------------------------------------------

def run_benchmark(args):
    """
    args: (package, dim, size, runs, threads)
    Returns a dict with results.
    """
    package, dim, size, runs, threads = args
    if package == 'scipy':
        if dim == 1:
            t = bench_scipy_1d(size, runs)
        else:  # dim == 2
            t = bench_scipy_2d(size, runs)   # size is a tuple for 2D
    else:  # pyfftw
        if dim == 1:
            t = bench_pyfftw_1d(size, runs, threads)
        else:
            t = bench_pyfftw_2d(size, runs, threads)
    return {
        'package': package,
        'dim': dim,
        'size': size,
        'time_sec': t,
        'threads': threads if package == 'pyfftw' else 'N/A'
    }

# ----------------------------------------------------------------------
# Main script
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark 1D/2D FFT/IFFT with scipy and pyfftw.'
    )
    parser.add_argument('--sizes-1d', nargs='+', type=int,
                        default=[1024, 4096, 16384],
                        help='List of 1D array sizes (powers of two recommended)')
    parser.add_argument('--sizes-2d', nargs='+', type=lambda s: tuple(map(int, s.split('x'))),
                        default=[(256,256), (512,512), (1024,1024)],
                        help='List of 2D dimensions as "NxM" (e.g., 512x512)')
    parser.add_argument('--runs', type=int, default=10,
                        help='Number of repeated runs for timing (default: 10)')
    parser.add_argument('--threads', type=int, default=1,
                        help='Number of threads for pyfftw (default: 1). '
                             'Set to >1 to use FFTW multithreading.')
    parser.add_argument('--processes', type=int,
                        default=mp.cpu_count(),
                        help='Number of parallel worker processes (default: all CPUs)')
    parser.add_argument('--output', type=str, help='Optional file to save results')

    args = parser.parse_args()

    # Prepare list of benchmark tasks
    tasks = []
    # 1D scipy
    for size in args.sizes_1d:
        tasks.append(('scipy', 1, size, args.runs, args.threads))
    # 2D scipy
    for shape in args.sizes_2d:
        tasks.append(('scipy', 2, shape, args.runs, args.threads))
    # 1D pyfftw
    for size in args.sizes_1d:
        tasks.append(('pyfftw', 1, size, args.runs, args.threads))
    # 2D pyfftw
    for shape in args.sizes_2d:
        tasks.append(('pyfftw', 2, shape, args.runs, args.threads))

    print(f"Running {len(tasks)} benchmarks with {args.processes} processes...")
    print(f"pyfftw threads per plan: {args.threads}")
    print("-" * 60)

    # Run tasks in parallel using a process pool
    with mp.Pool(processes=args.processes) as pool:
        results = pool.map(run_benchmark, tasks)

    # Sort results for readable output
    def sort_key(res):
        dim_order = {1:0, 2:1}
        size_val = res['size'][0] if res['dim']==2 else res['size']
        return (dim_order[res['dim']], size_val, res['package'])

    results.sort(key=sort_key)

    # Print table
    header = f"{'Package':<8} {'Dim':<4} {'Size':<15} {'Time (s)':<12} {'Threads'}"
    print(header)
    print("-" * len(header))
    for r in results:
        if r['dim'] == 1:
            size_str = str(r['size'])
        else:
            size_str = f"{r['size'][0]}x{r['size'][1]}"
        print(f"{r['package']:<8} {r['dim']:<4} {size_str:<15} {r['time_sec']:<12.6f} {r['threads']}")

    # Optionally save to file
    if args.output:
        import csv
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['package','dim','size','time_sec','threads'])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {args.output}")

if __name__ == '__main__':
    # Important for Windows: protect the entry point with if __name__ == '__main__'
    mp.freeze_support()  # optional, helps with some platforms
    main()