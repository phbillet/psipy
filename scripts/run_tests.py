#!/usr/bin/env python3
import subprocess
import os
import time
import resource

# Path to the tests directory
tests_dir = "tests"

# List all .py files in the tests directory
test_files = [
    f for f in os.listdir(tests_dir)
    if f.endswith(".py") and os.path.isfile(os.path.join(tests_dir, f))
]

# List to store results
results = []

# Execute each .py file
for test_file in test_files:
    file_path = os.path.join(tests_dir, test_file)
    print(f"Executing {test_file}...")

    # Measure CPU time and elapsed time
    start_time = time.time()
    start_cpu = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime

    try:
        subprocess.run(["python", file_path], check=True)
        success = True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error executing {test_file}: {e}\n")
        success = False

    end_time = time.time()
    end_cpu = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime

    elapsed_time = end_time - start_time
    cpu_time = end_cpu - start_cpu

    # Add results
    results.append({
        "file": test_file,
        "success": success,
        "elapsed_time": elapsed_time,
        "cpu_time": cpu_time
    })

    if success:
        print(f"✅ {test_file} executed successfully.\n")

# Display summary table
print("\nTest summary table:")
print("-" * 80)
print(f"{'File':<30} | {'Success':<8} | {'Elapsed time (s)':<15} | {'CPU time (s)':<12}")
print("-" * 80)

total_elapsed = 0.0
total_cpu = 0.0
success_count = 0

for result in results:
    status = "✅ Yes" if result["success"] else "❌ No"
    print(f"{result['file']:<30} | {status:<8} | {result['elapsed_time']:<15.3f} | {result['cpu_time']:<12.3f}")

    total_elapsed += result["elapsed_time"]
    total_cpu += result["cpu_time"]
    if result["success"]:
        success_count += 1

print("-" * 80)
print(f"{'Total':<30} | {success_count}/{len(results)} | {total_elapsed:<15.3f} | {total_cpu:<12.3f}")
