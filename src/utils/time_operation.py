import time
import functools
from collections import defaultdict

class ComputationTimer:
    def __init__(self):
        self.times = defaultdict(float)
        self.counts = defaultdict(int)

    def time_operation(self, op_name):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                end = time.perf_counter()
                duration = end - start
                self.times[op_name] += duration
                self.counts[op_name] += 1
                #print(f"Operation {op_name}: {duration:.6f} seconds (total: {self.times[op_name]:.4f}s)")
                return result
            return wrapper
        return decorator

    def total_time(self):
        return sum(self.times.values())

    def report(self):
        print("\n" + "="*50)
        print("COMPUTATION TIME SUMMARY")
        print("="*50)
        for op_name in sorted(self.times.keys()):
            avg = self.times[op_name] / self.counts[op_name] if self.counts[op_name] > 0 else 0
            print(f"{op_name}:")
            print(f"  Total: {self.times[op_name]:.6f} seconds")
            print(f"  Calls: {self.counts[op_name]}")
            print(f"  Average: {avg:.6f} seconds")
        print("-"*50)
        print(f"OVERALL TOTAL: {self.total_time():.6f} seconds")
        print("="*50)