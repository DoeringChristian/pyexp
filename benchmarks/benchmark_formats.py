"""Benchmark comparison between JSONL and Protobuf logging formats."""

import tempfile
import time
from pathlib import Path

from pyexp import Logger, LogReader


def benchmark_write(
    num_scalars: int = 10000,
    num_text: int = 1000,
    num_figures: int = 100,
    use_protobuf: bool = True,
) -> dict:
    """Benchmark write performance."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_dir = Path(tmp_dir) / "logs"

        # Benchmark scalars
        logger = Logger(log_dir, use_protobuf=use_protobuf)
        start = time.perf_counter()
        for i in range(num_scalars):
            logger.set_global_it(i)
            logger.add_scalar("loss", float(i) * 0.001)
            logger.add_scalar("accuracy", 1.0 - float(i) * 0.0001)
        logger.flush()
        scalar_time = time.perf_counter() - start

        # Benchmark text
        logger2 = Logger(log_dir / "text_test", use_protobuf=use_protobuf)
        start = time.perf_counter()
        for i in range(num_text):
            logger2.set_global_it(i)
            logger2.add_text("log", f"This is log message number {i} with some content")
        logger2.flush()
        text_time = time.perf_counter() - start

        # Benchmark figures (small dicts as proxy for real figures)
        logger3 = Logger(log_dir / "figure_test", use_protobuf=use_protobuf)
        start = time.perf_counter()
        for i in range(num_figures):
            logger3.set_global_it(i)
            logger3.add_figure("plot", {"data": list(range(100)), "title": f"Plot {i}"})
        logger3.flush()
        figure_time = time.perf_counter() - start

        # Get file sizes
        if use_protobuf:
            scalar_size = (log_dir / "events.pb").stat().st_size
            text_size = (log_dir / "text_test" / "events.pb").stat().st_size
            figure_size = (log_dir / "figure_test" / "events.pb").stat().st_size
        else:
            scalar_size = (log_dir / "scalars.jsonl").stat().st_size
            text_size = (log_dir / "text_test" / "text.jsonl").stat().st_size
            # For JSONL, figures are in separate files
            figure_size = sum(
                f.stat().st_size
                for f in (log_dir / "figure_test").rglob("*.cpkl")
            )

        return {
            "scalar_time": scalar_time,
            "text_time": text_time,
            "figure_time": figure_time,
            "scalar_size": scalar_size,
            "text_size": text_size,
            "figure_size": figure_size,
        }


def benchmark_read(
    num_scalars: int = 10000,
    num_text: int = 1000,
    use_protobuf: bool = True,
) -> dict:
    """Benchmark read performance."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_dir = Path(tmp_dir) / "logs"

        # Write data first
        logger = Logger(log_dir, use_protobuf=use_protobuf)
        for i in range(num_scalars):
            logger.set_global_it(i)
            logger.add_scalar("loss", float(i) * 0.001)
            logger.add_scalar("accuracy", 1.0 - float(i) * 0.0001)
        for i in range(num_text):
            logger.set_global_it(i)
            logger.add_text("log", f"Log message {i}")
        logger.flush()

        # Benchmark reading scalars
        start = time.perf_counter()
        reader = LogReader(log_dir)
        loss_data = reader.load_scalars("loss")
        acc_data = reader.load_scalars("accuracy")
        scalar_read_time = time.perf_counter() - start

        # Benchmark reading text (fresh reader to avoid cache)
        reader2 = LogReader(log_dir)
        start = time.perf_counter()
        text_data = reader2.load_text("log")
        text_read_time = time.perf_counter() - start

        return {
            "scalar_read_time": scalar_read_time,
            "text_read_time": text_read_time,
            "num_scalars_read": len(loss_data) + len(acc_data),
            "num_text_read": len(text_data),
        }


def format_size(size_bytes: int) -> str:
    """Format byte size as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def main():
    print("=" * 70)
    print("Pyexp Logging Format Benchmark")
    print("=" * 70)

    # Configuration
    num_scalars = 10000
    num_text = 1000
    num_figures = 100

    print(f"\nConfiguration:")
    print(f"  Scalars: {num_scalars} iterations x 2 tags = {num_scalars * 2} entries")
    print(f"  Text: {num_text} entries")
    print(f"  Figures: {num_figures} entries")

    # Run benchmarks
    print("\n" + "-" * 70)
    print("WRITE PERFORMANCE")
    print("-" * 70)

    jsonl_write = benchmark_write(num_scalars, num_text, num_figures, use_protobuf=False)
    proto_write = benchmark_write(num_scalars, num_text, num_figures, use_protobuf=True)

    print(f"\n{'Metric':<25} {'JSONL':<15} {'Protobuf':<15} {'Speedup':<10}")
    print("-" * 65)

    # Scalar write time
    speedup = jsonl_write["scalar_time"] / proto_write["scalar_time"]
    print(
        f"{'Scalar write time':<25} "
        f"{jsonl_write['scalar_time']*1000:>10.1f} ms   "
        f"{proto_write['scalar_time']*1000:>10.1f} ms   "
        f"{speedup:>6.2f}x"
    )

    # Text write time
    speedup = jsonl_write["text_time"] / proto_write["text_time"]
    print(
        f"{'Text write time':<25} "
        f"{jsonl_write['text_time']*1000:>10.1f} ms   "
        f"{proto_write['text_time']*1000:>10.1f} ms   "
        f"{speedup:>6.2f}x"
    )

    # Figure write time
    speedup = jsonl_write["figure_time"] / proto_write["figure_time"]
    print(
        f"{'Figure write time':<25} "
        f"{jsonl_write['figure_time']*1000:>10.1f} ms   "
        f"{proto_write['figure_time']*1000:>10.1f} ms   "
        f"{speedup:>6.2f}x"
    )

    print(f"\n{'Metric':<25} {'JSONL':<15} {'Protobuf':<15} {'Ratio':<10}")
    print("-" * 65)

    # File sizes
    ratio = jsonl_write["scalar_size"] / proto_write["scalar_size"]
    print(
        f"{'Scalar file size':<25} "
        f"{format_size(jsonl_write['scalar_size']):>12}   "
        f"{format_size(proto_write['scalar_size']):>12}   "
        f"{ratio:>6.2f}x"
    )

    ratio = jsonl_write["text_size"] / proto_write["text_size"]
    print(
        f"{'Text file size':<25} "
        f"{format_size(jsonl_write['text_size']):>12}   "
        f"{format_size(proto_write['text_size']):>12}   "
        f"{ratio:>6.2f}x"
    )

    ratio = jsonl_write["figure_size"] / proto_write["figure_size"]
    print(
        f"{'Figure file size':<25} "
        f"{format_size(jsonl_write['figure_size']):>12}   "
        f"{format_size(proto_write['figure_size']):>12}   "
        f"{ratio:>6.2f}x"
    )

    print("\n" + "-" * 70)
    print("READ PERFORMANCE")
    print("-" * 70)

    jsonl_read = benchmark_read(num_scalars, num_text, use_protobuf=False)
    proto_read = benchmark_read(num_scalars, num_text, use_protobuf=True)

    print(f"\n{'Metric':<25} {'JSONL':<15} {'Protobuf':<15} {'Speedup':<10}")
    print("-" * 65)

    # Scalar read time
    speedup = jsonl_read["scalar_read_time"] / proto_read["scalar_read_time"]
    print(
        f"{'Scalar read time':<25} "
        f"{jsonl_read['scalar_read_time']*1000:>10.1f} ms   "
        f"{proto_read['scalar_read_time']*1000:>10.1f} ms   "
        f"{speedup:>6.2f}x"
    )

    # Text read time
    speedup = jsonl_read["text_read_time"] / proto_read["text_read_time"]
    print(
        f"{'Text read time':<25} "
        f"{jsonl_read['text_read_time']*1000:>10.1f} ms   "
        f"{proto_read['text_read_time']*1000:>10.1f} ms   "
        f"{speedup:>6.2f}x"
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_jsonl_write = (
        jsonl_write["scalar_time"]
        + jsonl_write["text_time"]
        + jsonl_write["figure_time"]
    )
    total_proto_write = (
        proto_write["scalar_time"]
        + proto_write["text_time"]
        + proto_write["figure_time"]
    )

    total_jsonl_size = (
        jsonl_write["scalar_size"]
        + jsonl_write["text_size"]
        + jsonl_write["figure_size"]
    )
    total_proto_size = (
        proto_write["scalar_size"]
        + proto_write["text_size"]
        + proto_write["figure_size"]
    )

    print(f"""
Protobuf format advantages:
  - Write speed:    {total_jsonl_write/total_proto_write:.1f}x faster ({total_proto_write*1000:.0f}ms vs {total_jsonl_write*1000:.0f}ms)
  - File size:      {total_jsonl_size/total_proto_size:.1f}x smaller ({format_size(total_proto_size)} vs {format_size(total_jsonl_size)})
  - Single file:    All data in one events.pb file (simpler management)

JSONL format advantages:
  - Human readable: Can inspect with text editors
  - Selective read:  Faster for reading specific data types
  - Streaming:       Can read partial files while writing

Recommendation:
  Use Protobuf (default) for training runs where write performance matters.
  Use JSONL (use_protobuf=False) for debugging or when human readability is needed.
""")


if __name__ == "__main__":
    main()
