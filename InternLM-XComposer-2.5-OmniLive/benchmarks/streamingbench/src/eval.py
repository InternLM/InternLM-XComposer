import os
import sys
sys.path.append('benchmarks/streamingbench/src')
from utils.data_execution import load_data
from model.ixc2d5_ol import IXC2d5_OL
import argparse


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized non-empty chunks."""
    # Compute sizes of the chunks
    avg = len(lst) // n
    remainder = len(lst) % n

    # Create the chunks
    chunks = []
    start = 0
    for i in range(n):
        chunk_size = avg + (1 if i < remainder else 0)
        chunks.append(lst[start:start + chunk_size])
        start += chunk_size

    return chunks


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def main(args):
    data = load_data(args.data_file)
    data = get_chunk(data, args.num_chunks, args.chunk_id)

    ####### BENCHMARK #######

    if args.benchmark_name == "Streaming":
        from benchmark.StreamingBench import StreamingBench
        benchmark = StreamingBench(data)
    if args.benchmark_name == "StreamingProactive":
        from benchmark.StreamingBenchProactive import StreamingBenchProactive
        benchmark = StreamingBenchProactive(data)
    if args.benchmark_name == "StreamingSQA":
        from benchmark.StreamingBenchSQA import StreamingBenchSQA
        benchmark = StreamingBenchSQA(data)

    ####### MODEL ############
    model = IXC2d5_OL(args.ixc_model_path, max_frame=args.max_frame)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    benchmark.eval(data, model, args.output_file, video_folder=args.video_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True, help="Path to the data file")
    parser.add_argument("--video_folder", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--benchmark_name", type=str, required=True, help="Name of the benchmark")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_id", type=int, required=True, help="GPU id")
    parser.add_argument("--ixc-model-path", type=str, default="internlm-xcomposer2d5-ol-7b/base")
    parser.add_argument("--max-frame", type=int, default=64)

    args = parser.parse_args()
    print(args)
    main(args)