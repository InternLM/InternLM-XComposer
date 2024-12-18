import json
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str)
parser.add_argument("--num_chunks", type=int, default=1)
args = parser.parse_args()


def calc(folder, num_chunks):
    total = {}
    for i in range(num_chunks):
        part = json.load(open(f'{folder}/{i}_of_{num_chunks}.json'))
        for key in part:
            if key in total:
                total[key].extend(part[key])
            else:
                total[key] = part[key]

    for key in total:
        total[key] = np.mean(total[key])

    print(total)

    avg = np.mean([it for it in total.values()])
    print(f"avg: {avg}")


if __name__ == '__main__':
    calc(args.folder, args.num_chunks)