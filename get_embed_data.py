import argparse
from datasets import process_data, process_empFreq
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--repo', default=1, type=int)
    parser.add_argument('--empFreq', default=False, action='store_true')
    args = parser.parse_args()

    if args.empFreq:
        process_empFreq(args.dataset)
    else:
        process_data(args.dataset, args.repo)