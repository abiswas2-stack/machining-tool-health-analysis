import argparse
from machining_file import MachiningFile


def main():
    parser = argparse.ArgumentParser(
        description="Compare sensor signals between two machining data files."
    )
    parser.add_argument('--file1', required=True, help='Path to first .mat file (e.g. baseline)')
    parser.add_argument('--group1', required=True, help="Group name inside file1, e.g. 'BaselineCrop'")
    parser.add_argument('--file2', required=True, help='Path to second .mat file (e.g. a fault condition)')
    parser.add_argument('--group2', required=True, help="Group name inside file2, e.g. 'ToolWearCrop'")
    parser.add_argument('--field', default='SpindleX', help='Sensor field to analyze (default: SpindleX)')
    parser.add_argument('--segments', type=int, default=10, help='Number of segments to compare (default: 10)')

    args = parser.parse_args()

    file1 = MachiningFile(args.file1, args.group1)
    file2 = MachiningFile(args.file2, args.group2)

    result = file1.compare_to(file2, args.field, num_segments=args.segments)

    print(f"Comparing '{args.file1}' vs '{args.file2}' on field '{args.field}'")
    print(f"  Mean std (file1): {result['mean_baseline_std']:.4f}")
    print(f"  Mean std (file2): {result['mean_other_std']:.4f}")
    print(f"  % segments where file2 > file1: {result['percent_segments_higher']:.0f}%")


if __name__ == '__main__':
    main()