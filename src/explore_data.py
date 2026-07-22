from machining_file import MachiningFile

baseline = MachiningFile('../data/Segmented_Machining_Baseline.mat', 'BaselineCrop')
toolwear = MachiningFile('../data/Segmented_Machining_ToolWear.mat', 'ToolWearCrop')
misalignment = MachiningFile('../data/Segmented_Machining_Misalignment.mat', 'MisalignmentCrop')

result_toolwear = baseline.compare_to(toolwear, 'SpindleX', num_segments=10)
print("Tool Wear comparison:")
print(f"  Mean baseline std: {result_toolwear['mean_baseline_std']:.4f}")
print(f"  Mean fault std:    {result_toolwear['mean_other_std']:.4f}")
print(f"  % segments higher: {result_toolwear['percent_segments_higher']:.0f}%")

result_misalign = baseline.compare_to(misalignment, 'SpindleX', num_segments=10)
print("\nMisalignment comparison:")
print(f"  Mean baseline std: {result_misalign['mean_baseline_std']:.4f}")
print(f"  Mean fault std:    {result_misalign['mean_other_std']:.4f}")
print(f"  % segments higher: {result_misalign['percent_segments_higher']:.0f}%")