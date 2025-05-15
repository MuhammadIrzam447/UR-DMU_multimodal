import numpy as np

def compare_and_trim_lists(original_list_path, target_list_path):
    with open(original_list_path, 'r') as f:
        original_lines = [line.strip().split()[0] for line in f.readlines()]
    with open(target_list_path, 'r') as f:
        target_lines = [line.strip().split()[0] for line in f.readlines()]

    assert len(original_lines) == len(target_lines), "List files have different lengths!"

    mismatch_count = 0
    shape_mismatch_count = 0

    for idx, (orig_path, targ_path) in enumerate(zip(original_lines, target_lines)):
        try:
            orig_feat = np.load(orig_path)
            targ_feat = np.load(targ_path)
        except Exception:
            mismatch_count += 1
            continue

        if orig_feat.shape != targ_feat.shape:
            # Check if shapes are compatible for trimming
            if targ_feat.shape[1:] == orig_feat.shape[1:] and targ_feat.shape[0] > orig_feat.shape[0]:
                trimmed_feat = targ_feat[:orig_feat.shape[0]]
                np.save(targ_path, trimmed_feat)  # Overwrite the target file with trimmed feature
            shape_mismatch_count += 1
            mismatch_count += 1

    print("\n=== Comparison Summary ===")
    print(f"Total files compared   : {len(original_lines)}")
    print(f"Shape mismatches       : {shape_mismatch_count}")
    print(f"Total mismatches       : {mismatch_count}")

if __name__ == "__main__":
    original_list = "corrputed_list/test_audio_clean.list"
    target_list = "corrputed_list/test_audio_corrp_babble_3.list"

    # compare_and_trim_lists(original_list, target_list)
