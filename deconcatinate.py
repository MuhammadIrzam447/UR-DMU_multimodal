import numpy as np
import os

source_dir = "/home/muhammad-liaqat/datasetandfiles/corruptiondata/rgb_corrp/motion_blur_ten/"
output_dir = "/home/muhammad-liaqat/datasetandfiles/corruptiondata/rgb_corrp/motion_corrp/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for file in os.listdir(source_dir):
    if file.endswith(".npy"):
        print(f"Processing file: {file}")
        filepath = os.path.join(source_dir, file)
        data = np.load(filepath)
        data = data.transpose(1, 0, 2)
        base_name = os.path.splitext(file)[0]
        # base_name = base_name[:-4]
        if data.ndim == 3 and data.shape[1] == 10 and data.shape[2] == 1024:
            # Only keep the first 5 crops
            data = data[:, 0:5, :]  # shape: (n, 5, 1024)

            for crop_idx in range(5):
                crop_data = data[:, crop_idx, :]
                new_filename = f"{base_name[:-4]}__{crop_idx}.npy"  # double underscore
                save_path = os.path.join(output_dir, new_filename)
                np.save(save_path, crop_data)
        else:
            print(f"Skipping {file}: Shape is not (n, 10, 1024)")
            print(f"--> Actual shape of {file}: {data.shape}")

print("Processing complete.")
