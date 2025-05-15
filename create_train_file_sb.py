input_file = "list/XD_Train.list"   # path of the rgb file containing five crop
output_file = "list/XD_SB_Train.list"  # Output file to save modified content

with open(input_file, 'r') as f:
    lines = f.readlines()

modified_lines = []

for line in lines:
    line = line.strip()
    modified_lines.append(line)

    if line.endswith("__4.npy"):
        # Extract filename only
        filename = line.split("/")[-1].replace("__4.npy", "__vggish.npy")
        vgg_path = f"/home/muhammad-liaqat/datasetandfiles/violenceDetection/vggish-features/train/{filename}"
        # Append 5 lines
        modified_lines.extend([vgg_path] * 5)

# Write to output file
with open(output_file, 'w') as f:
    for item in modified_lines:
        f.write(item + '\n')

print(f"Modified file written to {output_file}")
