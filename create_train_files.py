try:
    # Open the input file for reading and the output file for writing
    with open('list/XD_Train.list', 'r') as infile, open('list/XD_audio_Train.list', 'w') as outfile:
        lines_written = 0  # Counter for the number of lines written to the output file

        # Iterate through each line in the input file
        for line in infile:
            # Strip any leading/trailing whitespace characters
            line = line.strip()
            # Check if the line ends with '__0.npy'
            if line.endswith('__0.npy'):
                # Write the line to the output file
                outfile.write(line + '\n')
                lines_written += 1

        if lines_written > 0:
            print(f"Operation successful: {lines_written} lines copied to 'XD_audio_Test.list'.")
        else:
            print("Operation completed, but no lines ending with '__0.npy' were found.")

except FileNotFoundError:
    print("Error: One or both of the input/output files do not exist.")
except Exception as e:
    print(f"An error occurred: {e}")

