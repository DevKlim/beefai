import os
import re

def remove_contributors_line(directory):
    """
    Checks each .txt file in the specified directory and removes the first line
    if it contains the word "Contributors".

    Args:
        directory (str): The path to the directory containing the .txt files.
    """

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)

            try:
                with open(filepath, 'r', encoding='utf-8') as f:  # Specify encoding for potential Unicode issues
                    lines = f.readlines()

                if lines:  # Check if the file isn't empty
                    first_line = lines[0].strip()  # Remove leading/trailing whitespace
                    if "Read More" in first_line:  # Check for "Contributors" in the first line
                        lines = lines[1:]  # Remove the first line

                        with open(filepath, 'w', encoding='utf-8') as f:  # Open in write mode to overwrite
                            f.writelines(lines)  # Write the modified content back to the file
                        print(f"Removed Contributors line from {filename}")
                    else:
                        print(f"No Contributors line found in first line of {filename}")
                else:
                    print(f"{filename} is empty.")


            except Exception as e:
                print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    data_directory = "data\\lyrics"  # Path to the directory containing the .txt files
    remove_contributors_line(data_directory)