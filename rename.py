import os
import shutil

root_dir = "/Users/seoli/Desktop/CS1430/CS1430/data/inference_results"
output_dir = "/Users/seoli/Desktop/CS1430/CS1430/data/reorganized_dataset"
os.makedirs(output_dir, exist_ok=True)

for age_folder in os.listdir(root_dir):
    age_path = os.path.join(root_dir, age_folder)
    if not os.path.isdir(age_path):
        continue 

    for file_name in os.listdir(age_path):
        if file_name.endswith(".jpg") or file_name.endswith(".chip.jpg"):  
            # Extract the person ID from the file name
            person_id = file_name.split("_")[0]

            # Create a folder for the person ID in the output directory
            person_dir = os.path.join(output_dir, f"person{person_id}")
            os.makedirs(person_dir, exist_ok=True)

            # Construct the new file name with the age
            new_file_name = f"{age_folder}.jpg"

            # Copy the file to the new directory with the new name
            src_path = os.path.join(age_path, file_name)
            dest_path = os.path.join(person_dir, new_file_name)
            shutil.copy(src_path, dest_path)

            print(f"Copied: {src_path} -> {dest_path}")

print("File renaming and organization complete.")
