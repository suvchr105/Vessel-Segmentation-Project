import os
import csv

# Paths to the folders containing the retina images
retina_folder1 = 'images'
retina_folder2 = 'mask_folder'
retina_folder3 = 'output_folder'

# Get the list of file paths in each retina folder
retina_file_paths1 = [os.path.join(retina_folder1, filename) for filename in os.listdir(retina_folder1)]
retina_file_paths2 = [os.path.join(retina_folder2, filename) for filename in os.listdir(retina_folder2)]
retina_file_paths3 = [os.path.join(retina_folder3, filename) for filename in os.listdir(retina_folder3)]

# Combine the file paths from all three folders
combined_file_paths = list(zip(retina_file_paths1, retina_file_paths2, retina_file_paths3))

# Path to save the CSV file
csv_file = 'combined_file_paths.csv'

# Save the list of file paths to a CSV file
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['images', 'maskname', 'vesselname'])
    writer.writerows(combined_file_paths)

print(f"Combined list of file paths saved to {csv_file} successfully.")
