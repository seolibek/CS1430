

'''wait i want to select specific photos from UTKFACE this script will do that.

parse images based on name, 5 total ethniciteis represented, age must be between 18 to 85, gender should be balanced

format:
[age]_[gender]_[race]_[date&time].jpg






POTENTIAL ISSUE: USING UTKFACE ON THE SAM DATASET GIVES REALLY WEIRD IMAGES; MUST PROBABL USE FFHQ - DATASET THEY 
ACTUALLY USED - SO WE DONT GET UNNATRUAL PHOTOS IN DATASET.

IN FFHQ DATSET, AGE AND RACE IS NOT INCLUDED IN METADATA, SO WE CANNOT SAMPLE/RESTRICT THE DATA WE USE



'''

import os
import random

#here's the link to download the full UTK face aligned and cropped dataset: https://www.kaggle.com/datasets/moritzm00/utkface-cropped/data
utkface_path = '/Users/seoli/Desktop/CS1430/CS1430/UTKFace' #put in your path for the utkface you downloaded from aove
output_dir = './data/'

num_images = 2000 #can change this up to you


##########     DO NOT CHANGE BELOW     ############
age_bins = list(range(18, 86, 6)) #because paper is [18,85]
ethnicities = [0, 1, 2, 3, 4] # utk has this separated : 0-white,1-black,2-asian,3-indian,4-others .. 
genders = [0,1]  #0 = M, 1 = F

# #calculates how many images you want to select per group (a combination of age bin, ethnicity, and gender). It ensures that images are evenly distributed across all groups.
images_per_bin = num_images // ((len(age_bins) - 1) * len(ethnicities) * len(genders))
print(images_per_bin)

selected_counts = {}
final_images = []

for filename in os.listdir(utkface_path):
    if len(final_images) >= num_images: 
        break

    try:
        if filename.endswith('.chip.jpg'):
            base_name = filename[:-9]  # Remove the last 9 characters for '.chip.jpg' out of an abundance of caution

        age, gender, ethnicity = map(int, base_name.split('_')[:3])
        print(f"Parsed metadata: Age={age}, Gender={gender}, Ethnicity={ethnicity}")
        
        if 18 <= age <= 85 and ethnicity in ethnicities:
            for i in range(len(age_bins) - 1):
                if age_bins[i] <= age < age_bins[i + 1]:
                    key = (age_bins[i], ethnicity, gender)
                    selected_counts.setdefault(key, 0)
                    if selected_counts[key] < images_per_bin:
                        final_images.append(filename)
                        selected_counts[key] += 1
                    break
    except ValueError:
        print('valueerror')
        continue

os.makedirs(output_dir, exist_ok=True)
for img in final_images:
    src = os.path.join(utkface_path, img)
    dst = os.path.join(output_dir, img)
    os.rename(src, dst)

print(f"Selected {len(final_images)} images for processing.")

#ignore
'''
and then if u wanna try running SAM yourself you could do it like this

git clone https://github.com/yuval-alaluf/SAM.git
cd SAM

either create a new env like following or use one that has these in it:
pip install torch torchvision
conda install -c conda-forge ninja

'''