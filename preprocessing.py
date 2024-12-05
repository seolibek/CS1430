'''

Script to take photos from UTK face dataset and generate the synthetic dataset as required by the Disney model.

Requirements:

    - must be in data/processed/train
    - need to download SAM - look into this can probably install as dependencies.. mayb this should be a colab
    - must be in format person[id]/[age].jpg, where each folder contains images of one person

uh yeah this should totally be colab i think

'''

#first download the pretrained SAM model? https://github.com/yuval-alaluf/SAM
# yeah this should totally be a notebook ill change it after
mkdir pretrained_models
pip install gdown
gdown "https://drive.google.com/u/0/uc?id=1XyumF6_fdAxFmxpFcmPf-q84LU_22EMC&export=download" -O pretrained_models/sam_ffhq_aging.pt
wget "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat"

#Having trained your model or if you're using a pretrained SAM model, you can use scripts/inference.py to run inference on a set of images.
python scripts/inference.py \
--exp_dir=/path/to/experiment \
--checkpoint_path=experiment/checkpoints/best_model.pt \
--data_path=/path/to/test_data \
--test_batch_size=4 \
--test_workers=4 \
--couple_outputs
--target_age= 18, 23, 28, 33, 38, 43, 48, 53, 58, 63, 68, 73, 78, 85