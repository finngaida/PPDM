# Note: this script assumes a working server setup already, refer to server_setup.sh for hints on how to get started on a Cloud
echo "WARNING! Don't attempt to actually run this script, it's just to aid in setting up a working environment."
exit 0

# ready to get the code
git clone --recursive https://github.com/finngaida/PPDM.git
cd PPDM

# let's create our environment
conda create -n PPDM python=3.7
# and install the requirements (might again take some minutes)
pip install -r requirements.txt

# need to download some more preprocessed data
PPDM_DATA_PATH=/media/data/ppdm # set to somewhere convenient
# Preprocessed HICO annotations
gdrive --service-account <account.json> download 1WI-gsNLS-t0Kh8TVki1wXqc3y2Ow1f2R --path $PPDM_DATA_PATH --recursive
gdrive --service-account <account.json> download 1le4aziSn_96cN3dIPCYyNsBXJVDD8-CZ --path $PPDM_DATA_PATH
# pretrained models
gdrive --service-account <account.json> download 1b-_sjq1Pe_dVxt5SeFmoadMfiPTPZqpz --path $PPDM_DATA_PATH/models
gdrive --service-account <account.json> download 1pl_-ael8wERdUREEnaIfqOV_VF2bEVRT --path $PPDM_DATA_PATH/models
gdrive --service-account <account.json> download 1-5bT5ZF8bXriJ-wAvOjJFrBLvZV2-mlV --path $PPDM_DATA_PATH/models

# make a bunch of links
mkdir Dataset/hico_det
ln -s /media/data/hico/images Datasets/hico_det/images
ln -s /media/data/ppdm/hico/annotations Datasets/hico_det/annotations
ln -s /media/data/ppdm/models models

# need to install some shit for NMS
cd src/lib/models/networks/DCNv2
sh make.sh # might need to `brew install ninja` for this
python setup.py build_ext # might need to `sudo chown -R fga .` for this
sudo python setup.py install

# and now we're actually ready to go. Choose your fighter
cd src
python main.py hoidet \
    --batch_size 112 \
    --master_batch 7 \
    --lr 4.5e-4 \
    --gpus -1 \
    --num_workers 16  \
    --load_model /media/data/PDDM/models/ctdet_coco_dla_2x.pth \
    --image_dir images/train2015 \
    --dataset hico \
    --exp_id hoidet_hico_dla