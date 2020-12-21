# Note: this script assumes a working server setup already, refer to server_setup.sh for hints on how to get started on a Cloud
echo "WARNING! Don't attempt to actually run this script, it's just to aid in setting up a working environment."
exit 0

# ready to get the code
git clone https://github.com/finngaida/PPDM.git
cd PPDM

# let's create our environment
conda create -n PPDM python=3.7
# and install the requirements (might again take some minutes)
pip install -r requirements.txt

# need to download some more preprocessed data
PPDM_DATA_PATH=/media/data/ppdm # set to somewhere convenient
gdrive --service-account <account.json> download 1WI-gsNLS-t0Kh8TVki1wXqc3y2Ow1f2R --path $PPDM_DATA_PATH
gdrive --service-account <account.json> download 1b-_sjq1Pe_dVxt5SeFmoadMfiPTPZqpz --path $PPDM_DATA_PATH
gdrive --service-account <account.json> download 1pl_-ael8wERdUREEnaIfqOV_VF2bEVRT --path $PPDM_DATA_PATH
gdrive --service-account <account.json> download 1-5bT5ZF8bXriJ-wAvOjJFrBLvZV2-mlV --path $PPDM_DATA_PATH
# make the link to COCO
ln -s /media/data/images Datasets/images
ln -s /media/data/hico Datasets/annotations

# need to install some shit for NMS
cd src/lib/models/networks/DCNv2
python build.py
# NOTE: with CUDA installed you can install the real deal via `sh make.sh`

# and now we're actually ready to go. Choose your fighter
cd src
python main.py  hoidet --batch_size 112 --master_batch 7 --lr 4.5e-4 --gpus 0,1,2,3,4,5,6,7  --num_workers 16  --load_model ../models/ctdet_coco_dla_2x.pth --image_dir images/train2015 --dataset hico --exp_id hoidet_hico_dla