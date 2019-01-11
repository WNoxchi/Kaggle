print("IMPORTING LIBRARIES..")

from fastai import *
from fastai.vision import *
from utils.utils import open_4_channel
from utils.resnet import Resnet4Channel

def resnet50(pretrained):
    return Resnet4Channel(encoder_depth=50)
# copied from https://github.com/fastai/fastai/blob/master/fastai/vision/learner.py
def _resnet_split(m): return (m[0][6], m[1])

print("CONFIGURING DATA..")

path = Config.data_path()/'proteinatlas'

np.random.seed(42)
src = (ImageItemList.from_csv(path, 'train.csv', folder='train', suffix='.png')
       .random_split_by_pct(0.2)
       .label_from_df(sep=' ', classes=[str(i) for i in range(28)])) # 27 classes

src.train.x.create_func = open_4_channel
src.train.x.open = open_4_channel
src.valid.x.create_func = open_4_channel
src.valid.x.open = open_4_channel

test_ids = list(sorted({fname.split('_')[0] for fname in os.listdir(path/'test')}))
test_fnames = [path/'test'/test_id for test_id in test_ids]

src.add_test(test_fnames, label='0');
src.test.x.create_func = open_4_channel
src.test.x.open = open_4_channel

protein_stats = ([0.08069, 0.05258, 0.05487, 0.08282], [0.13704, 0.10145, 0.15313, 0.13814])

# get_transforms returns tfms for train & valid: https://docs.fast.ai/vision.transform.html#get_transforms
train_tfms, _ = get_transforms(do_flip=True, flip_vert=True, max_rotate=30., max_zoom=1,
                            max_lighting=0.05, max_warp=0.)
size = 256

data = (src.transform((train_tfms, _), size=size)
        .databunch().normalize(protein_stats))

f1_score = partial(fbeta, thresh=0.2, beta=1)

learn = create_cnn(
    data, 
    resnet50, 
    cut=-2, 
    split_on=_resnet_split,
    loss_func=F.binary_cross_entropy_with_logits,
    path=path, 
    metrics=[f1_score]
)

# resize input data to 256x256; lower batch size to 48
bs   = 48
data = (src.transform((train_tfms, _), size=size)
        .databunch(bs=bs).normalize(protein_stats))
learn.data = data

print("TRAINING..")

lr = 3e-3

learn.fit_one_cycle(8, slice(lr))

learn.save(f'stage-1-rn50-datablocks-sz{size}')

print("PREDICTING..")

preds,_ = learn.get_preds(DatasetType.Test)
pred_labels = [' '.join(list([str(i) for i in np.nonzero(row>0.2)[0]])) for row in np.array(preds)]

print("SAVING PREDICTIONS..")

import datetime; date = str(datetime.date.today()).replace('-','')

(path/'submissions').mkdir(exist_ok=True)

subm_msg = f"RN50 Datablocks codealong; fastai {__version__}; size {size}; stage-2; threshold: 20%; date: {date}"
subm_name = f"rn50-datablocks-sz{size}-stg2-th20p-fastai-{__version__}-{date}.csv"

subm_df = pd.DataFrame({'Id':test_ids,'Predicted':pred_labels})
subm_df.to_csv(path/'submissions'/subm_name, header=True, index=False)

print(f"{'='*40}\nTRAINING COMPLETE\n{'='*40}")
      
      
# kaggle competitions submit -c human-protein-atlas-image-classification -f /home/jupyter/.fastai/data/proteinatlas/submissions/rn50-datablocks-sz256-stg2-th20p-fastai-1.0.39-20190111.csv -m "quick"