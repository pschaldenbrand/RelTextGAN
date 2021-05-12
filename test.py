import cv2
import numpy as np
import os
import torch
from torch import optim
import torch.nn as nn
import argparse
import datetime
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from reltextgan.data import DataSet
from reltextgan.text_functions import sentence_diff
from reltextgan.reltextgan import RelTextGenerator, RelTextDiscriminator, Discriminator, ImageToEncoding, get_n_params


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Paint by Relaxation')

parser.add_argument('--model', type=str, default=None, help='Directory path of models to resume training')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size')
parser.add_argument('--image_size', type=int, default=256, help='Image Dimensions. Used for both height and width, sorry.')
parser.add_argument('--enc_type', type=str, default='clip_sentence_encoding', help='clip_sentence_encoding or bert_sentence_encoding')

args = parser.parse_args()

data = DataSet(img_size=args.image_size)

coco_img_ids = data.coco_img_ids

test_prop = 0.1
val_prop = 0.1
train_prop = 0.8

random.seed(0)
random.shuffle(coco_img_ids)

n = len(coco_img_ids)

train_ids = coco_img_ids[:int(n*train_prop)]
val_ids = coco_img_ids[int(n*train_prop):int(n*train_prop) + int(n*val_prop)]
test_ids = coco_img_ids[int(n*train_prop) + int(n*val_prop):]

emb_size = 512 if args.enc_type=='clip_sentence_encoding' else 768

gen = RelTextGenerator(emb_size, args.image_size, ngf=84).to(device)

model_dir = args.model
print('Resuming models in: ', model_dir)
gen.load_state_dict(torch.load(os.path.join(model_dir, 'gen.pt'), map_location=device))

batch_size = args.batch_size

output_dir = 'test_results'
if not os.path.exists(output_dir): os.mkdir(output_dir)

for i in range(50):
    gen.eval()
    imgs, img_emb, encodings, alt_encodings, sentences, alt_sentences \
        = data.get_batch(test_ids, enc_type=args.enc_type, batch_size=batch_size, device=device)

    with torch.no_grad():
        trans_imgs = gen(imgs, encodings-alt_encodings)
        cycle_imgs = gen(trans_imgs, alt_encodings-encodings)

    fig, ax = plt.subplots(1,3, figsize=(10,20))
        
    ax[0].imshow(torch.clamp(imgs[0], 0,1).detach().cpu().numpy().transpose(1,2,0))
    ax[1].imshow(np.clip(torch.clamp(trans_imgs[0], 0,1).detach().cpu().numpy().transpose(1,2,0), 0, 1))
    ax[2].imshow(np.clip(torch.clamp(cycle_imgs, 0,1)[0].detach().cpu().numpy().transpose(1,2,0), 0, 1))
    
    ax[1].title.set_text(sentence_diff(sentences, alt_sentences, data.changeable_words))
    
    ax[0].set_xticks([]), ax[0].set_yticks([]), ax[0].set_xlabel('Original')
    ax[1].set_xticks([]), ax[1].set_yticks([]), ax[1].set_xlabel('Modified')
    ax[2].set_xticks([]), ax[2].set_yticks([]), ax[2].set_xlabel('Returned to Original')

    fig.savefig(os.path.join(output_dir, str(i) + '.png'), bbox_inches = 'tight', pad_inches = 0)
    plt.close(fig)

cs = nn.functional.cosine_similarity
cos_real, cos_pred, cos_baseline = [], [], []

for i in tqdm(range(len(test_ids))):
    gen.eval()
    imgs, img_emb, encodings, alt_encodings, sentences, alt_sentences \
        = data.get_batch(train_ids, enc_type=args.enc_type, batch_size=batch_size, device=device)

    with torch.no_grad():
        trans_imgs = gen(imgs, encodings-alt_encodings)
        trans_imgs_resized = nn.functional.upsample(trans_imgs, size=(224, 224))
        trans_imgs_encoded = data.clip_model.encode_image(trans_imgs_resized)

    cos_real.append(cs(img_emb, encodings).detach().cpu().numpy()[0])
    cos_pred.append(cs(trans_imgs_encoded, alt_encodings).detach().cpu().numpy()[0])
    cos_baseline.append(cs(img_emb, alt_encodings).detach().cpu().numpy()[0])

cos_real, cos_pred, cos_baseline = np.array(cos_real, dtype=np.float), np.array(cos_pred, dtype=np.float), np.array(cos_baseline, dtype=np.float)

print('real ', cos_real.mean(), cos_real.std())
print('pred ', cos_pred.mean(), cos_pred.std())
print('base ', cos_baseline.mean(), cos_baseline.std())
from scipy import stats
print('Correlation', stats.pearsonr(cos_real, cos_pred))

plt.hist(cos_real, bins=20, alpha=0.5, label='Real')
plt.hist(cos_pred, bins=20, alpha=0.5, label='Translated')
plt.hist(cos_baseline, bins=20, alpha=0.5, label='Baseline')
plt.legend(loc='upper right')
plt.savefig(os.path.join(output_dir, 'histogram.png'))