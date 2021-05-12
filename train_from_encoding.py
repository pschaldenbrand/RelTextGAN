import cv2
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import argparse
import datetime
import random
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter

from reltextgan.data import DataSet
from reltextgan.text_functions import sentence_diff
from reltextgan.reltextgan import ClipEncodings2Image, RelTextDiscriminator, Discriminator, ImageToEncoding, get_n_params

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

parser = argparse.ArgumentParser(description='Paint by Relaxation')

parser.add_argument('--resume', type=str, default=None, help='Directory path of models to resume training')
parser.add_argument('--image_size', type=int, default=256, help='Image Dimensions. Used for both height and width, sorry.')
parser.add_argument('--batch_size', type=int, default=2, help='Batch Size')
parser.add_argument('--lr', type=float, default=5e-6, help='Learning Rate')
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

print('{:d} Training examples.  {:d} validation, and {:d} test'.format(len(train_ids), len(val_ids), len(test_ids)))

emb_size = 512 if args.enc_type=='clip_sentence_encoding' else 768

# lil test
imgs, img_emb, sent_emb, alt_sent_emb, sentences, alt_sentences = data.get_batch(train_ids, device=device)


gen = ClipEncodings2Image(ngf=512).to(device)
d_real = Discriminator(args.image_size, nc=3, ndf=32).to(device)
img2enc = ImageToEncoding(img_size=args.image_size, encoding_size=emb_size).to(device)

if not os.path.exists('models'): os.mkdir('models')

if args.resume is not None:
    model_dir = args.resume
    print('Resuming models in: ', model_dir)
    gen.load_state_dict(torch.load(os.path.join(model_dir, 'gen.pt')))
    d_real.load_state_dict(torch.load(os.path.join(model_dir, 'd_real.pt')))
    img2enc.load_state_dict(torch.load(os.path.join(model_dir, 'img2enc.pt')))

print('\nParameter Counts')
print('gen           {:,}'.format(get_n_params(gen)))
print('d_real        {:,}'.format(get_n_params(d_real)))
print('img2enc       {:,}'.format(get_n_params(img2enc)))


date_and_time = datetime.datetime.now()
run_name = str(date_and_time.strftime("%m_%d__%H_%M_%S"))
"""https://github.com/milesial/Pytorch-UNet/blob/master/train.py"""
writer = SummaryWriter(log_dir='logdir_from_encoding/{}'.format(run_name))
global_step = 0


batch_size = args.batch_size
lr = args.lr

g_opt = optim.RMSprop(gen.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
d_real_opt = optim.RMSprop(d_real.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
img2enc_opt = optim.RMSprop(img2enc.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)

cycle_criterion = nn.L1Loss()
d_criterion = nn.BCELoss()
g_criterion = nn.BCELoss()

real_label = torch.full((batch_size,), 1., dtype=torch.float, device=device)
fake_label = torch.full((batch_size,), 0., dtype=torch.float, device=device)

up_img = nn.Upsample(size=224, mode='bilinear', align_corners=False)

for epoch in range(666):
    # d_align.train(), 
    gen.train(), d_real.train(), img2enc.train()

    for i in tqdm(range(0, int(len(train_ids)/batch_size)), desc="Epoch {}".format(epoch)):
        imgs, img_emb, encodings, alt_encodings, sentences, alt_sentences \
                = data.get_batch(train_ids, batch_size=batch_size, device=device)
        
        d_real.zero_grad(), gen.zero_grad(), img2enc.zero_grad()
        
        imgs_trans = gen(img_emb, encodings-alt_encodings)
        imgs_trans_emb = data.clip_model.encode_image(up_img(imgs_trans)) # TODO: do these need to be resized?
        imgs_ret = gen(imgs_trans_emb, alt_encodings-encodings)
        imgs_ret_emb = data.clip_model.encode_image(up_img(imgs_ret)) # TODO: do these need to be resized?

        # D Real or fake image
        #d_out = d_real(imgs).view(-1)
        real_imgs, _, _, _, _, _ = data.get_batch(train_ids, enc_type=args.enc_type, batch_size=batch_size, device=device)
        d_out = d_real(real_imgs).view(-1)
        l_d_real_real = d_criterion(d_out, real_label.clone().detach())
        
        d_out = d_real(imgs_trans.detach()).view(-1)
        l_d_real_fake = d_criterion(d_out, fake_label.clone().detach())

        l_d_real = l_d_real_real + l_d_real_fake
        l_d_real.backward()

        d_real_opt.step()
        
        # Img2enc learn how to convert img to encoding
        img2enc_out = img2enc(imgs)
        l_img2enc = cycle_criterion(img2enc_out, encodings)
        l_img2enc.backward()

        img2enc_opt.step()

        # G losses
        l_cycle = cycle_criterion(img_emb, imgs_ret_emb)

        g_d_real_output = d_real(imgs_trans).view(-1)
        l_g_real = g_criterion(g_d_real_output, torch.full((batch_size,), 1., dtype=torch.float, device=device))

        # g_img2enc_output = img2enc(imgs_trans)
        imgs_trans_resized = nn.functional.upsample(imgs_trans, size=(224, 224))
        g_img2enc_output = data.clip_model.encode_image(imgs_trans_resized)
        l_g_img2enc = cycle_criterion(g_img2enc_output, alt_encodings)
        
        l_g_total = l_cycle + l_g_real + l_g_img2enc
        
        l_g_total.backward()

        g_opt.step()
        
        writer.add_scalar('Loss/l_img2enc', l_img2enc.item(), global_step)
        writer.add_scalar('Loss/l_d_real_real', l_d_real_real.item(), global_step)
        writer.add_scalar('Loss/l_d_real_fake', l_d_real_fake.item(), global_step)
        writer.add_scalar('Loss/l_g_real', l_g_real.item(), global_step)
        writer.add_scalar('Loss/l_g_img2enc', l_g_img2enc.item(), global_step)
        writer.add_scalar('Loss/l_cycle', l_cycle.item(), global_step)
        writer.add_scalar('Loss/l_g_total', l_g_total.item(), global_step)
        
        global_step += 1

        if i % 200 == 0:
            gen.eval()
            with torch.no_grad():
                imgs, img_emb, encodings, alt_encodings, sentences, alt_sentences \
                    = data.get_batch(test_ids, batch_size=batch_size, device=device)

                trans_imgs = gen(img_emb, encodings-alt_encodings)
                trans_imgs_emb = data.clip_model.encode_image(up_img(imgs_trans)) # TODO: do these need to be resized?
                cycle_imgs = gen(trans_imgs_emb, alt_encodings-encodings)

                writer.add_images('test/images', torch.clamp(imgs, 0,1), global_step)
                writer.add_images('test/translated_images', torch.clamp(trans_imgs, 0,1), global_step)
                writer.add_images('test/translated_back', torch.clamp(cycle_imgs, 0,1), global_step)
                writer.add_text('test/sentence_with_changes', '\n\n'.join(sentence_diff(sentences, alt_sentences, data.changeable_words)), global_step)
            
            gen.train()
        
        if i%2000 == 0:
            model_dir = 'models/' + run_name + '/'
            if not os.path.exists(model_dir): os.mkdir(model_dir)
            torch.save(gen.state_dict(), model_dir + 'gen.pt')
            # torch.save(d_align.state_dict(), model_dir + 'd_align.pt')
            torch.save(d_real.state_dict(), model_dir + 'd_real.pt')
            torch.save(img2enc.state_dict(), model_dir + 'img2enc.pt')

writer.close()