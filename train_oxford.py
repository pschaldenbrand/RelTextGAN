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

from reltextgan.data_oxford import DataSet
from reltextgan.text_functions import sentence_diff
from reltextgan.reltextgan import RelTextGenerator, RelTextDiscriminator, Discriminator, ImageToEncoding, get_n_params, AlignmentDiscriminator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Paint by Relaxation')

parser.add_argument('--resume', type=str, default=None, help='Directory path of models to resume training')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size')
parser.add_argument('--image_size', type=int, default=256, help='Image Dimensions. Used for both height and width, sorry.')
parser.add_argument('--lr', type=float, default=5e-6, help='Learning Rate')
parser.add_argument('--enc_type', type=str, default='clip_sentence_encoding', help='clip_sentence_encoding or bert_sentence_encoding')
parser.add_argument('--other_way', type=bool, default=False)

args = parser.parse_args()

data = DataSet(img_size=args.image_size)

oxford_img_ids = data.oxford_img_ids

test_prop = 0.1
val_prop = 0.1
train_prop = 0.8

random.seed(0)
random.shuffle(oxford_img_ids)

n = len(oxford_img_ids)

train_ids = oxford_img_ids[:int(n*train_prop)]
val_ids = oxford_img_ids[int(n*train_prop):int(n*train_prop) + int(n*val_prop)]
test_ids = oxford_img_ids[int(n*train_prop) + int(n*val_prop):]

print('{:d} Training examples.  {:d} validation, and {:d} test'.format(len(train_ids), len(val_ids), len(test_ids)))

emb_size = 512 if args.enc_type=='clip_sentence_encoding' else 768

# lil test
imgs, img_emb, sent_emb, alt_sent_emb, sentences, alt_sentences = data.get_batch(train_ids, device=device, enc_type=args.enc_type)

gen = RelTextGenerator(emb_size, args.image_size, ngf=84).to(device)
d_real = Discriminator(args.image_size, nc=3, ndf=32).to(device)
img2enc = ImageToEncoding(img_size=args.image_size, encoding_size=emb_size).to(device)
d_align = AlignmentDiscriminator(emb_size).to(device)

if not os.path.exists('models_oxford'): os.mkdir('models_oxford')

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
print('d_align       {:,}'.format(get_n_params(d_align)))

date_and_time = datetime.datetime.now()
run_name = str(date_and_time.strftime("%m_%d__%H_%M_%S"))
"""https://github.com/milesial/Pytorch-UNet/blob/master/train.py"""
writer = SummaryWriter(log_dir='logdir_oxford/{}'.format(run_name))
global_step = 0


batch_size = args.batch_size
lr = args.lr

g_opt = optim.RMSprop(gen.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
d_real_opt = optim.RMSprop(d_real.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
img2enc_opt = optim.RMSprop(img2enc.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
d_align_opt = optim.RMSprop(d_align.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)


cycle_criterion = nn.L1Loss()
d_criterion = nn.BCELoss()
g_criterion = nn.BCELoss()

real_label = torch.full((batch_size,), 1., dtype=torch.float, device=device)
fake_label = torch.full((batch_size,), 0., dtype=torch.float, device=device)


for epoch in range(666):
    gen.train(), d_real.train(), img2enc.train(), d_align.train()

    for i in tqdm(range(0, int(len(train_ids)/batch_size)), desc="Epoch {}".format(epoch)):
        
        d_real.zero_grad(), gen.zero_grad(), img2enc.zero_grad(), d_align.zero_grad()

        # if args.other_way:
        #     imgs0, _, encodings0, _, _, _  = data.get_batch(train_ids, enc_type=args.enc_type, batch_size=batch_size, device=device)
        #     imgs1, _, encodings1, _, _, _  = data.get_batch(train_ids, enc_type=args.enc_type, batch_size=batch_size, device=device)
            
        #     imgs_trans = gen(imgs0, encodings0-encodings1)
        #     imgs_ret = gen(imgs_trans, encodings1-encodings0)

        #     # D Real or fake image
        #     real_imgs, _, _, _, _, _ = data.get_batch(train_ids, enc_type=args.enc_type, batch_size=batch_size, device=device)
        #     d_out = d_real(real_imgs).view(-1)
        #     l_d_real_real = d_criterion(d_out, real_label.clone().detach())
            
        #     d_out = d_real(imgs_trans.detach()).view(-1)
        #     l_d_real_fake = d_criterion(d_out, fake_label.clone().detach())
            
        #     l_d_real = l_d_real_real + l_d_real_fake #+ l_d_real_ret_fake
        #     l_d_real.backward()

        #     d_real_opt.step()

        #     # G losses
        #     l_cycle = cycle_criterion(imgs_ret, imgs)


        #     g_d_real_output = d_real(imgs_trans).view(-1)
        #     l_g_real = g_criterion(g_d_real_output, torch.full((batch_size,), 1., dtype=torch.float, device=device))

        #     imgs_trans_resized = nn.functional.upsample(imgs_trans, size=(224, 224))
        #     g_img2enc_output = data.clip_model.encode_image(imgs_trans_resized)
        #     l_g_img2enc = cycle_criterion(g_img2enc_output, encodings1)

        # else:
        imgs, img_emb, encodings, alt_encodings, sentences, alt_sentences \
                = data.get_batch(train_ids, enc_type=args.enc_type, batch_size=batch_size, device=device)
        imgs1, imgs1_emb, encodings1, imgs1_text_emb, _, _  = data.get_batch(train_ids, enc_type=args.enc_type, batch_size=batch_size, device=device)
    
        imgs_trans = gen(imgs, encodings-alt_encodings)
        imgs_ret = gen(imgs_trans, alt_encodings-encodings)

        imgs_trans_resized = nn.functional.upsample(imgs_trans, size=(224, 224))
        imgs_trans_encoded = data.clip_model.encode_image(imgs_trans_resized)

        # D Real or fake image
        real_imgs, _, _, _, _, _ = data.get_batch(train_ids, enc_type=args.enc_type, batch_size=batch_size, device=device)
        d_out = d_real(real_imgs).view(-1)
        # d_out = d_real(imgs).view(-1)
        l_d_real_real = d_criterion(d_out, real_label.clone().detach())
        
        d_out = d_real(imgs_trans.detach()).view(-1)
        l_d_real_fake = d_criterion(d_out, fake_label.clone().detach())
        
        l_d_real = l_d_real_real + l_d_real_fake #+ l_d_real_ret_fake
        l_d_real.backward()

        d_real_opt.step()

        # D align
        d_align_out = d_align(img_emb.detach(), imgs1_emb.detach(), encodings-imgs1_text_emb).view(-1)
        l_d_align = d_criterion(d_align_out, real_label.clone().detach())

        d_align_out = d_align(img_emb.detach(), imgs_trans_encoded.detach(), encodings-alt_encodings).view(-1)
        l_d_align += d_criterion(d_align_out, fake_label.clone().detach())

        # Wrong triplets
        _, imgs2_imgenc, imgs2_textenc, _, _, _  = data.get_batch(train_ids, enc_type=args.enc_type, batch_size=batch_size, device=device)
        d_align_out = d_align(imgs2_imgenc.detach(), imgs1_emb.detach(), encodings-imgs1_text_emb).view(-1)
        l_d_align += d_criterion(d_align_out, fake_label.clone().detach())

        d_align_out = d_align(img_emb.detach(), imgs1_emb.detach(), imgs2_textenc-imgs1_text_emb).view(-1)
        l_d_align += d_criterion(d_align_out, fake_label.clone().detach())

        d_align_out = d_align(img_emb.detach(), imgs1_emb.detach(), encodings-imgs2_textenc).view(-1)
        l_d_align += d_criterion(d_align_out, fake_label.clone().detach())

        d_align_out = d_align(img_emb.detach(), imgs2_imgenc.detach(), encodings-imgs1_text_emb).view(-1)
        l_d_align += d_criterion(d_align_out, fake_label.clone().detach())
        
        l_d_align.backward()

        d_align_opt.step()

        if args.enc_type =='bert_sentence_encoding':
            # Img2enc learn how to convert img to encoding
            img2enc_out = img2enc(imgs)
            l_img2enc = cycle_criterion(img2enc_out, encodings)
            l_img2enc.backward()
            img2enc_opt.step()

        # G losses
        l_cycle = cycle_criterion(imgs_ret, imgs)

        d_align_out = d_align(img_emb, imgs_trans_encoded, encodings-alt_encodings).view(-1)
        l_g_align = d_criterion(d_align_out, real_label.clone().detach())

        g_d_real_output = d_real(imgs_trans).view(-1)
        l_g_real = g_criterion(g_d_real_output, torch.full((batch_size,), 1., dtype=torch.float, device=device))

        if args.enc_type=='clip_sentence_encoding':
            l_g_img2enc = cycle_criterion(imgs_trans_encoded, alt_encodings)
        else:
            g_img2enc_output = img2enc(imgs_trans)
            l_g_img2enc = cycle_criterion(g_img2enc_output, alt_encodings)
        
        # Ensure that translated is different from original ##if enc-diff is non-zero
        # l_g_diff = 1 - cycle_criterion(imgs_trans, imgs)
        l_g_same = cycle_criterion(imgs, gen(imgs, encodings-encodings))

        # l_g_total = l_cycle + l_g_real + l_g_img2enc + l_g_align + l_g_same#+ l_g_diff
        l_g_total = l_cycle + l_g_real + 0.5*l_g_img2enc + 0.2*l_g_align + 0.2*l_g_same#+ l_g_diff
        
        l_g_total.backward()

        g_opt.step()
        
        if args.enc_type =='bert_sentence_encoding': writer.add_scalar('Loss/l_img2enc', l_img2enc.item(), global_step)
        writer.add_scalar('Loss/l_d_real_real', l_d_real_real.item(), global_step)
        writer.add_scalar('Loss/l_d_real_fake', l_d_real_fake.item(), global_step)
        writer.add_scalar('Loss/l_g_real', l_g_real.item(), global_step)
        writer.add_scalar('Loss/l_g_img2enc', l_g_img2enc.item(), global_step)
        writer.add_scalar('Loss/l_cycle', l_cycle.item(), global_step)
        writer.add_scalar('Loss/l_g_total', l_g_total.item(), global_step)
        writer.add_scalar('Loss/l_g_align', l_g_align.item(), global_step)
        writer.add_scalar('Loss/l_d_align', l_d_align.item(), global_step)
        writer.add_scalar('Loss/l_g_same', l_g_same.item(), global_step)
        # writer.add_scalar('Loss/l_g_diff', l_g_diff.item(), global_step)
        
        global_step += 1

        if i % 200 == 0:
            gen.eval()
            imgs, img_emb, encodings, alt_encodings, sentences, alt_sentences \
                = data.get_batch(test_ids, enc_type=args.enc_type, batch_size=batch_size, device=device)

            trans_imgs = gen(imgs, encodings-alt_encodings)
            cycle_imgs = gen(trans_imgs, alt_encodings-encodings)

            writer.add_images('test/images', torch.clamp(imgs, 0,1), global_step)
            writer.add_images('test/translated_images', torch.clamp(trans_imgs, 0,1), global_step)
            writer.add_images('test/translated_back', torch.clamp(cycle_imgs, 0,1), global_step)
            writer.add_text('test/sentence_with_changes', '\n\n'.join(sentence_diff(sentences, alt_sentences, data.changeable_words)), global_step)
            
            gen.train()
        
        if i%2000 == 0:
            model_dir = 'models_oxford/' + run_name + '/'
            if not os.path.exists(model_dir): os.mkdir(model_dir)
            torch.save(gen.state_dict(), model_dir + 'gen.pt')
            torch.save(d_real.state_dict(), model_dir + 'd_real.pt')
            torch.save(img2enc.state_dict(), model_dir + 'img2enc.pt')

writer.close()