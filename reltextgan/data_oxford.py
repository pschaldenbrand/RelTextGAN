import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import os
import torch
from textblob import TextBlob
import json
import re
from tqdm import tqdm
import clip
from PIL import Image
import random

from reltextgan.text_functions import alter_sentence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

from sentence_transformers import SentenceTransformer

# coco_annotations_train_fp = 'data/ms_coco/annotations_trainval2014/annotations/captions_train2014.json'
# coco_img_dir = 'data/ms_coco/train2014/train2014/'

oxford_img_dir = 'data/jpg/'
oxford_text_data_dir = 'data/text_c10'

class DataSet():

    def __init__(self, img_size=128):
        self.img_size = img_size
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.bert_model = SentenceTransformer('bert-base-nli-mean-tokens')
        self.changeable_words = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']

        # Load data_dict. Cache it if it's not already made
        data_dict_path = 'data/oxford_data_dict.pickle'
        if os.path.exists(data_dict_path):
            with open(data_dict_path, 'rb') as f:
                self.data_dict = pickle.load(f)
        else:
            self.data_dict = self.create_data_dict()
            with open(data_dict_path, 'wb') as f:
                pickle.dump(self.data_dict, f)
        
        self.oxford_img_ids = list(self.data_dict['captions'].keys())

        # Try to load cached data
        cache_img_fp = 'data/oxford_imgs' + str(self.img_size)  + '.np'
        if not os.path.exists(cache_img_fp):
            self.oxford_imgs, self.img_id_2_index = self.cache_imgs()
        else:
            self.oxford_imgs = np.load(cache_img_fp)
            with open('data/oxford_img_id_2_index' + str(self.img_size) + '.pickle', 'rb') as f:
                self.img_id_2_index = pickle.load(f)


    def create_data_dict(self):
        data_dict = {'file_path': {}, 'captions': {}, 'clip_sentence_encoding': {}, 'image_encoding': {}, 'bert_sentence_encoding' : {}}

        oxford_ids = [os.path.splitext(os.path.basename(filepath))[0] for filepath in glob.iglob(oxford_img_dir + '/*.jpg')]

        # Get Captions
        for img_id in tqdm(oxford_ids, desc="Getting captions"):
            text_fn = next(glob.iglob(oxford_text_data_dir + '/**/' + img_id + '.txt'))
            lines = []
            with open(text_fn, 'r') as f:
                for caption in f:
                    caption = caption.rstrip("\n") # strip newline off
                    # Check if the caption containes one of the changeable words!!
                    caption_words = re.sub(r'[^\w\s]','',caption).split(' ')
                    if not any(word in caption_words for word in self.changeable_words):
                        continue
                    
                    try:
                        clip.tokenize([caption])
                    except:
                        continue

                    if img_id in data_dict['captions'].keys():
                        data_dict['captions'][img_id] += [caption]
                    else:
                        data_dict['captions'][img_id] = [caption]

                    data_dict['file_path'][img_id] = os.path.join(oxford_img_dir, img_id + '.jpg')

        oxford_img_ids = list(data_dict['captions'].keys())

        # Extract features from each image
        with torch.no_grad():
            for i in tqdm(range(len(oxford_img_ids)), desc="Extracting CLIP Features"):
                img_id = oxford_img_ids[i]

                # Text Features
                tokenized_text =  clip.tokenize(data_dict['captions'][img_id]).to(device)
                data_dict['clip_sentence_encoding'][img_id] = self.clip_model.encode_text(tokenized_text).detach().cpu()

                data_dict['bert_sentence_encoding'][img_id] = self.bert_model.encode(data_dict['captions'][img_id])
                
                # Image Features
                image = self.preprocess(Image.open(data_dict['file_path'][img_id])).unsqueeze(0).to(device)
                data_dict['image_encoding'][img_id] = self.clip_model.encode_image(image).detach().cpu()
        return data_dict

    def cache_imgs(self):
        imgs = np.empty((len(self.oxford_img_ids), self.img_size, self.img_size, 3), dtype=np.uint8)
        img_id_2_index = {}
        for i in tqdm(range(len(self.oxford_img_ids)), desc="Cacheing Images"):
            img_id = self.oxford_img_ids[i]
            img = cv2.imread(self.data_dict['file_path'][img_id])[:,:,::-1]
            img = cv2.resize(img, (self.img_size,self.img_size))
            imgs[i] = img
            img_id_2_index[img_id] = i
        with open('data/oxford_imgs' + str(self.img_size)  + '.np', 'wb') as f:
            np.save(f, imgs)
        with open('data/oxford_img_id_2_index' + str(self.img_size)  + '.pickle', 'wb') as f:
            pickle.dump(img_id_2_index, f)
        return imgs, img_id_2_index

    def get_batch(self, ids, batch_size=8, device='cpu', enc_type='clip_sentence_encoding'):
        batch_ids = random.sample(ids, batch_size)

        emb_size = 512 if enc_type=='clip_sentence_encoding' else 768

        imgs = torch.ones((batch_size, 3, self.img_size, self.img_size), dtype=torch.float32, device=device)
        img_emb = torch.ones((batch_size, 512), dtype=torch.float32, device=device)
        sent_emb = torch.ones((batch_size, emb_size), dtype=torch.float32, device=device)
        
        sentences = []
        alt_sentences = []
        i = 0
        for img_id in batch_ids:
            img = self.oxford_imgs[self.img_id_2_index[img_id]] / 255.
            imgs[i] = torch.from_numpy(img.transpose((2,0,1)))

            img_emb[i] = self.data_dict['image_encoding'][img_id]
            
            lines = self.data_dict['captions'][img_id]
            cap_ind = np.random.randint(len(lines))
            line = lines[cap_ind] # Randomly take one of the 10
            sentences.append(line)
            
            sent_emb[i] = self.data_dict[enc_type][img_id][cap_ind] if enc_type=='clip_sentence_encoding' else \
                    torch.from_numpy(self.data_dict[enc_type][img_id][cap_ind])
            
            alt_sentence = alter_sentence(line, self.changeable_words)
            alt_sentences.append(alt_sentence)
            
            i += 1

        with torch.no_grad():
            if enc_type=='clip_sentence_encoding':
                tokenized_text =  clip.tokenize(alt_sentences).to(device)
                alt_sent_emb = self.clip_model.encode_text(tokenized_text).detach()
            else:
                alt_sent_emb = torch.from_numpy(self.bert_model.encode(alt_sentences)).to(device)
        
        return imgs, img_emb.float(), sent_emb.float(), alt_sent_emb.float(), \
               sentences, alt_sentences