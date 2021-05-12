# RelTextGAN

Towards editing images via natural language.  Trained using image captioning data.

## Dependencies
```
$ pip install ftfy regex tqdm
$ pip install -f https://download.pytorch.org/whl/cu110/torch_stable.html torch==1.7.1+cu110 torchvision==0.8.2
$ pip install git+https://github.com/openai/CLIP.git
```

## Data

Please download [Oxford-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) and [MS-COCO](https://cocodataset.org/#download) datasets and place them into a directory named `data/`

## Training

To train COCO model and Oxford model respectively

```
python train.py [--image_size 256] [--batch_size 8] [--enc_type clip_sentence_encoding|bert_sentence_encoding]

python train_oxford.py [--image_size 256] [--batch_size 8] [--enc_type clip_sentence_encoding|bert_sentence_encoding]
```

## Testing

```
python test.py --model [parent location of gen.pt model]
python test_oxford.py  --model [parent location of gen.pt model]
```
