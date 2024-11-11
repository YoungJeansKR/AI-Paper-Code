## Contents
ㅇ
<!-- toc -->

- [Computer Vision](#computer-vision)
  - [Classification](#classification)
  - [Object Detection](#object-detection)
  - [Segmentation](#segmentation)
  - [Generative Models](#generative-models)
  - [Face Recognition](#face-recognition)
  - [Temporal Action Localization](#temporal-action-localization)
  - [Novel View Synthesis](#novel-view-synthesis)
- [Natural Language Processing](#natural-language-processing)
  - [Architecture](#architecture)
  - [Parameter Efficient Fine-Tuning](#parameter-efficient-fine-tuning)
- [Vision-Language Models](#vision-language-models)
- [Speech Processing](#speech-processing)
  - [Speech Synthesis](#speech-synthesis)
  - [Voice Conversion](#voice-conversion)
- [Reinforcement Learning](#reinforcement-learning)
- [Explainable AI](#explainable-ai)
- [Adversarial Attack](#adversarial-attack)
- [Self-Supervised Learning](#self-supervised-learning)
- [Miscellaneous](#miscellaneous)
  - [Recommendation Systems](#recommendation-systems)

<!-- tocstop -->

## Computer Vision

### Classification

* LeNet-5(1998) [[PDF](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)/[Code](https://github.com/kyj950514/AI-Paper-Review/blob/main/Classification/LeNet_5(1998).ipynb)]

* AlexNet(2012) [[PDF](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)/[Code](https://github.com/kyj950514/AI-Paper-Review/blob/main/Classification/AlexNet(2012).ipynb)]

* VGGNet(2014) [[PDF](https://arxiv.org/pdf/1409.1556.pdf)/[Code](https://github.com/kyj950514/AI-Paper-Review/blob/main/Classification/VGGnet(2014).ipynb)]

* GoogLeNet(2014) [[PDF](https://arxiv.org/pdf/1409.4842.pdf)/[Code](https://github.com/kyj950514/AI-Paper-Review/blob/main/Classification/GoogLeNet(2014).ipynb)]

* ResNet(2015) [[PDF](https://arxiv.org/pdf/1512.03385.pdf)/[Code](https://github.com/kyj950514/AI-Paper-Review/blob/main/Classification/ResNet(2015).ipynb)]

* DenseNet(2017) [[PDF](https://arxiv.org/pdf/1608.06993.pdf)/[Code](https://github.com/kyj950514/AI-Paper-Review/blob/main/Classification/DenseNet(2017).ipynb)]

* MobileNetV1(2017) [[PDF](https://arxiv.org/pdf/1704.04861.pdf)]

* MobileNetV2(2018) [[PDF](https://arxiv.org/pdf/1801.04381.pdf)]

* MobileNetV3(2019) [[PDF](https://arxiv.org/pdf/1905.02244.pdf)]

* EfficientNet(2019) [[PDF](https://arxiv.org/pdf/1905.11946.pdf)/[Code](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)]

* Vision Transformer(2020) [[PDF](https://arxiv.org/pdf/2010.11929.pdf)/[Code](https://github.com/kyj950514/AI-Paper-Review/blob/main/Classification/ViT(2020).ipynb)]

* Swin Transformer(2021) [[PDF](https://arxiv.org/pdf/2103.14030.pdf)/[Code](https://github.com/microsoft/Swin-Transformer)]

* EfficientNetV2(2021) [[PDF](https://arxiv.org/pdf/2104.00298.pdf)/[Code](https://github.com/google/automl/tree/master/efficientnetv2)]

* CoAtNet(2021) [[PDF](https://arxiv.org/pdf/2106.04803.pdf)/[Code](https://github.com/chinhsuanwu/coatnet-pytorch)]

* Vision Mamba(2024) [[PDF](https://arxiv.org/pdf/2401.09417)/[Code](https://github.com/hustvl/Vim)]

### Object Detection

* R-CNN(2013) [[PDF](https://arxiv.org/pdf/1311.2524.pdf)]

* SPPNet(2014) [[PDF](https://arxiv.org/pdf/1406.4729.pdf)]

* Fast R-CNN(2014) [[PDF](https://arxiv.org/pdf/1504.08083.pdf)/[Code](https://github.com/rbgirshick/fast-rcnn)]

* Faster R-CNN(2015) [[PDF](https://arxiv.org/pdf/1506.01497.pdf)]

* YOLOv1(2016) [[PDF](https://arxiv.org/pdf/1506.02640.pdf)]

* SSD(2016) [[PDF](https://arxiv.org/pdf/1512.02325.pdf)/[Code](https://github.com/weiliu89/caffe/tree/ssd)]

* FPN(2017) [[PDF](https://arxiv.org/pdf/1612.03144.pdf)]

* YOLOv2(2017) [[PDF](https://arxiv.org/pdf/1612.08242.pdf)]

* CornerNet(2019) [[PDF](https://arxiv.org/pdf/1808.01244)]

* CenterNet(2019) [[PDF](https://arxiv.org/pdf/1904.07850.pdf)]

* EfficientDet(2020) [[PDF](https://arxiv.org/pdf/1911.09070.pdf)/[Code](https://github.com/google/automl/tree/master/efficientdetr)]

* DETR(2020) [[PDF](https://arxiv.org/pdf/2005.12872.pdf)/[Code](https://github.com/facebookresearch/detr)]

* Deformable DETR(2021) [[PDF](https://arxiv.org/pdf/2010.04159)/[Code](https://github.com/fundamentalvision/Deformable-DETR)]

* RT-DETR(2024) [[PDF](https://arxiv.org/pdf/2304.08069)/[Code](https://github.com/lyuwenyu/RT-DETR)]

### Segmentation

* DeepLabV1(2014) [[PDF](https://arxiv.org/pdf/1412.7062.pdf)]

* FCN(2015) [[PDF](https://arxiv.org/pdf/1411.4038.pdf)]

* SegNet(2015) [[PDF](https://arxiv.org/pdf/1511.00561.pdf)]

* U-Net(2015) [[PDF](https://arxiv.org/pdf/1505.04597.pdf)]

* DeepLabV2(2016) [[PDF](https://arxiv.org/pdf/1606.00915.pdf)]

* DeepLabV3(2017) [[PDF](https://arxiv.org/pdf/1706.05587.pdf)]

* Mask R-CNN(2017) [[PDF](https://arxiv.org/pdf/1703.06870.pdf)]

* DeepLabV3+(2018) [[PDF](https://arxiv.org/pdf/1802.02611.pdf)]

### Generative Models

* VAE(2013) [[PDF](https://arxiv.org/pdf/1312.6114)]

* GAN(2014) [[PDF](https://arxiv.org/pdf/1406.2661.pdf)]

* CGAN(2014) [[PDF](https://arxiv.org/pdf/1411.1784.pdf)]

* DCGAN(2015) [[PDF](https://arxiv.org/pdf/1511.06434.pdf)]

* Pix2Pix(2016) [[PDF](https://arxiv.org/pdf/1611.07004.pdf)]

* PGGAN(2017) [[PDF](https://arxiv.org/pdf/1710.10196)]

* CycleGAN(2017) [[PDF](https://arxiv.org/pdf/1703.10593)]

* SytleGAN(2014) [[PDF](https://arxiv.org/pdf/1812.04948f)]

* DDPM(2020) [[PDF](https://arxiv.org/pdf/2006.11239)]

* DDIM(2020) [[PDF](https://arxiv.org/pdf/2010.02502)]

### Face Recognition

* DeepFace(2014) [[PDF](https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf)]

* FaceNet(2015) [[PDF](https://arxiv.org/pdf/1503.03832.pdf)]

* ArkFace(2018) [[PDF](https://arxiv.org/pdf/1801.07698.pdf)]

### Temporal Action Localization

* C3D(2015) [[PDF](https://arxiv.org/pdf/1412.0767)]

* S-CNN(2016) [[PDF](https://arxiv.org/pdf/1601.02129)]

* TAD(2017) [[PDF](https://arxiv.org/pdf/1704.06228)]

* RTD-Net(2021) [[PDF](https://arxiv.org/pdf/2102.01894)]

* E2E-TAD(2022) [[PDF](https://arxiv.org/pdf/2204.02932)]

* TADTR(2022) [[PDF](https://arxiv.org/pdf/2106.10271)]

* ReAct(2022) [[PDF](https://arxiv.org/pdf/2207.07097)]

### Novel View Synthesis

* NeRF(2020) [[PDF](https://arxiv.org/pdf/2003.08934)]

## Natural Language Processing

### Architecture

* LSTM(1997) [[PDF](https://www.bioinf.jku.at/publications/older/2604.pdf)]

* Bi-LSTM(1997) [[PDF](https://deeplearning.cs.cmu.edu/S24/document/readings/Bidirectional%20Recurrent%20Neural%20Networks.pdf)]

* Seq2Seq(2014) [[PDF](https://arxiv.org/pdf/1409.3215.pdf)]

* GRU(2014) [[PDF](https://arxiv.org/pdf/1412.3555.pdf)]

* Attention(2014) [[PDF](https://arxiv.org/pdf/1508.04025.pdf)]

* Transforemr(2017) [[PDF](https://arxiv.org/pdf/1706.03762.pdf)]

* BERT(2018) [[PDF](https://arxiv.org/pdf/1810.04805.pdf)]

* GPT(2018) [[PDF](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)]

* Mamba(2023) [[PDF](https://arxiv.org/pdf/2312.00752)]

* Jamba(2024) [[PDF](https://arxiv.org/pdf/2403.19887)/[Code](https://huggingface.co/ai21labs/Jamba-v0.1)]

### Parameter Efficient Fine-Tuning

* Adapter(2019) [[PDF](https://arxiv.org/pdf/1902.00751)]

* Prefix-Tuning(2021) [[PDF](https://arxiv.org/pdf/2101.00190)]

* LoRA(2021) [[PDF](https://arxiv.org/pdf/2106.09685)]

* MAM Adapter(2022) [[PDF](https://arxiv.org/pdf/2110.04366)/[Code](https://github.com/jxhe/unify-parameter-efficient-tuning)]

* QLoRA(2024) [[PDF](https://arxiv.org/pdf/2305.14314)]

## Vision-Language Models

* CLIP(2021) [[PDF](https://arxiv.org/pdf/2103.00020)/[Code](https://github.com/OpenAI/CLIP)]

* ALIGN(2021) [[PDF](https://arxiv.org/pdf/2102.05918)]

* GLIP(2021) [[PDF](https://arxiv.org/pdf/2112.03857)/[Code](https://github.com/microsoft/GLIP)]

* BLIP(2022) [[PDF](https://arxiv.org/pdf/2201.12086)/[Code](https://github.com/salesforce/BLIP)]

* GLIPv2(2022) [[PDF](https://arxiv.org/abs/2206.05836)/[Code](https://github.com/microsoft/GLIP)]

## Speech Processing

### Speech Synthesis

* WaveNet(2016) [[PDF](https://arxiv.org/pdf/1609.03499)]

* Tacotron(2017) [[PDF](https://arxiv.org/pdf/1703.10135)]

* Tacotron2(2018) [[PDF](https://arxiv.org/pdf/1712.05884)]

* FastSpeech(2019) [[PDF](https://arxiv.org/pdf/1905.09263)]

* Transformer TTS(2019) [[PDF](https://arxiv.org/pdf/1809.08895)]

* HiFi-GAN(2020) [[PDF](https://arxiv.org/pdf/2010.05646)]

### Voice Conversion

* AutoVC(2019) [[PDF](https://arxiv.org/pdf/1905.05879)]

* DiffSVC(2021) [[PDF](https://arxiv.org/pdf/2105.13871)]

## Reinforcement Learning

* DQN(2013) [[PDF](https://arxiv.org/pdf/1312.5602)]

## Explainable AI

* LRP(2015) [[PDF](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0130140&type=printable)]

* CAM(2016) [[PDF](https://arxiv.org/pdf/1512.04150)]

* Grad-CAM(2017) [[PDF](https://arxiv.org/pdf/1610.02391)]

## Adversarial Attack

* FGSM(2015) [[PDF](https://arxiv.org/pdf/1412.6572)]

* CW Attack(2017) [[PDF](https://arxiv.org/pdf/1608.04644)]

* PGD Training(2017) [[PDF](https://arxiv.org/pdf/1706.06083)]

* Adversarial Patch(2018) [[PDF](https://arxiv.org/pdf/1712.09665)]

* BPDA(2018) [[PDF](https://arxiv.org/pdf/1802.00420)]

* Pixel DP(2019) [[PDF](https://arxiv.org/pdf/1802.03471)/[Code](https://github.com/columbia/pixeldp)]

## Self-Supervised Learning

* DeepCluster(2018) [[PDF](https://arxiv.org/pdf/1807.05520)]

* NPID(2018) [[PDF](https://arxiv.org/pdf/1805.01978)]

* SimCLR(2020) [[PDF](https://arxiv.org/pdf/2002.05709)]

* MoCo(2020) [[PDF](https://arxiv.org/pdf/1911.05722)]

* BYOL(2020) [[PDF](https://arxiv.org/pdf/2006.07733)/[Code](https://github.com/google-deepmind/deepmind-research/tree/master/byol)]

* DINO(2021) [[PDF](https://arxiv.org/pdf/2104.14294)/[Code](https://github.com/facebookresearch/dino)]

* MAE(2022) [[PDF](https://arxiv.org/pdf/2111.06377)]

## Miscellaneous

### Recommendation Systems

* DNN for Youtube(2016) [[PDF](https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/45530.pdf)]
