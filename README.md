# BRINet for referring image localization and segmentation
This repository contains code for:

'Bi-directional Relationship Inferring Network for Referring Image Segmentation', CVPR 2020.

'Bidirectional Relationship Inferring Network for Referring Image Localization and Segmentation', TNNLS 2021

If you find this work useful in your research, please consider citing:

```
@inproceedings{hu2020bi,
  title={Bi-directional relationship inferring network for referring image segmentation},
  author={Hu, Zhiwei and Feng, Guang and Sun, Jiayu and Zhang, Lihe and Lu, Huchuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4424--4433},
  year={2020}
}
```
```
@article{feng2021bidirectional,
  title={Bidirectional Relationship Inferring Network for Referring Image Localization and Segmentation},
  author={Feng, Guang and Hu, Zhiwei and Zhang, Lihe and Sun, Jiayu and Lu, Huchuan},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2021},
  publisher={IEEE}
}
```
## Paper link
The paper can be found in [paper](https://github.com/fengguang94/BRINet/tree/master/pdf).


## Code

### Requirement
- Python 3.5
- Tensorflow 1.8
- [PyDenseCRF](https://github.com/lucasb-eyer/pydensecrf)

### Setup
Partial coda and data preparation are borrowed from [TF-phrasecut-public](https://github.com/chenxi116/TF-phrasecut-public). Please follow their instructions to make your setup ready. DeepLab backbone network is based on [TF-deeplab ](https://github.com/chenxi116/TF-deeplab) as well as the pretrained model for initializing weights of our model. 

### Sample code
#### Training
```
python main_cmsa.py -m train -w deeplab -d Gref -t train -g 0 -i 800000
```


#### Testing 
```
python main_cmsa.py -m test -w deeplab -d Gref -t val -g 0 -i 800000
```

