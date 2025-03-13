# ProDet
The official code for paper "[Can We Leave Deepfake Data Behind in Training Deepfake Detector](https://arxiv.org/pdf/2408.17052)" (NIPS2024 poster)

![main_archi.pdf](./main_archi.png)

ProDet is implemented within the framework of [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench). The provided code should be placed in the corresponding folders in DeepfakeBench, and test/train on DeepfakeBench as well. 

You may find the overall-best checkpoint of our method from [Google Drive](https://drive.google.com/drive/folders/16IDcKVqziJ-Qv_IfZOAcYGwdeZAM-x5s?usp=drive_link), which is recommended for the comparing experiments in your own research. 

There is no additional package required beyond DeepfakeBench and this repository, hence you should easily reproduce the training of our paper with an established DeepfakeBench environment as:

```python
python training/train.py 
--detector_path 
./training/config/detector/prodet.yaml 
--train_dataset 
"FaceForensics++"  
--test_dataset 
"FaceForensics++" "Celeb-DF-v2" "DFDCP" 
```

You may also directly use the trained model for evaluation:

```python
python3 training/test.py 
--detector_path ./training/config/detector/prodet.yaml 
--test_dataset "Celeb-DF-v1" "Celeb-DF-v2" "DFDCP" 
--weights_path ./training/weights/ProDet_best.pth
```


If you found our code useful to your research, please cite it as follows:
```bibtex
@article{cheng2024can,
  title={Can We Leave Deepfake Data Behind in Training Deepfake Detector?},
  author={Cheng, Jikang and Yan, Zhiyuan and Zhang, Ying and Luo, Yuhao and Wang, Zhongyuan and Li, Chen},
  journal={arXiv preprint arXiv:2408.17052},
  year={2024}
}

