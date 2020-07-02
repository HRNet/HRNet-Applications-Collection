# A collection of HRNet applications
(Please feel freely add your applications if not included)
## Classification, segmentation and detection 
### ImageNet classification
[Deep High-Resolution Representation Learning for Visual Recognition](https://arxiv.org/abs/1908.07919). Jingdong Wang, Ke Sun, Tianheng Cheng, Borui Jiang, Chaorui Deng, Yang Zhao, Dong Liu, Yadong Mu,  Mingkui Tan, Xinggang Wang, Wenyu Liu, and Bin Xiao. TPAMI. 2020. [code](https://github.com/HRNet/HRNet-Image-Classification)

### Semantic segmentation
[Deep High-Resolution Representation Learning for Visual Recognition](https://arxiv.org/abs/1908.07919). Jingdong Wang, Ke Sun, Tianheng Cheng, Borui Jiang, Chaorui Deng, Yang Zhao, Dong Liu, Yadong Mu,  Mingkui Tan, Xinggang Wang, Wenyu Liu, and Bin Xiao. TPAMI. 2020. [code](https://github.com/HRNet/HRNet-Semantic-Segmentation)

[Hierarchical Multi-Scale Attention for Semantic Segmentation](https://arxiv.org/abs/2005.10821). Andrew Tao, Karan Sapra, and Bryan Catanzaro: CoRR abs/2005.10821 (2020). [Rank #1, cityscapes benchmark](https://www.cityscapes-dataset.com/benchmarks/#pixel-level-results)

[MSeg: A Composite Dataset for Multi-domain Semantic Segmentation](http://vladlen.info/papers/MSeg.pdf). John Lambert, Zhuang Liu, Ozan Sener, James Hays, and Vladlen Koltun. CVPR 2020. [code](https://github.com/mseg-dataset/mseg-semantic)

[Object-Contextual Representations for Semantic Segmentation](https://arxiv.org/abs/1909.11065). Yuhui Yuan, Xilin Chen, Jingdong Wang. CoRR abs/1909.11065 (2019). [code](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR), [code](https://github.com/openseg-group/OCNet.pytorch)

[Disentangled Non-Local Neural Networks](https://arxiv.org/abs/2006.06668). Minghao Yin, Zhuliang Yao, Yue Cao, Xiu Li, Zheng Zhang, Stephen Lin, Han Hu. CoRR abs/2006.06668 (2020)

### Instance segmentation and panoptic segmentation
[Panoptic-DeepLab: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation](https://arxiv.org/abs/1911.10194). Bowen Cheng, Maxwell D. Collins, Yukun Zhu, Ting Liu, Thomas S. Huang, Hartwig Adam, Liang-Chieh Chen. CVPR 2020. The winner of Mapillary Vistas Panoptic Segmentation Task, [COCO + Mapillary Joint Recognition Challenge Workshop at ICCV 2019](https://cocodataset.org/workshop/coco-mapillary-iccv-2019.html#mapillary-panoptic). [code](https://github.com/bowenc0221/panoptic-deeplab)

[1st Place Solutions for OpenImage2019 - Object Detection and Instance Segmentation](https://arxiv.org/abs/2003.07557). Yu Liu, Guanglu Song, Yuhang Zang, Yan Gao, Enze Xie, Junjie Yan, Chen Change Loy, Xiaogang Wang. CoRR abs/2003.07557 (2020)

### Object detection 
[Deep High-Resolution Representation Learning for Visual Recognition](https://arxiv.org/abs/1908.07919). Jingdong Wang, Ke Sun, Tianheng Cheng, Borui Jiang, Chaorui Deng, Yang Zhao, Dong Liu, Yadong Mu,  Mingkui Tan, Xinggang Wang, Wenyu Liu, and Bin Xiao. TPAMI. 2020. [code on MMDetection](https://github.com/HRNet/HRNet-Object-Detection), [code on Mask RCNN](https://github.com/HRNet/HRNet-MaskRCNN-Benchmark), [code with FCOS](https://github.com/HRNet/HRNet-FCOS)


[1st Place Solutions of Waymo Open Dataset Challenge 2020: 2D Object Detection Track](https://drive.google.com/file/d/14VwSjMeRZUtisZtqQPmbll6w4zvZXIAQ/view). Zehao Huang, Zehui Chen, Qiaofei Li, Hongkai Zhang and Naiyan Wang. CVPRW 2020. 

[CenterNet: Keypoint Triplets for Object Detection](https://openaccess.thecvf.com/content_ICCV_2019/papers/Duan_CenterNet_Keypoint_Triplets_for_Object_Detection_ICCV_2019_paper.pdf). Kaiwen Duan, Song Bai, Lingxi Xie, Honggang Qi, Qingming Huang, Qi Tian. ICCV 2019.

[FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/pdf/1904.01355.pdf
). Zhi Tian, Chunhua Shen, Hao Chen, and Tong He. ICCV 2019. [code](https://github.com/HRNet/HRNet-FCOS)

## Human-centric vision

### Human pose estimation
[Deep High-Resolution Representation Learning for Human Pose Estimation]([https://arxiv.org/abs/1902.09212](https://arxiv.org/abs/1902.09212)). Ke Sun, Bin Xiao, Dong Liu, Jingdong Wang. CVPR 2019. [code](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) [video](https://www.youtube.com/watch?v=sIP3MrFWCpg)
    
[Distribution-Aware Coordinate Representation for Human Pose Estimation](https://arxiv.org/abs/1910.06278). Feng Zhang, Xiatian Zhu, Hanbin Dai, Mao Ye, and Ce Zhu. CVPR 2020. [code](https://github.com/ilovepose/DarkPose)

[The Devil Is in the Details: Delving Into Unbiased Data Processing for Human Pose Estimation](https://arxiv.org/abs/1911.07524). Junjie Huang, Zheng Zhu, Feng Guo, Guan Huang. CVPR 2020. [code](https://github.com/HuangJunJie2017/UDP-Pose)

### Video pose estimation
[Learning Temporal Pose Estimation from Sparsely-Labeled Videos]([https://arxiv.org/abs/1906.04016](https://arxiv.org/abs/1906.04016)). Gedas Bertasius, Christoph Feichtenhofer, Du Tran, Jianbo Shi, Lorenzo Torresani. NeurIPS 2019. [code]([https://github.com/facebookresearch/PoseWarper](https://github.com/facebookresearch/PoseWarper))

### 3D human pose estimation
[Cascaded deep monocular 3D human pose estimation with evolutionary training data](https://arxiv.org/abs/2006.07778). Shichao Li, Lei Ke, Kevin Pratama, Yu-Wing Tai, Chi-Keung Tang, Kwang-Ting Cheng. CoRR abs/2006.07778 (2020)

[Motion Guided 3D Pose Estimation from Videos](https://arxiv.org/abs/2004.13985). Jingbo Wang, Sijie Yan, Yuanjun Xiong, Dahua Lin. CoRR abs/2004.13985 (2020)

[Weakly-Supervised 3D Human Pose Learning via Multi-view Images in the Wild](https://arxiv.org/abs/2003.07581). Umar Iqbal, Pavlo Molchanov, and Jan Kautz. CVPR 2020


### Pedestrian Detection
[Pedestrian Detection: The Elephant In The Room](https://arxiv.org/abs/2003.08799). Irtiza Hasan, Shengcai Liao, Jinpeng Li, Saad Ullah Akram, Ling Shao. CoRR abs/2003.08799 (2020). [code](https://github.com/hasanirtiza/Pedestron)

### Face alignment
[Deep High-Resolution Representation Learning for Visual Recognition](https://arxiv.org/abs/1908.07919). Jingdong Wang, Ke Sun, Tianheng Cheng, Borui Jiang, Chaorui Deng, Yang Zhao, Dong Liu, Yadong Mu,  Mingkui Tan, Xinggang Wang, Wenyu Liu, and Bin Xiao. TPAMI. 2020. [code](https://github.com/HRNet/HRNet-Facial-Landmark-Detection)

### Face recognition	
[FAN-Face: a Simple Orthogonal Improvement to Deep Face Recognition](https://www.adrianbulat.com/downloads/AAAI20/FANFace.pdf). Jing Yang, Adrian Bulat, Georgios Tzimiropoulos. AAAI 2020.


### Sign language recognition
[Spatial-Temporal Multi-Cue Network for Continuous Sign Language Recognition](https://arxiv.org/abs/2002.03187). Hao Zhou, Wengang Zhou, Yun Zhou, Houqiang Li. AAAI 2020.


### Multi-Object Tracking
[A Simple Baseline for Multi-Object Tracking. Yifu Zhang, Chunyu Wang](https://arxiv.org/abs/2004.01888), Xinggang Wang, Wenjun Zeng, Wenyu Liu. CoRR abs/2004.01888 (2020). [code](https://github.com/ifzhang/FairMOT)

### Fashion Image Retrieval
[Which Is Plagiarism: Fashion Image Retrieval Based on Regional Representation for Design Protection](http://openaccess.thecvf.com/content_CVPR_2020/papers/Lang_Which_Is_Plagiarism_Fashion_Image_Retrieval_Based_on_Regional_Representation_CVPR_2020_paper.pd). Yining Lang, Yuan He, Fan Yang, Jianfeng Dong, Hui Xue. CVPR 2020.









## Fine-grained visual categorization

[Semi-Supervised Recognition under a Noisy and Fine-grained Dataset](https://arxiv.org/abs/2006.10702). Cheng Cui, Zhi Ye, Yangxi Li, Xinjian Li, Min Yang, Kai Wei, Bing Dai, Yanmei Zhao, Zhongji Liu, Rong Pang. CoRR abs/2006.10702 (2020). [code](https://github.com/PaddlePaddle/PaddleClas)

## Pretraing

[Learning High-Resolution Domain-Specific Representations with a GAN Generator](https://arxiv.org/abs/2006.10451). Danil Galeev, Konstantin Sofiiuk, Danila Rukhovich, Mikhail Romanov, Olga Barinova, Anton Konushin. CoRR abs/2006.10451 (2020)





## Table detection

[CascadeTabNet: An approach for end to end table detection and structure recognition from image-based documents](https://arxiv.org/abs/2004.12629). Devashish Prasad, Ayan Gadpal, Kshitij Kapadni, Manish Visave, Kavita Sultanpure. CVPRW 2020. [code](https://github.com/DevashishPrasad/CascadeTabNet)

## Computational photography

[Foreground-aware Semantic Representations for Image Harmonization](https://arxiv.org/abs/2006.00809). Konstantin Sofiiuk, Polina Popenova, Anton Konushin. CoRR abs/2006.00809 (2020). [code](https://github.com/saic-vul/image_harmonization)

[High-Resolution Network for Photorealistic Style Transfer](https://arxiv.org/abs/1904.11617). Ming Li, Chunyang Ye, Wei Li. CoRR abs/1904.11617 (2019). [code](https://github.com/limingcv/Photorealistic-Style-Transfer)

[Progressive Image Inpainting with Full-Resolution Residual Network](https://arxiv.org/abs/1907.10478). Zongyu Guo, Zhibo Chen, Tao Yu, Jiale Chen, Sen Liu. ACM Multimedia 2019: 2496-2504. [code](https://github.com/ZongyuGuo/Inpainting_FRRN)

[NTIRE 2019 Challenge on Image Enhancement: Methods and Results](http://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Ignatov_NTIRE_2019_Challenge_on_Image_Enhancement_Methods_and_Results_CVPRW_2019_paper.pdf). CVPR Workshops 2019. The winner, the Mt.Stars team, adopted the HRNet.






## 6-DoF Pose Estimation

[Neural Mesh Refiner for 6-DoF Pose Estimation](https://arxiv.org/abs/2003.07561). Di Wu, Yihao Chen, Xianbiao Qi, Yongjian Yu, Weixuan Chen, Rong Xiao. CoRR abs/2003.07561 (2020). [code](https://github.com/stevenwudi/Kaggle_PKU_Baidu)






## Co-Segmentation
	
[Deep Object Co-Segmentation via Spatial-Semantic Network Modulation](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-KaihuaZhang.1673.pdf). Kaihua Zhang, Jin Chen, Bo Liu, Qingshan Liu. AAAI 2020.

## DL platform
Baidu PaddlePaddle. [PaddleSeg model_zoo](
https://github.com/PaddlePaddle/PaddleSeg/blob/release/v0.5.0/docs/model_zoo.md) [PaddleSeg HRNet tutorial](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v0.5.0/turtorial/finetune_hrnet.md)

GLUON-CV [HRNet](https://gluon-cv.mxnet.io/_modules/gluoncv/model_zoo/hrnet.html)
