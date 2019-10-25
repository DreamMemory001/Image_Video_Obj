# Image_and_Video_Obj
## 基于图像，视频的目标检测总结的一点知识

[基于图像和视频目标检测的区别 ](https://www.zhihu.com/question/52185576)

### 图像目标检测

**目前的五大图像处理任务**

* 1.图像分类
* 2.图像分类与定位
* 3.语义分割
* 4.图像目标检测
* 5.实例分割

**六大图像数据库**

* 1.PASCAL Visual Object Classes (VOC) 挑战（人、车、自行车、公交车、飞机、羊、牛、桌等20大类 ）- 
* 2.MS COCO: Microsoft Common Object in Context（80大类， 多目标）
* 3.ImageNet Object Detection: ILSVRC DET 任务（200类别，578,482 图片） 
* 4.Oxford-IIIT Pet Dataset（37 类别，每个类别 200 图片）
* 5.Cityscapes Dataset（30 类别，25,000 + 真实开车场景图片）
* 6.ADE20K Dataset（150+ 类别，22,000 + 普通场景图片）

相比于视频目标检测，图像目标检测相对来说近些年的发展在逐步的走向完善。

从2015年faster-RCNN的的提出，一直到YOLOv3的发表。

最近的图像目标检测的论文中，比较典型的有SNIPER、CornerNet、ExtremeNet、TridentNet、FSAF、FCOS、FoveaBox、两个CenterNet 和 CornerNet-Lite 等等。

- * 1.**SNIPER: Efficient Multi-Scale Training**    **mAP**：47.6        [paper](https://arxiv.org/abs/1805.09300 ) -----[code](https://github.com/MahyarNajibi/SNIPER/ )
- * 2.**TridentNet：Scale-Aware Trident Networks for Object Detection**   **mAP**:48.4    [paper](https://arxiv.org/abs/1901.01892)---[code](https://github.com/TuSimple/simpledet)
- * 3.**HTC + DCN + ResNeXt-101-FPN**    **mAP**:50.7  [paper](https://arxiv.org/abs/1901.07518)----[code](https://github.com/open-mmlab/mmdetection)
- * 4.**NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection**     [paper](https://arxiv.org/abs/1904.07392 )
- * 5.**CornerNet-Saccade+gt attention**    **mAP**:50.3    [paper](https://arxiv.org/abs/1904.08900)----[code](https://github.com/princeton-vl/CornerNet-Lite)
- * 6.**Cascade R-CNN：High Quality Object Detection and Instance Segmentation**     **mAP**:50.9     [paper](https://arxiv.org/abs/1906.09756)---[code](https://github.com/zhaoweicai/Detectron-Cascade-RCNN )
- * 7.**Learning Data Augmentation Strategies for Object Detection**   **mAP**:50.7    [paper](https://arxiv.org/abs/1906.11172)---[code](https://github.com/tensorflow/tpu/tree/master/models/official/detection)


### 视频目标检测的发展 

> Introduction

基于视频的目标检测与普通的图片目标检测的任务是一样的，都是需要给出图片上的物体类别以及对应的位置，但是视频的目标检测相比于图片的目标检测会有更多的难点及其要求。视频中会存在运动模糊，摄像头失焦的现象已经目标物体可能会保持一种奇怪的姿态或者存在严重的遮挡，这往往需要利用视频中的时序信息来对视频中的信息进行推断及其预测。除此之外，视频由于包含大量的视频帧，直接将基于图片的检测模型迁移到视频可能会带来巨大的计算量，视频中的时序信息可能能够帮助我们不用依赖重复的特征计算就能得到相应的解惑，主流的一些方法包括使用光流来进行捕捉视频中的时序信息。

> Methods

目前利用光流法大体分为俩个流派，一个是利用光流来进行洗漱关键帧之间的特征传递，从而节省计算量来达到速度和精度的trade-off；一个是利用光流来将充分利用邻帧的信息，对视频帧进行特征增强，以达到更高的精度，但是速度一般比较慢。大概有如下四个工作。

* 1.来自MSRA的Deep Feature Flow（DFF）,DFF的核心就是利用光流的warp操作来进行特征传递。具体地讲，如下图1，DFF在一段视频帧里面以固定间隔选取稀疏地关键帧，其他的帧则为非关键帧。对于关键帧，DFF 用一个特征提取网络res101去提取高层语义特征，进而检测器则以这些特征为输入从而得到检测结果；对于非关键帧，DFF先经过一个光流网络计算该非关键帧与在此之前最近的关键帧的光流图，然后利用得到的光流图和关键帧的高层特征进行warp操作，从而将特征对齐并传递到该非关键帧，检测器基于此特征输出该非关键帧的检测结果。DFF利用相对轻量的光流网络和warp操作代替原来的res101来得到相应的特征，达到节省计算量的目的。最终在关键帧间隔为10的情况下，达到73.1mAP/20fps(K40). 在比baseline（73.9Map/5fps）损失了0.8mAP的情况下得到了5倍的提速。

 <div align="center"><img src="/images/warp.png"></div>
     <div align="center">图1</div>

所谓的warp操作：warp最开始是用在对图片像素点进行对齐的操作。光流图本质就是记录了某帧图片像素点到另外一帧的运动场，光流图上的每一个点对应着图片上该点的运动矢量。如图2所示，假设我们知道第t帧中的点会运动到第t+1帧的点，这样就得到了运动矢量。如果我们此时要求得第t帧中的像素值，则可以根据其运动矢量和第t+1帧中的像素值来进行双线性插值得到。

 <div align="center"><img src="/images/p2.png"></div>
  <div align="center">图2</div>
    

假设落到点，则有：

DFF则将warp操作扩展到feature map上，从而达到进行特征传递的目的。

* 2.MSRA的Flow Guided Feature Aggregation（FGFA）。与DFF不同，FGFA追求精度而不考虑速度，其对于视频的所有帧都利用特征网络res101提取了特征，为了增强特征，其还利用光流将相邻多帧的特征给warp到当前帧，然后所有的特征输入到一个小的embedding网络从而得到每个特征的相对重要性权重，进而利用这些权重对这些特征进行加权求和，最后得到的增强后的特征再送入检测器，以得到检测结果。通过这样的方式，FGFA得到了更高的精度，但是损失了很多速度，最后的结果为76.3mAP/1.36fps。
* 3.商汤的Impression Network，这个工作是在DFF的基础上做的，Impression除了将关键帧的特征利用warp传递到非关键帧之外，还提出关键帧之间的特征传播与增强，以求保留更多的时序上下文信息。具体地，关键帧的特征利用warp传递到下一个关键帧，两个特征经过几层卷积网络得到相应的重要性权重，进而对其进行加权求和得到新的关键帧的特征，不同的相邻关键帧之间以这种方式不断迭代进行，达到将重要的信息在整个视频中传递的目的。Impression在得到较好结果的同时也有不错的速度，在GTX 1060上达到75.5mAP/20fps。
* 4.MSRA的Towards good performance video object detection。之前DFF和Impression选取关键帧的方式都是启发式地以固定间隔来选取，这个工作给光流网络增加了一个输出，新的输出map上的每一个点的值代表两帧的feature map上的对应点的特征一致性程度。而这个工作觉得当上一个关键帧与当前帧的特征一致性程度总体低到一定的阈值的时候，就要将该帧当作新的关键帧。除了提出动态选取关键帧的方法之外，其还提出根据两帧之间的特征一致性程度来动态更新非关键帧的特征，而不是跟之前的工作一样直接利用warp过来的特征。具体地，当feature map上的某个位置特征一致性比较低的时候，我们就不利用warp过来的特征来更新非关键帧的特征，而使用原来的特征；如果比较高的话，则直接使用warp过来的特征。通过这两个技巧，这个工作得到了比较好的一个trade-off：78.6mAP(+deformable)/8.6fps(K40)。

### 视频检测的意义：

传统的基于图片的目标检测方法已经非常成熟，对于视频目标检测来说，如果视频流按帧一张一张使用图片的目标检测算法来处理会出现以下两类问题：

​         1.因为视频流的图片信息具有时间和空间相关性，相邻帧之间的特城提取网络会输出有冗余的特征图信息，会造成没必要的计算浪费。

​         2.图片的目标检测算法在目标物体运动模糊，拍摄焦距失调，物体部分遮挡，非刚性物体罕见变形姿态的情况下，很难获得较为准确的结果，而这些情况（如下图）在视频的拍摄中情况较为多见。

 <div align="center"><img src="/images/p3.png"></div>

> 上述的意思来自于[atrous alood的论文笔记](https://www.zhihu.com/people/yry-79-23)]内容详情请查看链接

### 收集的视频和图像的目标检测的数据集

1. ILSVRC2015: Object detection from video (VID)

   [ILSVRC2015数据集地址](<http://bvisionweb1.cs.unc.edu/ilsvrc2015/ILSVRC2015_VID.tar.gz> )

   [ILSVRC2017数据集地址](http://bvisionweb1.cs.unc.edu/ILSVRC2017/download-videos-1p39.php)

ImageNet VID challenges，这是在kaggle上的关于ImageNet上基于视频的目标检测挑战，目的是为了识别和标          记视频中的普通目标。

该数据集文件如下：

* imagenet_object_detection_video_train.tar.gz包含了训练集和校准集的图像数据和GT。
* imagenet_object_detection_video_test.tar.gz包含了测试集的图像数据。 
  * 其中图像标注格式都是基于PASCAL VOC数据集格式的XML文件（可以使用PASCAL开发工具套件来解析标注）。
  * 每一个视频都是以JPEG格式存储，代表不同帧。
  * ImageSet文件夹包含了定义了主要的检测任务的图像列表。例如，文件夹ILSVRC2015_VID_train_0000/ILSVRC2015_train_00025030表示一个视频，其中该文件夹中的000000.JPEG文件表示第一帧，并且000000.xml表示该帧的标注。

  

2. YouTube-Objects dataset v2.2

    YouTube-Objects数据集由从YouTube收集的视频组成，查询PASCAL VOC Challenge的10个对象类别的名称。             每个对象包含9到24个视频。每个视频的持续时间在30秒到3分钟之间变化。视频被弱标注，即我们确保每个视频包含相应类的至少一个对象。该数据集包括aeroplane、bird、boat、car、cat、cow、dog、horse、motorbike和train这10个类别，具体可在网页上查看[YouTube-Objects v2.3 Preview](YouTube-Objects v2.3 Preview)。

​      [YouTube-Objects 数据集](http://calvin.inf.ed.ac.uk/datasets/youtube-objects-dataset/ )

3. Yahoo实验室公开的一亿Flickr的图像和视频

   [Yahoo实验室数据集](http://yahoolabs.tumblr.com/post/89783581601/one-hundred-million-creative-commons-flickr-images-for)

4. 人脸识别的数据集

   [人脸识别](http://www.face-rec.org/databases/)

5. 比较新的计算机视觉数据网站

   [计算机视觉数据](http://riemenschneider.hayko.at/vision/dataset/)

6. 猫狗图片

​       [猫狗图片](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)

### 相关资料

* [ImageNet Object Detection from Video Challenge](https://www.kaggle.com/account/login?returnUrl=%2Fc%2Fimagenet-object-detection-from-video-challenge) kaggle上一个ImageNet基于视频的目标检测的比赛，可以做为初始数据测试相应的算法。

* [Optimizing Video Object Detection via a Scale-Time Lattice](https://arxiv.org/pdf/1804.05472.pdf) 特别推荐阅读的一篇论文。

* [FlowNet: Learning Optical Flow with Convolutional Networks](https://arxiv.org/abs/1504.06852) 这篇介绍使用CNN来计算光流的模型

* [PYSOT](https://github.com/STVIR/pysot) 商汤科技最新的视频目标跟踪的成果

*  ### 最新的一些论文成果

  ### Arxiv

  - **Looking Fast and Slow:** Mason Liu, Menglong Zhu, Marie White, Yinxiao Li, Dmitry Kalenichenko. "Looking Fast and Slow: Memory-Guided Mobile Video Object Detection" Arxiv(2019). [ [paper](https://arxiv.org/pdf/1903.10172v1.pdf)]
  - **Towards High Performance for Mobiles:** Xizhou Zhu, Jifeng Dai, Xingchi Zhu, Yichen Wei, Lu Yuan. "Towards High Performance Video Object Detection for Mobiles" Arxiv(2018). [ [paper](https://arxiv.org/pdf/1804.05830.pdf)]

  ### ICCV2019

  - **Sequence Level Semantics Aggregation:** Haiping Wu, Yuntao Chen, Naiyan Wang, Zhaoxiang Zhang. "Sequence Level Semantics Aggregation for Video Object Detection" ICCV(2019). [ [paper](https://arxiv.org/pdf/1907.06390v1.pdf)]
  - **Average Delay:** Huizi Mao, Xiaodong Yang, William J. Dally. "A Delay Metric for Video Object Detection: What Average Precision Fails to Tell." ICCV (2019). [ [paper](https://arxiv.org/pdf/1908.06368.pdf)]
  - **Fully Motion-Aware Networks:** Jiajun Deng, Yingwei Pan, Ting Yao, Wengang Zhou, Houqiang Li, Tao Mei. "Relation Distillation Networks for Video Object Detection." ICCV (2019). [ [paper](https://arxiv.org/pdf/1908.09511.pdf)]

  ### ECCV2018

  - **Fully Motion-Aware Network:** Shiyao Wang, Yucong Zhou, Junjie Yan, Zhidong Deng. "Fully Motion-Aware Network for Video Object Detection." ECCV (2018). [ [code](https://github.com/wangshy31/MANet_for_Video_Object_Detection.git)]
  - **SpatioTemporal Sampling Network:** Gedas Bertasius, Lorenzo Torresani, ianbo Shi. "Object Detection in Video with Spatiotemporal Sampling Networks." ECCV (2018). [ [paper](https://arxiv.org/pdf/1803.05549.pdf)]
  - **Aligned Spatial-Temporal Memory:** Fanyi Xiao, Yong Jae Lee. "Video Object Detection with an Aligned Spatial-Temporal Memory." ECCV(2018). [ [paper](https://arxiv.org/abs/1712.06317)]

  ### CVPR2018

  - **Towards High Performance:** Xizhou Zhu, Jifeng Dai, Lu Yuan, Yichen Wei. "Towards High Performance Video Object Detection." CVPR (2018). [ [paper](https://arxiv.org/abs/1711.11577)]
  - **Scale-Time Lattice:** Kai Chen, Jiaqi Wang, Shuo Yang, Xingcheng Zhang, Yuanjun Xiong, Chen Chang Loy, Dahua Lin. "Optimizing Video Object Detection vis a Scale-Time Lattice." CVPR (2018).  [paper](http://mmlab.ie.cuhk.edu.hk/projects/ST-Lattice/ST-Lattice.pdf)
  - **Mobile Video Object Detection:** Mason Liu, Menglong Zhu. "Mobile Video Object Detection with Temporally-Aware Feature Maps." CVPR (2018). [ [paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Mobile_Video_Object_CVPR_2018_paper.pdf)]

  ### ICCV2017

  - **FGFA:** Xizhou Zhu, Yujie Wang, Jifeng Dai, Lu Yuan, Yichen Wei. "Flow-Guided Feature Aggregation for Video Object Detection." ICCV (2017). [ [paper](https://arxiv.org/abs/1703.10025)]---------[[code](https://github.com/msracver/Flow-Guided-Feature-Aggregation)]
  - **D_T:** Christoph Feichtenhofer, Axel Pinz, Andrew Zisserman. "Detect to Track and Track to Detect." ICCV (2017). [ [paper](http://www.robots.ox.ac.uk/~vgg/publications/2017/Feichtenhofer17/feichtenhofer17.pdf)]---------[[project](http://www.robots.ox.ac.uk/~vgg/research/detect-track/)]

  ### CVPR2017

  - **DFF:** Xizhou Zhu, Yuwen Xiong, Jifeng Dai, Lu Yuan, Yichen Wei. "Deep Feature Flow for Video Recognition." CVPR (2017). [paper](https://arxiv.org/abs/1611.07715)

  ## Object-Detection

  object detection papers based deep learning

  ### Arxiv

  - **Light-Head R-CNN:** Zeming Li, Chao Peng, Gang Yu, Xiangyu Zhang, Yangdong Deng, Jian Sun. "Light-Head R-CNN: In Defense of Two-Stage Object Detector."[[paper](https://arxiv.org/abs/1711.07264)]
  - **YOLOv3:** Joseph Redmon, Ali Farhadi. "YOLOv3: An Incremental Improvement." [[paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf)]

  ### ECCV2018

  - **DetNet:** Zeming Li, Chao Peng, Gang Yu, Xiangyu Zhang, Yangdong Deng, Jian Sun. "DetNet: A Backbone network for Object Detection." [[paper](https://arxiv.org/pdf/1804.06215.pdf)]
  - **IOU-Net:** Borui Jiang, Ruixuan Luo, Jiayuan Mao, Tete Xiao, Yuning Jiang. "Acquisition of Localization Confidence for Accurate Object Detection." ECCV(2018).  [paper](https://arxiv.org/pdf/1807.11590.pdf)

  ### CVPR2018

  - **SNIP:** Bharat Singh, Larry S. Davis. "An Analysis of Scale Invariance in Object Detection - SNIP." [paper](https://arxiv.org/pdf/1711.08189.pdf)--------[code](https://github.com/bharatsingh430/snip)
  - **Cascade-RCNN:** Zhaowei Cai, Nuno Vasconcelos. "Cascade R-CNN: Delving into High Quality Object Detectio." [paper](https://arxiv.org/pdf/1712.00726.pdf)]
  - **Relation-Networks:** Han Hu, Jiayuan Gu, Zheng Zhang, Jifeng Dai, Yichen Wei. "Relation Networks for Object Detection." [paper](https://arxiv.org/pdf/1711.11575.pdf)

  ### ICCV2017

  - **RetinaNet:** Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollar. "Focal Loss for Dense Object Detection." [[paper](https://arxiv.org/abs/1708.02002)]
  - **Mask R-CNN:** Kaiming He, Georgia Gkioxari, Piotr Dollar, Ross Girshick. "Mask R-CNN."[paper](https://arxiv.org/abs/1703.06870)------[caffe2_code](https://github.com/facebookresearch/Detectron)----[pytorch_code](https://github.com/facebookresearch/maskrcnn-benchmark)

  ### CVPR2017

  - **YOLO9000:** Joseph Redmon, Ali Farhadi. "YOLO9000: Better, Faster, Stronger." [paper](https://arxiv.org/abs/1612.08242)---------[project](https://pjreddie.com/publications/)
  - **FPN:** Tsung-Yi Lin, Piotr Dollar, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie. "Feature Pyramid Networks for Object Detection." [[paper](https://arxiv.org/abs/1612.03144)]

  ### NIPS2016

  - **R-FCN:** Jifeng Dai, Yi Li, Kaiming He, Jian Sun. "R-FCN: Object Detection via Region-based Fully Convolutional Networks." [[paper](https://arxiv.org/abs/1605.06409)]

  ### ECCV2016

  - **SSD:** Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C.Berg. "SSD: Single Shot MultiBox Detector." [paper](https://arxiv.org/abs/1512.02325)                                                            [code](https://github.com/weiliu89/caffe/tree/ssd)

  ### CVPR2016

  - **YOLO:** Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi. "You Only Look Once: Unified, Real-Time Object Detection." [paper](https://arxiv.org/abs/1506.02640)----------[project](https://pjreddie.com/publications/)

  ### NIPS2015

  - **Faster R-CNN:** Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks." [paper](https://arxiv.org/abs/1506.01497)-------[code](https://github.com/rbgirshick/py-faster-rcnn)

  ### ICCV2015

  - **Fast R-CNN:** Ross Girshick. "Fast R-CNN." [[paper](https://arxiv.org/abs/1504.08083)]

  ### CVPR2014

  - **R-CNN:** Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. "Rich feature hierarchies for accurate object detection and semantic segmentation." [[paper](https://arxiv.org/abs/1311.2524)]

###  视频语义分割

> 相比于图像语义，视频语义分割具有高帧数（15-30帧/s）,前后帧之间高相关性的特点，并且在自动驾驶任务中，对RGB摄像头传入的视频信号具有很高的实时性的要求。

* 1.[STFCN: Spatio-Temporal FCN for Semantic Video Segmentation](https://arxiv.org/pdf/1608.05971.pdf)

 本篇论文在FCN的基础上进行改进的，利用LSTM将FCNs并联成为一个叫做时空卷积网络的结构（spatio-  temporal CNNs）

主要的贡献：

   1.该方法提升了原有的语义分割结果。

   2.提出了一种结合实践特征与空间特征的端到端架构。

 <div align="center"><img src="/images/p4.png"></div>

* 2.[Semantic Video Segmentation by Gated Recurrent Flow Propagation](https://arxiv.org/pdf/1612.08871.pdf)     (基于门控递归流传播的语义视频分割)

在视频语义分割问题当中，还有一个无法避免的问题就是确少高质量的标注数据，因为视频任务数据量大（假设一秒30帧，一分钟的数据就是1800帧）而语义分割的数据标注极为繁琐耗时（大约30分钟可以标注一张）。因此，如何有效利用视频语义分割任务中少量高质量标注数据集达到好的分割效果也是一个很好的研究方向。针对少量标注样本问题，主要解决方案就是进行弱监督或者半监督学习，弱监督学习方法不适用完整标注数据集进行训练，而是使用大量的分类或者检测数据集进行训练，从而减少标注成本提高分割准确率；半监督学习则是使用少量标注数据集训练网络以求得到一个较好的泛化模型，在视频语义分割任务当中就是关键帧提取，只针对视频中少量关键帧的标注数据进行训练，使模型适用于整个视频流。

在本篇论文中，作者设计了一个叫做Spatio-Temporal Transformer Gated Recurrent Unit（不会翻译）的单元来融合各帧信息，作者认为相邻两帧之间包含大量冗余信息，但是两帧之间差异较大（漂移形变）的区域包含的信息将十分有意义，作者使用了[光流](http://www.docin.com/p-1725315067.html)来衡量漂移形变比较明显的区域。

 <div align="center"><img src="/images/p5.png"></div>

 

### 目标检测 

* [cascade-rcnn](https://github.com/zhaoweicai/cascade-rcnn)
* [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch)
* [mAP](https://github.com/Cartucho/mAP) mean AP python版本，对于理解object detection的评估有帮助。

**mAP评价**

* [mAP（mean average precision）](https://blog.csdn.net/chenyanqiao2010/article/details/50114799)
* [Object-Detection-Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics) 常见的目标检测评估指标。
* [Evaluation of ranked retrieval results](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html)
* [The PASCAL Visual Object Classes Challenge 2012 (VOC2012) Development Kit](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html#SECTION00050000000000000000)
* [COCO Detection Challenge](https://competitions.codalab.org/competitions/5181)
* [Measuring Object Detection models - mAP - What is Mean Average Precision?](http://tarangshah.com/blog/2018-01-27/what-is-map-understanding-the-statistic-of-choice-for-comparing-object-detection-models/) 较好地计算了目标检测中的评价模型。
* [Intersection over Union (IoU) for object detection](https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/) pyimagesearch中IOU目标检测的相关定义。

