# Awesome-Wheat-HSI-DeepLearning
A summary of articles, and visualizations related to hyperspectral imaging (HSI) Using Deep learning in wheat crops

🌟 What is Hyperspectral Imaging (HSI)?

Hyperspectral imaging (HSI) is a powerful technology that captures and analyzes light across a wide range of wavelengths. Unlike traditional cameras that record only red, green, and blue (RGB) colors, hyperspectral sensors collect data in hundreds of continuous spectral bands, creating a "spectral fingerprint" for each pixel in an image.

📸 How it Works:

Each pixel in a hyperspectral image contains a detailed spectrum of light, enabling the identification and analysis of materials, objects, and processes that are invisible to the naked eye.

🌍 Applications:

- Agriculture: Monitoring crop health, identifying diseases, and optimizing yields.
- Environmental Science: Tracking pollution and studying ecosystems.
- Medicine: Detecting diseases and analyzing tissue.
- Remote Sensing: Land cover classification and mineral exploration.
  
🔍 Why It Matters:
Hyperspectral imaging allows us to see beyond what human eyes can perceive, opening up endless possibilities for research, technology, and innovation.

<p align="center">
  <img width="600" src="/hyperspectral1.png" "Example of anomaly detection.">
</p>

## Table of Contents

- [Methodology](#Methodology)
- [Supervised Learning](#Supervised-Learning)
  - [CNN](#CNN)
  - [RNN](#RNN)
  - [Transformer](#Transformer)
  - [Mamba](#Mamba)
  - [SAE](#SAE)
  - [TL](#TL)
  - [DBN](#DBN)
- [Semi-Supervised Learning](#Semi-Supervised-Learning)
- [Unsupervised Learning](#Unsupervised-Learning)
  - [DBN](#DBN)
  - [SAE](#SAE)
  - [Diffusion](#Diffusion)
  - [Change detection](#change-detection)
- [Applications of HSI technology in wheat crops](#Applications-of-hsi-technology-in-wheat-crops)
  - [Wheat Crop Classification](#Wheat-Crop-Classification)
  - [Wheat Crop Nutrient Estimation](#Wheat-Crop-Nutrient-Estimation)
  - [Wheat Crop Yield Estimation](#Wheat-Crop-Yield-Estimation)
  - [Wheat Crop Disease Monitoring and Detection](#Wheat-Crop-Disease-Monitoring-and-Detection)

## Methodology
### Supervised Learning
### CNN
- Kshitiz Dhakal, Upasana Sivaramakrishnan, Xuemei Zhang, Kassaye Belay, Joseph Oakes, Xing Wei, and Song Li. Machine learning analysis of
hyperspectral images of damaged wheat kernels. Sensors, 23(7):3523, 2023. [Paper](https://www.mdpi.com/1424-8220/23/7/3523)
- Zilong Zhong, Jonathan Li, Zhiming Luo, and Michael Chapman. Spectral–spatial residual network for hyperspectral image classification: A 3-d
deep learning framework. IEEE Transactions on Geoscience and Remote Sensing, 56(2):847–858, 2017. [Paper](https://ieeexplore.ieee.org/abstract/document/8061020)
- Yushi Chen, Hanlu Jiang, Chunyang Li, Xiuping Jia, and Pedram Ghamisi. Deep feature extraction and classification of hyperspectral images based
on convolutional neural networks. IEEE transactions on geoscience and remote sensing, 54(10):6232–6251, 2016. [Paper](https://ieeexplore.ieee.org/abstract/document/7514991)
- Xiaofei Yang, Yunming Ye, Xutao Li, Raymond YK Lau, Xiaofeng Zhang, and Xiaohui Huang. Hyperspectral image classification with deep learning
models. IEEE Transactions on Geoscience and Remote Sensing, 56(9):5408–5423, 2018. [Paper](https://ieeexplore.ieee.org/abstract/document/8340197)
- LIU Bing, YU Xuchu, ZHANG Pengqiang, and TAN Xiong. Deep 3d convolutional network combined with spatial-spectral features for hyperspectral
image classification. Acta Geodaetica et Cartographica Sinica, 48(1):53, 2019. [Paper](http://xb.chinasmp.com/EN/Y2019/V48/I1/53)
- Swalpa Kumar Roy, Gopal Krishna, Shiv Ram Dubey, and Bidyut B Chaudhuri. Hybridsn: Exploring 3-d–2-d cnn feature hierarchy for hyperspectral
image classification. IEEE Geoscience and Remote Sensing Letters, 17(2):277–281, 2019. [Paper](https://ieeexplore.ieee.org/abstract/document/8736016)
- Weiwei Song, Shutao Li, Leyuan Fang, and Ting Lu. Hyperspectral image classification with deep feature fusion network. IEEE Transactions on
Geoscience and Remote Sensing, 56(6):3173–3184, 2018. [Paper](https://ieeexplore.ieee.org/abstract/document/8283837)
- Vimal K. Shrivastava Somenath Bera and Suresh Chandra Satapathy. Advances in hyperspectral image classification based on convolutional
neural networks: A review. Tech Science Press (TSP), 2022. [Paper](https://www.techscience.com/CMES/v133n2/48965/html)
- Ying Li, Haokui Zhang, and Qiang Shen. Spectral–spatial classification of hyperspectral imagery with 3d convolutional neural network. Remote
Sensing, 9(1):67, 2017. [Paper](https://www.mdpi.com/2072-4292/9/1/67)
- Leyuan Fang, Zhiliang Liu, and Weiwei Song. Deep hashing neural networks for hyperspectral image feature extraction. IEEE Geoscience and
Remote Sensing Letters, 16(9):1412–1416, 2019. [Paper](https://ieeexplore.ieee.org/abstract/document/8663589)
- Hyungtae Lee and Heesung Kwon. Going deeper with contextual cnn for hyperspectral image classification. IEEE Transactions on Image Processing,
26(10):4843–4855, 2017. [Paper](https://ieeexplore.ieee.org/abstract/document/7973178)
### RNN
- R Venkatesan and Sevugan Prabu. Hyperspectral image features classification using deep learning recurrent neural networks. Journal of medical
systems, 43(7):216, 2019. [Paper](https://link.springer.com/article/10.1007/s10916-019-1347-9)
- Haowen Luo. Shorten spatial-spectral rnn with parallel-gru for hyperspectral image classification. arXiv preprint arXiv:1810.12563, 2018. [Paper](https://arxiv.org/abs/1810.12563)
- Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence
modeling. arXiv preprint arXiv:1412.3555, 2014. [Paper](https://arxiv.org/abs/1412.3555)
- Lichao Mou, Pedram Ghamisi, and Xiao Xiang Zhu. Deep recurrent neural networks for hyperspectral image classification. IEEE transactions on
geoscience and remote sensing, 55(7):3639–3655, 2017. [Paper](https://ieeexplore.ieee.org/abstract/document/7914752)
- Emile Ndikumana, Dinh Ho Tong Minh, Nicolas Baghdadi, Dominique Courault, and Laure Hossard. Deep recurrent neural network for agricultural
classification using multitemporal sar sentinel-1 for camargue, france. Remote Sensing, 10(8):1217, 2018. [Paper](https://www.mdpi.com/2072-4292/10/8/1217)
- Lauri Salmela, Nikolaos Tsipinakis, Alessandro Foi, Cyril Billet, John M Dudley, and Goëry Genty. Predicting ultrafast nonlinear dynamics in fibre optics with a recurrent neural network. Nature machine intelligence, 3(4):344–354, 2021. [Paper](https://www.nature.com/articles/s42256-021-00297-z)
### Transformer
- Lanxue Dang, Libo Weng, Yane Hou, Xianyu Zuo, and Yang Liu. Double-branch feature fusion transformer for hyperspectral image classification.
Scientific Reports, 13(1):272, 2023. [Paper](https://www.nature.com/articles/s41598-023-27472-z)
- Neetu Sigger, Quoc-Tuan Vien, Sinh Van Nguyen, Gianluca Tozzi, and Tuan Thanh Nguyen. Unveiling the potential of diffusion model-based
framework with transformer for hyperspectral image classification. Scientific Reports, 14(1):8438, 2024. [Paper](https://www.nature.com/articles/s41598-024-58125-4)
- Shukai Liu, Changqing Yin, and Huijuan Zhang. Cesa-mcformer: An efficient transformer network for hyperspectral image classification by
eliminating redundant information. Sensors, 24(4):1187, 2024. [Paper](https://www.mdpi.com/1424-8220/24/4/1187)
- Jiaxing Xie, Jiajun Hua, Shaonan Chen, PeiwenWu, Peng Gao, Daozong Sun, Zhendong Lyu, Shilei Lyu, Xiuyun Xue, and Jianqiang Lu. Hypersformer:
A transformer-based end-to-end hyperspectral image classification method for crop classification. Remote Sensing, 15(14):3491, 2023. [Paper](https://www.mdpi.com/2072-4292/15/14/3491)
### Mamba
- Jiamu Sheng, Jingyi Zhou, Jiong Wang, Peng Ye, and Jiayuan Fan. Dualmamba: A lightweight spectral-spatial mamba-convolution network for
hyperspectral image classification. arXiv preprint arXiv:2406.07050, 2024. [Paper](https://arxiv.org/abs/2406.07050)
- Judy X Yang, Jun Zhou, Jing Wang, Hui Tian, and Alan Wee Chung Liew. Hsimamba: Hyperpsectral imaging efficient feature learning with
bidirectional state space for classification. arXiv preprint arXiv:2404.00272, 2024. [Paper](https://arxiv.org/abs/2404.00272)
- Weilian Zhou, Sei-Ichiro Kamata, Haipeng Wang, Man-Sing Wong, et al. Mamba-in-mamba: Centralized mamba-cross-scan in tokenized mamba
model for hyperspectral image classification. arXiv preprint arXiv:2405.12003, 2024. [Paper](https://www.sciencedirect.com/science/article/pii/S0925231224015224)
- Lingbo Huang, Yushi Chen, and Xin He. Spectral-spatial mamba for hyperspectral image classification. arXiv preprint arXiv:2404.18401, 2024. [Paper](https://arxiv.org/abs/2404.18401)
- Yan He, Bing Tu, Bo Liu, Jun Li, and Antonio Plaza. 3dss-mamba: 3d-spectral-spatial mamba for hyperspectral image classification. arXiv preprint
arXiv:2405.12487, 2024. [Paper](https://arxiv.org/abs/2405.12487)
- Aitao Yang, Min Li, Yao Ding, Leyuan Fang, Yaoming Cai, and Yujie He. Graphmamba: An efficient graph structure learning vision mamba for
hyperspectral image classification. arXiv preprint arXiv:2407.08255, 2024. [Paper](https://ieeexplore.ieee.org/abstract/document/10746459)
### SAE
- Chen Xing, Li Ma, and Xiaoquan Yang. Stacked denoise autoencoder based feature extraction and classification for hyperspectral images. Journal
of Sensors, 2016(1):3632943, 2016. [Paper](https://onlinelibrary.wiley.com/doi/full/10.1155/2016/3632943)
- Yang Bai, Xiyan Sun, Yuanfa Ji, Wentao Fu, and Jinli Zhang. Two-stage multi-dimensional convolutional stacked autoencoder network model for
hyperspectral images classification. Multimedia Tools and Applications, 83(8):23489–23508, 2024. [Paper](https://link.springer.com/article/10.1007/s11042-023-16456-w)
- Kun Liang, Jiani Huang, Ruiyin He, Qiujin Wang, Yinyin Chai, and Mingxia Shen. Comparison of vis-nir and swir hyperspectral imaging for the
non-destructive detection of don levels in fusarium head blight wheat kernels and wheat flour. Infrared Physics & Technology, 106:103281, 2020. [Paper](https://www.sciencedirect.com/science/article/abs/pii/S1350449519309661)
### TL
- Yao Zhang, Jian Hui, Qiming Qin, Yuanheng Sun, Tianyuan Zhang, Hong Sun, and Minzan Li. Transfer-learning-based approach for leaf chlorophyll
content estimation of winter wheat from hyperspectral data. Remote Sensing of Environment, 267:112724, 2021. [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0034425721004442)
- Hao Zhou, Xianwang Wang, Kunming Xia, Yi Ma, and Guowu Yuan. Transfer learning-based hyperspectral image classification using residual
dense connection networks. Sensors, 24(9):2664, 2024. [Paper](https://www.mdpi.com/1424-8220/24/9/2664)
- Xin Zhao, Yi Liang, Alan JX Guo, and Fei Zhu. Classification of small-scale hyperspectral images with multi-source deep transfer learning. Remote Sensing Letters, 11(4):303–312, 2020. [Paper](https://www.tandfonline.com/doi/abs/10.1080/2150704X.2020.1714772)
- Rohit Bharti, Dipen Saini, and Rahul Malik. A novel approach for hyper spectral images using transfer learning. In IOP Conference Series: Materials Science and Engineering, volume 1022, page 012120. IOP Publishing, 2021. [Paper](https://iopscience.iop.org/article/10.1088/1757-899X/1022/1/012120/meta)
- Xuefeng Jiang, Yue Zhang, Yi Li, Shuying Li, and Yanning Zhang. Hyperspectral image classification with transfer learning and markov random
fields. IEEE Geoscience and Remote Sensing Letters, 17(3):544–548, 2019. [Paper](https://ieeexplore.ieee.org/abstract/document/8758842)
- Xin Zhao, Shuo Liu, Haotian Que, Min Huang, and Qibing Zhu. Adfsnet: An adaptive domain feature separation network for the classification of
wheat seed using hyperspectral images. Sensors, 23(19):8116, 2023. [Paper](https://www.mdpi.com/1424-8220/23/19/8116)
### DBN
- Chenming Li, Yongchang Wang, Xiaoke Zhang, Hongmin Gao, Yao Yang, and Jiawei Wang. Deep belief network for spectral–spatial classification
of hyperspectral remote sensor data. Sensors, 19(1):204, 2019. [Paper](https://www.mdpi.com/1424-8220/19/1/204)
- A Sellami and IR Farah. Spectra-spatial graph-based deep restricted boltzmann networks for hyperspectral image classification. In 2019 PhotonIcs & Electromagnetics Research Symposium-Spring (PIERS-Spring), pages 1055–1062. IEEE, 2019. [Paper](https://ieeexplore.ieee.org/abstract/document/9017309)
- Atif Mughees and Linmi Tao. Multiple deep-belief-network-based spectral-spatial classification of hyperspectral images. Tsinghua Science and
Technology, 24(2):183–194, 2018. [Paper](https://ieeexplore.ieee.org/abstract/document/8595297)
### Semi-Supervised Learning
- Weidong Zhang, Zexu Li, Guohou Li, Peixian Zhuang, Guojia Hou, Qiang Zhang, and Chongyi Li. Gacnet: Generate adversarial-driven cross-aware
network for hyperspectral wheat variety identification. IEEE Transactions on Geoscience and Remote Sensing, 2023. [Paper](https://ieeexplore.ieee.org/abstract/document/10375525)
- Zhi He, Han Liu, Yiwen Wang, and Jie Hu. Generative adversarial networks-based semi-supervised learning for hyperspectral image classification. Remote Sensing, 9(10):1042, 2017. [Paper](https://www.mdpi.com/2072-4292/9/10/1042)
- Ying Zhan, Yufeng Wang, and Xianchuan Yu. Semisupervised hyperspectral image classification based on generative adversarial networks and
spectral angle distance. Scientific Reports, 13(1):22019, 2023. [Paper](https://www.nature.com/articles/s41598-023-49239-2)
- Lin Zhu, Yushi Chen, Pedram Ghamisi, and Jón Atli Benediktsson. Generative adversarial networks for hyperspectral image classification. IEEE Transactions on Geoscience and Remote Sensing, 56(9):5046–5063, 2018. [Paper](https://ieeexplore.ieee.org/abstract/document/8307247)
- Xiaobo Liu, Yulin Qiao, Yonghua Xiong, Zhihua Cai, and Peng Liu. Cascade conditional generative adversarial nets for spatial-spectral hyperspectral sample generation. Science China Information Sciences, 63:1–16, 2020. [Paper](https://link.springer.com/article/10.1007/s11432-019-2798-9)
- Zhixiang Xue. A general generative adversarial capsule network for hyperspectral image spectral-spatial classification. Remote Sensing Letters,
11(1):19–28, 2020. [Paper](https://www.tandfonline.com/doi/abs/10.1080/2150704X.2019.1681598)
- Hongmin Gao, Dan Yao, Mingxia Wang, Chenming Li, Haiyun Liu, Zaijun Hua, and Jiawei Wang. A hyperspectral image classification method
based on multi-discriminator generative adversarial networks. Sensors, 19(15):3269, 2019. [Paper](https://www.mdpi.com/1424-8220/19/15/3269)
- Hao Li, Liu Zhang, Heng Sun, Zhenhong Rao, and Haiyan Ji. Discrimination of unsound wheat kernels based on deep convolutional generative
adversarial network and near-infrared hyperspectral imaging technology. Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy,
268:120722, 2022. [Paper](https://www.sciencedirect.com/science/article/abs/pii/S1386142521012993)
- Junjie Wang, Feng Gao, Junyu Dong, and Qian Du. Adaptive dropblock-enhanced generative adversarial networks for hyperspectral image
classification. IEEE Transactions on Geoscience and Remote Sensing, 59(6):5040–5053, 2020. [Paper](https://ieeexplore.ieee.org/abstract/document/9173809)
- Jiaguo Zhao, Junjie Zhang, Huaxi Huang, and Jian Zhang. Enhancing semi-supervised few-shot hyperspectral image classification via progressive sample selection. Remote Sensing, 16(10):1747, 2024. [Paper](https://www.mdpi.com/2072-4292/16/10/1747)
- Qingyan Wang, Meng Chen, Junping Zhang, Shouqiang Kang, and Yujing Wang. Improved active deep learning for semi-supervised classification
of hyperspectral image. Remote Sensing, 14(1):171, 2021. [Paper](https://www.mdpi.com/2072-4292/14/1/171)
- Hao Wu and Saurabh Prasad. Semi-supervised deep learning using pseudo labels for hyperspectral image classification. IEEE Transactions on
Image Processing, 27(3):1259–1270, 2017. [Paper](https://ieeexplore.ieee.org/abstract/document/8105856)
- Bei Fang, Ying Li, Haokui Zhang, and Jonathan Cheung-Wai Chan. Semi-supervised deep learning classification for hyperspectral image based on dual-strategy sample selection. Remote Sensing, 10(4):574, 2018. [Paper](https://www.mdpi.com/2072-4292/10/4/574?se=toc&so=cu)
- Zhiyou Zhang. Semi-supervised hyperspectral image classification algorithm based on graph embedding and discriminative spatial information.
Microprocessors and Microsystems, 75:103070, 2020. [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0141933120300466)
### Unsupervised Learning
### DBN
- Jiangong Yang, Yanhui Guo, and Xili Wang. Feature extraction of hyperspectral images based on deep boltzmann machine. IEEE Geoscience and
Remote Sensing Letters, 17(6):1077–1081, 2019. [Paper](https://ieeexplore.ieee.org/abstract/document/8879495)
- Zhengying Li, Hong Huang, Zhen Zhang, and Guangyao Shi. Manifold-based multi-deep belief network for feature extraction of hyperspectral
image. Remote Sensing, 14(6):1484, 2022. [Paper](https://www.mdpi.com/2072-4292/14/6/1484)
### SAE
- Atif Mughees and Linmi Tao. Efficient deep auto-encoder learning for the classification of hyperspectral images. In 2016 international conference on virtual reality and visualization (ICVRV), pages 44–51. IEEE, 2016. [Paper](https://ieeexplore.ieee.org/abstract/document/7938171)
- Afsana Afrin, Md Rakibul Haque, and Md Al Mamun. Enhancing hyperspectral image compression through stacked autoencoder approach. In
2024 6th International Conference on Electrical Engineering and Information & Communication Technology (ICEEICT), pages 1372–1377. IEEE, 2024. [Paper](https://ieeexplore.ieee.org/abstract/document/10534540)
- Lei Deng, Bing Zhou, Jiaju Ying, and Runze Zhao. A noise estimation method for hyperspectral image based on stacked autoencoder. IEEE Access, 2023. [Paper](https://ieeexplore.ieee.org/abstract/document/10225536)
- Chunhong Cao, Wei Song, Han Xiang, Hongbo Yi, Fen Xiao, and Xieping Gao. A two-stream stacked autoencoder with inter-class separability for
bilinear hyperspectral unmixing. IEEE Transactions on Computational Imaging, 2024. [Paper](https://ieeexplore.ieee.org/abstract/document/10444024)
- Lloyd Windrim, Rishi Ramakrishnan, Arman Melkumyan, Richard J Murphy, and Anna Chlingaryan. Unsupervised feature-learning for hyperspectral
data with autoencoders. Remote Sensing, 11(7):864, 2019. [Paper](https://www.mdpi.com/2072-4292/11/7/864)
### Diffusion
- Li Pang, Xiangyu Rui, Long Cui, Hongzhong Wang, Deyu Meng, and Xiangyong Cao. Hir-diff: Unsupervised hyperspectral image restoration via
improved diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3005–3014, 2024. [Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Pang_HIR-Diff_Unsupervised_Hyperspectral_Image_Restoration_Via_Improved_Diffusion_Models_CVPR_2024_paper.html)
- Xiangrong Zhang, Shunli Tian, Guanchun Wang, Huiyu Zhou, and Licheng Jiao. Diffucd: Unsupervised hyperspectral image change detection
with semantic correlation diffusion model. arXiv preprint arXiv:2305.12410, 2023. [Paper](https://arxiv.org/abs/2305.12410)
- Sam L Polk, Kangning Cui, Aland HY Chan, David A Coomes, Robert J Plemmons, and James M Murphy. Unsupervised diffusion and volume
maximization-based clustering of hyperspectral images. Remote Sensing, 15(4):1053, 2023. [Paper](https://www.mdpi.com/2072-4292/15/4/1053)
### Change detection
- Meiqi Hu, Chen Wu, Bo Du, and Liangpei Zhang. Binary change guided hyperspectral multiclass change detection. IEEE Transactions on Image Processing, 32:791–806, 2023. [Paper](https://ieeexplore.ieee.org/document/10011164)
- Bo Du, Lixiang Ru, Chen Wu, and Liangpei Zhang. Unsupervised deep slow feature analysis for change detection in multi-temporal remote sensing images. IEEE Transactions on Geoscience and Remote Sensing, 57(12):9976–9992, 2019. [Paper](https://ieeexplore.ieee.org/document/8824216)
- Meiqi Hu, Chen Wu, and Liangpei Zhang. Hypernet: Self-supervised hyperspectral spatial–spectral feature understanding network for hyperspectral change detection. IEEE Transactions on Geoscience and Remote Sensing, 60:1–17, 2022. [Paper](https://ieeexplore.ieee.org/document/9934933)
### Applications of HSI technology in wheat crops
### Wheat Crop Classification
- Xiu Jin, Lu Jie, Shuai Wang, Hai Jun Qi, and Shao Wen Li. Classifying wheat hyperspectral pixels of healthy heads and fusarium head blight disease using a deep neural network in the wild field. Remote Sensing, 10(3):395, 2018. [Paper](https://www.mdpi.com/2072-4292/10/3/395)
- Kshitiz Dhakal, Upasana Sivaramakrishnan, Xuemei Zhang, Kassaye Belay, Joseph Oakes, Xing Wei, and Song Li. Machine learning analysis of hyperspectral images of damaged wheat kernels. Sensors, 23(7):3523, 2023. [Paper](https://www.mdpi.com/1424-8220/23/7/3523)
- Erik Schou Dreier, Klavs Martin Sorensen, Toke Lund-Hansen, Birthe Møller Jespersen, and Kim Steenstrup Pedersen. Hyperspectral imaging for classification of bulk grain samples with deep convolutional neural networks. Journal of Near Infrared Spectroscopy, 30(3):107–121, 2022. [Paper](https://journals.sagepub.com/doi/abs/10.1177/09670335221078356)
- Dongyan Zhang, Gao Chen, Huihui Zhang, Ning Jin, Chunyan Gu, Shizhuang Weng, Qian Wang, and Yu Chen. Integration of spectroscopy and image for identifying fusarium damage in wheat kernels. Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy, 236:118344, 2020. [Paper](https://www.sciencedirect.com/science/article/abs/pii/S138614252030322X)
- Etienne David, Simon Madec, Pouria Sadeghi-Tehran, Helge Aasen, Bangyou Zheng, Shouyang Liu, Norbert Kirchgessner, Goro Ishikawa, Koichi Nagasawa, Minhajul A Badhon, et al. Global wheat head detection (gwhd) dataset: a large and diverse dataset of high-resolution rgb-labelled images to develop and benchmark wheat head detection methods. Plant Phenomics, 2020. [Paper](https://spj.science.org/doi/full/10.34133/2020/3521852?adobe_mc=MCMID%3D13000678418609464879081490540568399952%7CMCORGID%3D242B6472541199F70A4C98A6%2540AdobeOrg%7CTS%3D1670889600)
- Surabhi Lingwal, Komal Kumar Bhatia, and Manjeet Singh Tomer. Image-based wheat grain classification using convolutional neural network. Multimedia Tools and Applications, pages 1–25, 2021. [Paper](https://link.springer.com/article/10.1007/s11042-020-10174-3)
- Kadir Sabanci, Ahmet Kayabasi, and Abdurrahim Toktas. Computer vision-based method for classification of wheat grains using artificial neural network. Journal of the Science of Food and Agriculture, 97(8):2588–2593, 2017. [Paper](https://scijournals.onlinelibrary.wiley.com/doi/abs/10.1002/jsfa.8080)
- Wei Hu, Yangyu Huang, Li Wei, Fan Zhang, and Hengchao Li. Deep convolutional neural networks for hyperspectral image classification. Journal of Sensors, 2015(1):258619, 2015. [Paper]()
- Viktor Slavkovikj, Steven Verstockt, Wesley De Neve, Sofie Van Hoecke, and Rik Van de Walle. Hyperspectral image classification with convolutional neural networks. In Proceedings of the 23rd ACM international conference on Multimedia, pages 1159–1162, 2015. [Paper](https://onlinelibrary.wiley.com/doi/full/10.1155/2015/258619)
- Yidan Bao, Chunxiao Mi, Na Wu, Fei Liu, and Yong He. Rapid classification of wheat grain varieties using hyperspectral imaging and chemometrics. Applied Sciences, 9(19):4119, 2019. [Paper]()
- Kemal Özkan, SEKE Erol, and IŞIK Şahin. Wheat kernels classification using visible-near infrared camera based on deep learning. Pamukkale Üniversitesi Mühendislik Bilimleri Dergisi, 27(5):618–626, 2021. [Paper](https://www.mdpi.com/2076-3417/9/19/4119)
- Lv Yipeng, Lv Wenbing, Han Kaixuan, Tao Wentao, Zheng Ling, Weng Shizhuang, and Huang Linsheng. Determination of wheat kernels damaged by fusarium head blight using monochromatic images of effective wavelengths from hyperspectral imaging coupled with an architecture self-search deep network. Food Control, 135:108819, 2022. [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0956713522000123)
- Jingwu Zhu, Hao Li, Zhenhong Rao, and Haiyan Ji. Identification of slightly sprouted wheat kernels using hyperspectral imaging technology and different deep convolutional neural networks. Food Control, 143:109291, 2023. [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0956713522004844)
- Haotian Que, Xin Zhao, Xiulan Sun, Qibing Zhu, and Min Huang. Identification of wheat kernel varieties based on hyperspectral imaging technology and grouped convolutional neural network with feature intervals. Infrared Physics & Technology, 131:104653, 2023. [Paper](https://www.sciencedirect.com/science/article/abs/pii/S1350449523001111)
### Wheat Crop Nutrient Estimation
- Junjie Ma, Bangyou Zheng, and Yong He. Applications of a hyperspectral imaging system used to estimate wheat grain protein: A review. Frontiers in Plant Science, 13:837200, 2022. [Paper](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2022.837200/full)
- Naiyue Hu, Wei Li, Chenghang Du, Zhen Zhang, Yanmei Gao, Zhencai Sun, Li Yang, Kang Yu, Yinghua Zhang, and Zhimin Wang. Predicting micronutrients of wheat using hyperspectral imaging. Food Chemistry, 343:128473, 2021. [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0308814620323359)
- Yufei Song, Guifa Teng, Yingchun Yuan, Tianzhen Liu, and Zhimei Sun. Assessment of wheat chlorophyll content by the multiple linear regression of leaf image features. Information processing in Agriculture, 8(2):232–243, 2021. [Paper](https://www.sciencedirect.com/science/article/pii/S2214317319303038)
- Baohua Yang, Jifeng Ma, Xia Yao, Weixing Cao, and Yan Zhu. Estimation of leaf nitrogen content in wheat based on fusion of spectral features and deep features from near infrared hyperspectral imagery. Sensors, 21(2):613, 2021. [Paper](https://www.mdpi.com/1424-8220/21/2/613)
- Ghizlane Astaoui, Jamal Eddine Dadaiss, Imane Sebari, Samir Benmansour, and Ettarid Mohamed. Mapping wheat dry matter and nitrogen content dynamics and estimation of wheat yield using uav multispectral imagery machine learning and a variety-based approach: Case study of morocco. AgriEngineering, 3(1):29–49, 2021. [Paper](https://www.mdpi.com/2624-7402/3/1/3)
- Ning Lu, Yapeng Wu, Hengbiao Zheng, Xia Yao, Yan Zhu, Weixing Cao, and Tao Cheng. An assessment of multi-view spectral information from uav-based color-infrared images for improved estimation of nitrogen nutrition status in winter wheat. Precision Agriculture, 23(5):1653–1674, 2022. [Paper](https://link.springer.com/article/10.1007/s11119-022-09901-7) 
- Ruiqi Du, Junying Chen, Youzhen Xiang, Zhitao Zhang, Ning Yang, Xizhen Yang, Zijun Tang, Han Wang, Xin Wang, Hongzhao Shi, et al. Incremental learning for crop growth parameters estimation and nitrogen diagnosis from hyperspectral data. Computers and Electronics in Agriculture, 215:108356, 2023. [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0168169923007445)
### Wheat Crop Yield Estimation
- Enhui Cheng, Bing Zhang, Dailiang Peng, Liheng Zhong, Le Yu, Yao Liu, Chenchao Xiao, Cunjun Li, Xiaoyi Li, Yue Chen, et al. Wheat yield estimation using remote sensing data based on machine learning approaches. Frontiers in Plant Science, 13:1090970, 2022. [Paper](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2022.1090970/full)
- Ali Moghimi, Ce Yang, and James A Anderson. Aerial hyperspectral imagery and deep neural networks for high-throughput yield phenotyping in wheat. Computers and Electronics in Agriculture, 172:105299, 2020. [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0168169919320721)
- Yu Liu, Liang Sun, Binhui Liu, Yongfeng Wu, Juncheng Ma, Wenying Zhang, Bianyin Wang, and Zhaoyang Chen. Estimation of winter wheat yield using multiple temporal vegetation indices derived from uav-based multispectral and hyperspectral imagery. Remote Sensing, 15(19):4800, 2023. [Paper](https://www.mdpi.com/2072-4292/15/19/4800)
- Kai-Yun Li, Raul Sampaio de Lima, Niall G Burnside, Ele Vahtmäe, Tiit Kutser, Karli Sepp, Victor Henrique Cabral Pinheiro, Ming-Der Yang, Ants Vain, and Kalev Sepp. Toward automated machine learning-based hyperspectral image analysis in crop yield and biomass estimation. Remote Sensing, 14(5):1114, 2022. [Paper](https://www.mdpi.com/2072-4292/14/5/1114)
- Shuaipeng Fei, Muhammad Adeel Hassan, Yonggui Xiao, Awais Rasheed, Xianchun Xia, Yuntao Ma, Luping Fu, Zhen Chen, and Zhonghu He. Application of multi-layer neural network and hyperspectral reflectance in genome-wide association study for grain yield in bread wheat. Field Crops Research, 289:108730, 2022. [Paper](https://www.sciencedirect.com/science/article/abs/pii/S037842902200301X)
- Yucun Yang, Rui Nan, Tongxi Mi, Yingxin Song, Fanghui Shi, Xinran Liu, Yunqi Wang, Fengli Sun, Yajun Xi, and Chao Zhang. Rapid and nondestructive evaluation of wheat chlorophyll under drought stress using hyperspectral imaging. International Journal of Molecular Sciences, 24(6):5825, 2023. [Paper](https://www.mdpi.com/1422-0067/24/6/5825)
- Shaohua Zhang, Xinghui Qi, Jianzhao Duan, Xinru Yuan, Haiyan Zhang, Wei Feng, Tiancai Guo, and Li He. Comparison of attention mechanismbased deep learning and transfer strategies for wheat yield estimation using multisource temporal drone imagery. IEEE Transactions on Geoscience and Remote Sensing, 62:1–23, 2024. [Paper](https://ieeexplore.ieee.org/document/10538477)
- Ruomei Zhao, Weijie Tang, Mingjia Liu, Nan Wang, Hong Sun, Minzan Li, and Yuntao Ma. Spatial-spectral feature extraction for in-field chlorophyll content estimation using hyperspectral imaging. Biosystems Engineering, 246:263–276, 2024. [Paper](https://www.sciencedirect.com/science/article/abs/pii/S1537511024001867)
- Tao Liu, Tianle Yang, Shaolong Zhu, Nana Mou, Weijun Zhang, Wei Wu, Yuanyuan Zhao, Zhaosheng Yao, Jianjun Sun, Chen Chen, et al. Estimation of wheat biomass based on phenological identification and spectral response. Computers and Electronics in Agriculture, 222:109076, 2024. [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0168169924004678)
### Wheat Crop Disease Monitoring and Detection
- Long Wan, Hui Li, Chengsong Li, Aichen Wang, Yuheng Yang, and Pei Wang. Hyperspectral sensing of plant diseases: Principle and methods. Agronomy, 12(6):1451, 2022. [Paper](https://www.mdpi.com/2073-4395/12/6/1451)
- Guoqing Feng, Ying Gu, Cheng Wang, Yanan Zhou, Shuo Huang, and Bin Luo. Wheat fusarium head blight automatic non-destructive detection based on multi-scale imaging: A technical perspective. Plants, 13(13):1722, 2024. [Paper](https://www.mdpi.com/2223-7747/13/13/1722)
- Anton Terentev, Vladimir Badenko, Ekaterina Shaydayuk, Dmitriy Emelyanov, Danila Eremenko, Dmitriy Klabukov, Alexander Fedotov, and Viktor Dolzhenko. Hyperspectral remote sensing for early detection of wheat leaf rust caused by puccinia triticina. Agriculture, 13(6):1186, 2023. [Paper](https://www.mdpi.com/2077-0472/13/6/1186)
- Wijayanti Nurul Khotimah, Mohammed Bennamoun, Farid Boussaid, Lian Xu, David Edwards, and Ferdous Sohel. Mce-st: Classifying crop stress using hyperspectral data with a multiscale conformer encoder and spectral-based tokens. International Journal of Applied Earth Observation and Geoinformation, 118:103286, 2023. [Paper](https://www.sciencedirect.com/science/article/pii/S1569843223001085)
- Imran Haider Khan, Haiyan Liu, Wei Li, Aizhong Cao, Xue Wang, Hongyan Liu, Tao Cheng, Yongchao Tian, Yan Zhu, Weixing Cao, et al. Early detection of powdery mildew disease and accurate quantification of its severity using hyperspectral images in wheat. Remote Sensing, 13(18):3612, 2021. [Paper](https://www.mdpi.com/2072-4292/13/18/3612)
- Muhammad Baraa Almoujahed, Aravind Krishnaswamy Rangarajan, Rebecca L Whetton, Damien Vincke, Damien Eylenbosch, Philippe Vermeulen, and Abdul M Mouazen. Detection of fusarium head blight in wheat under field conditions using a hyperspectral camera and machine learning. Computers and Electronics in Agriculture, 203:107456, 2022. [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0168169922007645)
- Aravind Krishnaswamy Rangarajan, Rebecca Louise Whetton, and Abdul Mounem Mouazen. Detection of fusarium head blight in wheat using hyperspectral data and deep learning. Expert Systems with Applications, 208:118240, 2022. [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417422013872)
- Zi-Heng Feng, Lu-Yuan Wang, Zhe-Qing Yang, Yan-Yan Zhang, Xiao Li, Li Song, Li He, Jian-Zhao Duan, and Wei Feng. Hyperspectral monitoring of powdery mildew disease severity in wheat based on machine learning. Frontiers in Plant Science, 13:828454, 2022. [Paper](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2022.828454/full)
- Laixiang Xu, Bingxu Cao, Fengjie Zhao, Shiyuan Ning, Peng Xu, Wenbo Zhang, and Xiangguan Hou. Wheat leaf disease identification based on deep learning algorithms. Physiological and Molecular Plant Pathology, 123:101940, 2023. [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0885576522001552)
- Igor Sereda, Roman Danilov, Oksana Kremneva, Mikhail Zimin, and Yuri Podushin. Development of methods for remote monitoring of leaf diseases in wheat agrocenoses. Plants, 12(18):3223, 2023. [Paper](https://www.mdpi.com/2223-7747/12/18/3223)
- Gerald Blasch, Tadesse Anberbir, Tamirat Negash, Lidiya Tilahun, Fikrte Yirga Belayineh, Yoseph Alemayehu, Girma Mamo, David P Hodson, and Francelino A Rodrigues Jr. The potential of uav and very high-resolution satellite imagery for yellow and stem rust detection and phenotyping in ethiopia. Scientific Reports, 13(1):16768, 2023. [Paper](https://www.nature.com/articles/s41598-023-43770-y)
- Shenglong Chang, Guijun Yang, Jinpeng Cheng, Ziheng Feng, Zehua Fan, Xinming Ma, Yong Li, Xiaodong Yang, and Chunjiang Zhao. Recognition of wheat rusts in a field environment based on improved densenet. Biosystems Engineering, 238:10–21, 2024. [Paper](https://www.sciencedirect.com/science/article/abs/pii/S1537511023002696)
