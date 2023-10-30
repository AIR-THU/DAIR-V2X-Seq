<div align="center">   
  
# V2X-Seq: A Large-Scale Sequential Dataset for  Vehicle-Infrastructure Cooperative Perception and Forecasting, CVPR2023
</div> 

<h3 align="center">
    <a href="https://thudair.baai.ac.cn/index">Project Page</a> |
    <a href="#dataset">Dataset Download</a> |
    <a href="https://arxiv.org/abs/2305.05938">arXiv</a> |
    <a href="https://github.com/AIR-THU/DAIR-V2X-Seq/">V2X-Seq</a> 
</h3>

  ![](./resources/tracking-demo.gif "")
  ![](./resources/forecasting-demo.gif "")

## Table of Contents:
1. [Introduction](#introduction)
2. [News](#news)
3. [Dataset Download](#dataset)
4. [Getting Started](#start)
5. [Benchmark](#benchmark)
6. [TODO List](#todo)
7. [Citation](#citation)
8. [Contaction](#contaction)

## Introduction <a name="introduction"></a>
V2X-Seq is the first large-scale, real-world, and sequential V2X dataset, which includes data frames, trajectories, vector maps, and traffic lights captured from natural scenery. V2X-Seq comprises two parts: the sequential perception dataset, which includes more than 15,000 frames captured from 95 scenarios, and the trajectory forecasting dataset, which contains about 80,000 infrastructure-view scenarios, 80,000 vehicle-view scenarios, and 50,000 cooperative-view scenarios captured from 28 intersections' areas, covering 672 hours of data.

## News <a name="news"></a>
* [2023.09] We have released the code for V2X-Seq, including Sequential Perception Dataset (SPD) and Trajectory Forecasting Dataset (TFD).
* [2023.05] V2X-Seq dataset is availale [here](https://thudair.baai.ac.cn/index). It can be unlimitedly downloaded within mainland China. Example dataset can be downloaded directly. 
* [2023.03] Our new dataset "V2X-Seq: A Large-Scale Sequential Dataset for Vehicle-Infrastructure Cooperative Perception and Forecasting" has been accepted by CVPR2023. Congratulations! We will release the dataset sooner. 

## Dataset Download <a name="dataset"></a>

V2X-Seq is one of our [DAIR-V2X dataset series](https://thudair.baai.ac.cn/index). Example dataset can be downloaded directly. Download links are as follows:

- Sequential Perception Dataset (SPD): [dataset download link](https://thudair.baai.ac.cn/coop-forecast).
- Trajectory Forecasting Dataset (TFD): [dataset download link](https://thudair.baai.ac.cn/cooplocus).
- Example Dataset: [SPD-Example](https://drive.google.com/file/d/1gjOmGEBMcipvDzu2zOrO9ex_OscUZMYY/view?usp=drive_link), [TFD-Example](https://drive.google.com/file/d/1vV2BZvBWkum-j0r82JOjAajlSWB7kyU2/view?usp=sharing).

## Getting Started <a name="start"></a>
Please refer to [getting_started.md](docs/get_started.md) for usage of V2X-Seq dataset and benchmarks reproduction.

## Benchmark <a name="benchmark"></a>
- [SPD Benchmark](https://github.com/AIR-THU/DAIR-V2X/configs/vic3d-tracking)
- [TFD Benchmark](docs/benchmarks/vic-traj-forecasting)

## TODO List <a name="todo"></a>
- [x] Dataset release
- [x] Dataset API
- [x] Evaluation code
- [ ] Training&Evaluation code for VIC3D Tracking Benchmark
- [x] Training&Evaluation code for VIC Trajectory Forecasting

## Citation <a name="citation"></a>
Please consider citing our paper if the project helps your research with the following BibTex:
```bibtex
@inproceedings{v2x-seq,
  title={V2X-Seq: A large-scale sequential dataset for vehicle-infrastructure cooperative perception and forecasting},
  author={Yu, Haibao and Yang, Wenxian and Ruan, Hongzhi and Yang, Zhenwei and Tang, Yingjuan and Gao, Xu and Hao, Xin and Shi, Yifeng and Pan, Yifeng and Sun, Ning and Song, Juan and Yuan, Jirui and Luo, Ping and Nie, Zaiqing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023},
}
```
```bibtex
@inproceedings{dair-v2x,
  title={Dair-v2x: A large-scale dataset for vehicle-infrastructure cooperative 3d object detection},
  author={Yu, Haibao and Luo, Yizhen and Shu, Mao and Huo, Yiyi and Yang, Zebang and Shi, Yifeng and Guo, Zhenglong and Li, Hanyu and Hu, Xing and Yuan, Jirui and Nie, Zaiqing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21361--21370},
  year={2022}
}
```

## Contaction <a name="contaction"></a>

If any questions and suggenstations, please email to dair@air.tsinghua.edu.cn. 

## Related Resources <a name="related"></a>

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

- [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X) (:rocket:Ours!)
- [FFNET](https://github.com/haibao-yu/FFNet-VIC3D) (:rocket:Ours!)
- [argoverse-api](https://github.com/argoverse/argoverse-api)
- [TNT](https://github.com/pytorch/tnt)
- [HiVT](https://github.com/ZikangZhou/HiVT)