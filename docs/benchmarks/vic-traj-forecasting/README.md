# Benchmark for Trajectory Forecasting Tasks on TFD Dataset

## Benchmark

| Using Inf. Trajectories | Base Model | minADE | minFDE |  MR  | Checkpoint | README |
| :-------------------------------: | :--------------: | :----: | :----: | :--: | :--: | :--: |
|                 ✕                 |       TNT        |  8.45  | 17.93  | 0.77 | [checkpoint](../../../projects/TNT_plugin/models/single_veh.pth) | [TNT README](./TNT/README.md)
|              Online               |       TNT        |  7.38  | 15.27  | 0.72 | [checkpoint](../../../projects/TNT_plugin/models/online.pth) |
|              Offline              |       TNT        |  4.38  |  9.37  | 0.62 | [checkpoint](../../../projects/TNT_plugin/models/offline.pth) |
|                 ✕                 |       HiVT       |  1.44  |  2.52  | 0.36 | [checkpoint](../../../projects/HiVT_plugin/checkpoints/single_veh.ckpt) | [HiVT README](./HiVT/README.md)
|              Online               |       HiVT       |  1.25  |  2.33  | 0.35 | [checkpoint](../../../projects/HiVT_plugin/checkpoints/online.ckpt) |
|              Offline              |       HiVT       |  1.39  |  2.23  | 0.32 | [checkpoint](../../../projects/HiVT_plugin/checkpoints/offline.ckpt) |


## Citation
```bibtex
@inproceedings{zhou2022hivt,
  title={Hivt: Hierarchical vector transformer for multi-agent motion prediction},
  author={Zhou, Zikang and Ye, Luyao and Wang, Jianping and Wu, Kui and Lu, Kejie},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8823--8833},
  year={2022}
}
```

```bibtex
@inproceedings{zhao2021tnt,
  title={Tnt: Target-driven trajectory prediction},
  author={Zhao, Hang and Gao, Jiyang and Lan, Tian and Sun, Chen and Sapp, Ben and Varadarajan, Balakrishnan and Shen, Yue and Shen, Yi and Chai, Yuning and Schmid, Cordelia and others},
  booktitle={Conference on Robot Learning},
  pages={895--904},
  year={2021},
  organization={PMLR}
}
```

```bibtex
@inproceedings{yu2023v2x,
  title={V2X-Seq: A Large-Scale Sequential Dataset for Vehicle-Infrastructure Cooperative Perception and Forecasting},
  author={Yu, Haibao and Yang, Wenxian and Ruan, Hongzhi and Yang, Zhenwei and Tang, Yingjuan and Gao, Xu and Hao, Xin and Shi, Yifeng and Pan, Yifeng and Sun, Ning and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5486--5495},
  year={2023}
}
```