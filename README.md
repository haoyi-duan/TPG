# TPG

### **Timestamps as Prompts for the Geography-aware Location Recommendation** 

###### *CIKM*'2023



## Introduction

Our team revisited the problem of location recommendation and pointed out that temporal information of POI was indispensable in real-world applications, and that current methods did not make effective use of geographic information but suffered from the hard boundary problem when encoding geographic information by gridding. Therefore, we proposed a Temporal Prompt-based and Geography-aware (TPG) framework which has the unique ability of interval prediction. Our contributions are as follows:

- **Temporal prompt** is firstly designed to incorporate temporal information of next location. 
- **Shifted window** is then devised to augment geographic data for addressing the hard boundary problem. 

Via extensive comparisons with existing methods and ablation studies on four real-world datasets, we demonstrate the effectiveness and superiority of the proposed method under various settings. Most importantly, our proposed model has the unique ability of interval prediction, i.e., predicting the location that a user wants to go to at a certain time while the most recent check-in behavioral data is masked. The experimental results on four benchmark datasets demonstrated the superiority of TPG comparing with other state-of-the-art methods. The results indicated that temporal signal of next location is of great significance. We also demonstrate through ablation studies that our proposed shifted window mechanism is capable of overcoming defects of previous approaches.

![image-20221118004807537](images/image-20221118004807537.png)

<center style="color:#000000">Figure 1.</center>

Figure 1. is an illustration of how **TPG** performs **next location recommendation** (denoted by purple line) and **interval prediction** (denoted by red lines) tasks. Different colors of markers denote different categories of POIs. Given the user historical check-in sequence is POI 1-6 from Wednesday to Thursday, the model can know the next four locations the user will go are POI 1 at 5:43 Friday, POI 4 at 12:00 Friday, POI 7 at 9:08 Saturday, and POI 8 at 14:45 Saturday. Predicting POI 1 at 5:43 Friday is the task of next location recommendation. By making use of temporal prompt, TPG can also predict the location that a user wants to go at a certain time (i.e., interval prediction). For example, the model can predict POI 4 at 12:00 Friday (interval 1), POI 7 at 9:08 Saturday (interval 2), and POI 8 at 14:45 Saturday (interval 3), only based on historical check-in sequence POI 1-6.

## Framework

![image-20221118011716980](images/frame.png)

<center style="color:#000000">Figure 2.</center>

The overall architecture of our TPG framework is described in Figure 2. Based on the transformer's encoder-decoder structure, TPG can be divided into three parts, i.e., **geography-aware encoder**, **history encoder**, and **temporal prompt-based decoder**. 

## Requirements

```c++
pip3 install -r requirements.txt
```

## Usage

1. Clone this repo

   ```
   git clone https://github.com/haoyi-duan/TPG.git
   ```

2. Training

   ```c
   cd ./Bigscity-LibCity
   python run_model.py --task traj_loc_pred --dataset foursquare_nyc --model TPG
   ```

## Acknowledgement

Our code was based on the ``Unified Library and Performance Benchmark`` -- LibCity. 

## Citation

```
TPG citation
```

LibCity citation

```
@article{libcitylong,
  title={Towards Efficient and Comprehensive Urban Spatial-Temporal Prediction: A Unified Library and Performance Benchmark}, 
  author={Jingyuan Wang and Jiawei Jiang and Wenjun Jiang and Chengkai Han and Wayne Xin Zhao},
  journal={arXiv preprint arXiv:2304.14343},
  year={2023}
}
```

