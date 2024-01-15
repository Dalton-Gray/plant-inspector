# Plant Inspector Artificial Neural Network

#### Overview

I trained a efficientnets and mobilenets on PlantVillage dataset (PVD) (citation ), PlantDoc dataset (PDD) [citation], FieldPlant dataset [citation], matching the performance reported for efficientnet b0 in (pairwise).

When training on PVD I noticed that even the smallest available models would quickly achieve 99% accuracy. This can be explained by the dataset bias present in the dataset whereby the background alone of the image can be used to predict the image class with a high degree of accuracy [https://ui.adsabs.harvard.edu/abs/2022arXiv220604374A/abstract]. I'd like to experiment with data augmentation techniques like image segmentation and cropping to see if they can account for this bias. Newly collected data should also take care not to replicate this mistake, as it causes models that perform well in practice but bad in the field. 


Models train on PDD struggle to reach above 70% accuracy, even models like resnet50, which were much larger than those in the paper. There are also clearly some data quality issues since the data was collected through web scraping rather than expert plant pathologist collection and labelling.

[ IMAGES HERE]


Fig 1: image showing data quality issues


Additionally, the dataset is an order of magnitude smaller than PVD, coming in at only 2500 images. We can see based on this graph that the efficientnet b0 has not fully saturated and accuracy continues to trend upwards as we add more data.


[graph here]

fig 2: 



Limitations

1. due to time constraints I did not have a chance to review all datasets, so it's possible some of these problems have been addressed. However, this project has given me a deeper understanding of some of the potential pitfalls one could face in an ML plant pathology project and it will only aid me in digging deeper into the field







#### Evaluation

| col1 | col2 | col3 |
| ---- | ---- | ---- |
|      |      |      |
|      |      |      |

#### References

1. PlantDoc dataset: https://arxiv.org/abs/1911.10317
2. 

---

#### Project setup

1. Create the conda envinronment `conda create -n plant_inspector python=3.10`
2. Install Python dependencies `pip install -r requirements.txt`
3. Download the PlantDoc dataset `git clone https://github.com/pratikkayal/PlantDoc-Dataset.git`
4. Run the program `python app.py`
