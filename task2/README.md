# Task 2
## Set up
To install all needed libraries you should run 
```
pip install -r requirements.txt
```
Actually there will be one more installation(CIFAR-100 dataset) but as I run it in Google collab and it is needed only for model tuning.
## Files Content
### annotation.json
Stores labeled entities in text data. (More details in `ner.ipynb`)
### demo.ipynb
Demonstration of pipeline in successful case, in the case of failure, and in the case when a given class is not among the trained ones.
### image.png
Just for debug ;-)
### image_processing.ipynb
Here I find desired classes, clear dataset, build and tune model.(More details are descripted there)
### main.py
Here is code for demo but Im not sure that it was good idea remain it all as functions.
### ner.ipynb
Here I process users input and retreive animal class.(More information there)
### ner_train_data.txt
Data created by ChatGPT and labeled by me for creating annotations ![here](https://arunmozhi.in/ner-annotator/)
