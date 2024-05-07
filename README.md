# Hangul recognicion

This is moment 3 of my ML class

Dataset used: https://www.kaggle.com/datasets/wayperwayp/hangulkorean-characters/data

# Using my own model as seen in part3.py

Here is an example of the training/testing output:

![image of test data](https://github.com/Sunstrophe/Hangul-recogniction/blob/main/test_data.png?raw=true)

Comments:
- It's a pretty small data set which probably affected my score after making a simple model to train and test this data set on.
- The test data is on third of the training data which is definitely a factor in the results.

  
# Using transfer learning with resnet50

![image of data from resnet50](https://github.com/Sunstrophe/Hangul-recogniction/blob/main/resnet50_data.png?raw=true)

Comments:
- Massively more accurate
- Load time approximately 15min running on cpu
