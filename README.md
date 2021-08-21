# Emotion_Detection
I have worked on a project to classify people's emotions based on their facial images using AI. This project covers building and training of a system that automatically classifies peoples expressions. We have more than 20000 facial images with their associated facial expression labels and around 2000 images with their facial key-points annotations. We will essentially feed these images to our Emotion AI model which can 1) Predict which emotion the person is shwoing and 2) Predict key facial points of the person.

## Part 1.Facial Key Points Detection
In this section we will create a deep learning model based on Convolutional Neural Network and Residual Blocks to predict facial key-points.
- Data set consists of x and y coordinates of 15 facial key points.
- Images are 96 by 96 pixes.
- Images are gray scaled.


## Part 2.Import Libraries & Analyse Datasets
We would need the following libraries for visualisation of images and training our model
- matplotlib
- tensorflow
- sklearn
- cv2
- seaborn
- numpy
- pandas

We load the dataset to look at the columns which the various coordinates of the facial features which uses pixels to specify various facial key points like right eye corner, mouth corners, etc.
![Screenshot (62)](https://user-images.githubusercontent.com/70371572/130304115-925d1da2-26d2-4a2a-aa5b-35fef44c0260.png)
![Screenshot (63)](https://user-images.githubusercontent.com/70371572/130304141-46661b78-eeea-4f74-b6ff-e22a7b0f137e.png)
![Screenshot (64)](https://user-images.githubusercontent.com/70371572/130304149-4db162a4-61fd-4249-ad63-ca34d559619f.png)

Now we check the different data types associated with each columns using keyfacial_df.info().
![Screenshot (65)](https://user-images.githubusercontent.com/70371572/130304248-41b43f0a-e9ff-4644-ae2e-5b89f7fe3958.png)

We also check for any null values using keyfacial_df.isnull().sum().
![Screenshot (66)](https://user-images.githubusercontent.com/70371572/130304248-41b43f0a-e9ff-4644-ae2e-5b89f7fe3958.png)
