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

Now we check the different data types associated with each columns using ```keyfacial_df.info().```
![Screenshot (65)](https://user-images.githubusercontent.com/70371572/130304248-41b43f0a-e9ff-4644-ae2e-5b89f7fe3958.png)

We also check for any null values using ```keyfacial_df.isnull().sum().```
![Screenshot (66)](https://user-images.githubusercontent.com/70371572/130304280-7da8e0b1-eedb-40ac-a9fb-bbca702d8f08.png)

We also check for max, min and verage values for all 15 facial points using ```keyfacial_df.describe()```
![Screenshot (67)](https://user-images.githubusercontent.com/70371572/130304391-94ffdc0d-8478-4b17-a451-0f22e06e254c.png)

## Part 3.Perform Image Visulaisation
We plot a random image from the dataset along with facial keypoints. Image data is obtained from df['Image'] and plotted using plt.imshow. 15 x and y coordinates are marked for the corresponding image with the information that x-coordinates are in even columns like 0,2,4 and y-coordinates are in odd columns like 1,3,5. We access their value using .loc command that gets the values for coordinates of the image based on the column it is refering to.

```i = np.random.randint(1, len(keyfacial_df))
plt.imshow(keyfacial_df['Image'][i], cmap = 'gray')
for j in range(1, 31, 2):
        plt.plot(keyfacial_df.loc[i][j-1], keyfacial_df.loc[i][j], 'rx')
```
        
![Screenshot (68)](https://user-images.githubusercontent.com/70371572/130304485-91dd9924-4d83-4e80-82f3-f3a09f8f01c5.png)

Now we plot a grid to view more images
![Screenshot (69)](https://user-images.githubusercontent.com/70371572/130304626-e9302820-c31e-48a4-bef2-e3262972d860.png)

