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
```
i = np.random.randint(1, len(keyfacial_df))
plt.imshow(keyfacial_df['Image'][i], cmap = 'gray')
for j in range (1, 31, 2):
    plt.plot(keyfacial_df.loc[i][j-1], keyfacial_df.loc[i][j], 'rx')
```
![Screenshot (69)](https://user-images.githubusercontent.com/70371572/130304626-e9302820-c31e-48a4-bef2-e3262972d860.png)

## TASK 4. Image Augmentation
We want to Horizintally flip the images i.e.  flip the images along y axis.This means y coordinate values would be the same and only x coordiante values would change, we subtract our initial x-coordinate values from width of the image(96).

### Original Image and Horizonatlly flipped Image
![Screenshot (70)](https://user-images.githubusercontent.com/70371572/130325501-4adbae90-050e-4483-bc23-ca4a11eb225d.png)
Now we do this for every image in our dataframe and then concatenate the original dataframe with the augmented dataframe. We can also try experimenting with other ways to change the image by increasing or decreasing brightness which can be achieved by multiplying pixel values by random values between 1.5 and 2.

## TASK 5. Perform Data Normalisation & Training Data Preparation

Now we normalise the images by dividing the value present in 31st column by 255. Then we create an empty array of shape (x, 96, 96, 1) to feed the model
```X = np.empty((len(img), 96, 96, 1))```
Then we iterate through the img list and add image values to the empty array after expanding it's dimension from (96, 96) to (96, 96, 1)
```
for i in range(len(img)):
  X[i,] = np.expand_dims(img[i], axis = 2)
```
Now we split the data into 80 % training and 20% testing. 
```X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)```

## TASK 6. Understanding Theory behind NEURAL NETWORKS, GRADIENT DESCENT ALGORITHM, CONVOLUTIONAL NEURAL NETWORKS AND RESNETS

![Screenshot (71)](https://user-images.githubusercontent.com/70371572/130325538-6f293de7-1939-47d5-9051-bde20589b82e.png)
- Artificial Neuron replicates the Human Neuron using by accepting input, assigning weights to connect various layers and produces an output.
- All multiple neurons are connected in a multi-layer fashion.
- The more in-between or hidden layers there are the more "deep" the network will get.

Dividing data into Training and Testing
![Screenshot (72)](https://user-images.githubusercontent.com/70371572/130325563-ee69fa80-9231-4385-9bc6-0d946779c0ae.png)
- generally 80% training and 20% testing
- for cross validation we can divide into 60% training, 20% validation and 20% testing.
- Training set is usually used for gradient calculation and weight update.
- Validation is used for cross validation to overcome the overfitting problem which occurs when algorithm focuses on training set details and starts to lose the generalisation ability.

Gradient Descent 
![Screenshot (73)](https://user-images.githubusercontent.com/70371572/130325579-2279d1f1-a032-48cc-bc83-a6c6782a22a0.png)

- Optimization algorithm to obtain the optimised network weight and bias values.
- works using iteration to minimize the cost function.
- it calculates the gradient of cost function and moving in the negative direction until global minimum is achieved.
- The size of the steps taken are called the learning rate.
- If learning rate increases then the area covered in te search space will increase so we might reach global minimum faster.
- For smaller learning rates training will take much longer to reach optimized weight values.

Convolutional Neural Networks
![Screenshot (75)](https://user-images.githubusercontent.com/70371572/130325609-e14c4f0b-e6ee-4838-ad24-9a16a56f40b7.png)

RESNET(Residual Network)
- As CNNs grow deeper, vanishing gradient tend to occur which negatively impacts performance.
- Residual Neural Network includes "skip connection" feature that enables training of 152 layers without vanishing Gradient Issue.
- This is done by identity mapping on top of CNN.
- ![Screenshot (80)](https://user-images.githubusercontent.com/70371572/130325757-6267fb7e-6fae-42fd-961b-22a798da1812.png)
![Screenshot (79)](https://user-images.githubusercontent.com/70371572/130325769-3b8a1f58-8921-4618-8596-a9a2449f87ba.png)
![Screenshot (78)](https://user-images.githubusercontent.com/70371572/130325706-fbc60a8e-08a5-4ab7-844f-8f36cb7c131b.png)

Now we can build the Deep Residual Neural Network for Key Facial points detection model
We replicate the above flow of control by using:
- ResBlock: which has 3 parts namely Convolution Block, Identity block, Identity block.
Has two paths Main Path and Short Path. These consists of Convolution (Conv2D with filter (1,1) shift by pixels) followed by MaxPool2D, BatchNormalization, and Relu Activation Function in different orders. After the paths execute we add the results and feed this as an input to the indentity block.

- Then after 3 stages of performing Convolution, BatchNormalization and calling ResBlock, we use dropout which is a regularisation technique used to ensure generalisation capability of the network. Using 
```X = Dropout(0.2)(X)```
We drop 20% of the Neurons, the network avoids having interdependency between the neurons able to improve the network performance.

![Screenshot (81)](https://user-images.githubusercontent.com/70371572/130328084-d428e5de-97d8-475f-b860-f58fcb3730dc.png)
![Screenshot (82)](https://user-images.githubusercontent.com/70371572/130328080-079b733a-a1f2-4a46-b8f8-969f157563cd.png)
