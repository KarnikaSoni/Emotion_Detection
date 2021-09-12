# Emotion_Detection
I have worked on a project to classify people's emotions based on their facial images using AI. This project covers building and training of a system that automatically classifies peoples expressions. We have more than 20000 facial images with their associated facial expression labels and around 2000 images with their facial key-points annotations. We will essentially feed these images to our Emotion AI model which can 1) Predict which emotion the person is showing and 2) Predict key facial points of the person.

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

We also check for max, min and verage values for all 15 facial points using ```keyfacial_df.describe()```
![Screenshot (67)](https://user-images.githubusercontent.com/70371572/130304391-94ffdc0d-8478-4b17-a451-0f22e06e254c.png)

## Part 3.Perform Image Visualization
We plot a random image from the dataset along with facial keypoints. Image data is obtained from df['Image'] and plotted using plt.imshow. 15 x and y coordinates are marked for the corresponding image with the information that x-coordinates are in even columns like 0,2,4 and y-coordinates are in odd columns like 1,3,5. We access their value using .loc command that gets the values for coordinates of the image based on the column it is referring to.

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

## Part 4. Image Augmentation
We want to Horizontally flip the images i.e.  flip the images along y axis.This means y coordinate values would be the same and only x co-ordinate values would change, we subtract our initial x-coordinate values from width of the image(96).

### Original Image and Horizontally flipped Image
![Screenshot (70)](https://user-images.githubusercontent.com/70371572/130325501-4adbae90-050e-4483-bc23-ca4a11eb225d.png)
Now we do this for every image in our dataframe and then concatenate the original dataframe with the augmented dataframe. We can also try experimenting with other ways to change the image by increasing or decreasing brightness which can be achieved by multiplying pixel values by random values between 1.5 and 2.

## Part 5. Perform Data Normalisation & Training Data Preparation

Now we normalise the images by dividing the value present in 31st column by 255. Then we create an empty array of shape (x, 96, 96, 1) to feed the model
```X = np.empty((len(img), 96, 96, 1))```
Then we iterate through the img list and add image values to the empty array after expanding it's dimension from (96, 96) to (96, 96, 1)
```
for i in range(len(img)):
  X[i,] = np.expand_dims(img[i], axis = 2)
```
Now we split the data into 80 % training and 20% testing. 
```X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)```

## Part 6. Understanding Theory behind Neural Networks, Gradient Descent Algorithm, Convolutional Neural Networks & RESNETS

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

## Part 7. Train Key Facial Points Detection using Deep Learning Model
We use Adam Optimizer to compile and run. More information about Adam optimizer: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam. It uses Gradient Descent method. 

```checkpointer = ModelCheckpoint(filepath = "FacislKeyPoints_weigths.hdf5", verbose = 1, save_best_only = True)```
We check which model has the least validation loss, save only the best which would trained throughout the various epochs.
```history = model_1_facialKeyPoints.fit(X_train, y_train, batch_size = 32, epochs = 2, validation_split = 0.05, callbacks=[checkpointer])```
Fit the training data to the model with 5% validation data set. The loss decreases, accuracy increases, and weights are optimized to reduce the error.
![Screenshot (81)](https://user-images.githubusercontent.com/70371572/130328841-f8a166d0-2537-4c84-8f7d-041a4c2aaf1f.png)

### Assess the Performance
We run our Model on data that it has never seen before contained in X_test and y_test. The accuracy is 85%.
```
41/41 [==============================] - 5s 102ms/step - loss: 9.3887 - accuracy: 0.8575
Accuracy : 0.8419002890586853
```
We plot the loss for both training and validation for 2 epochs.
![Screenshot (85)](https://user-images.githubusercontent.com/70371572/130329023-c454f03e-a12b-4930-a03e-b3f671e8cb2f.png)

## Part 8. Explore Data for Facial Expression Detection, Image Augmentation 
```facialexpression_df.head()```
![Screenshot (86)](https://user-images.githubusercontent.com/70371572/130329199-e5e20be7-7cbb-43e3-84c5-d8b9a581248c.png)
Now we visualize 1 image of each emotion category:
![Screenshot (88)](https://user-images.githubusercontent.com/70371572/130329242-7348af76-c1a6-4053-8c05-bf8608d63f76.png)
![Screenshot (87)](https://user-images.githubusercontent.com/70371572/130329246-29ff2322-ac0a-4d94-8877-b7ea3a0b43cf.png)
![Screenshot (89)](https://user-images.githubusercontent.com/70371572/130329252-fe37c473-a882-4387-ad1a-542e9e157c8a.png)

Now we plot the number of images we have for each emotion to see if the data set is balanced or not.
![Screenshot (90)](https://user-images.githubusercontent.com/70371572/130329305-2cc0780a-1759-4995-aad9-d590d6c3b571.png)

As emotion 1 is significantly lower then other 4, it is an unbalanced dataset.

Then we do image pre-processing
```
X_train = X_train/255
X_val   = X_val /255
X_Test  = X_Test/255
```
### Train Deep Learning Model for FACIAL EXPRESSION CLASSIFICATION
After this we perform training of the model using identical steps from Task 6 and 7.
```model_2_emotion.summary()```
![Screenshot (91)](https://user-images.githubusercontent.com/70371572/130329505-4e1c3f1a-ba35-4c6b-bf0c-3ac73da29871.png)
![Screenshot (92)](https://user-images.githubusercontent.com/70371572/130329506-6840976d-f328-4692-8c41-f52c69bbc99f.png)

Then we Adam optimizer again and then save the best model.
```
Epoch 1/2
345/345 [==============================] - ETA: 0s - loss: 1.2713 - accuracy: 0.4773
Epoch 00001: val_loss improved from inf to 1.49306, saving model to FacialExpression_weights.hdf5
345/345 [==============================] - 433s 1s/step - loss: 1.2713 - accuracy: 0.4773 - val_loss: 1.4931 - val_accuracy: 0.3713
Epoch 2/2
345/345 [==============================] - ETA: 0s - loss: 0.9051 - accuracy: 0.6389
Epoch 00002: val_loss improved from 1.49306 to 1.03988, saving model to FacialExpression_weights.hdf5
345/345 [==============================] - 434s 1s/step - loss: 0.9051 - accuracy: 0.6389 - val_loss: 1.0399 - val_accuracy: 0.5806
```

### Assess the performance of TRAINED FACIAL EXPRESSION CLASSIFIER MODEL
We run our Model on data that it has never seen before contained in X_test and y_test. The accuracy is 87%.
```
39/39 [==============================] - 5s 130ms/step - loss: 0.3553 - accuracy: 0.8706
Test Accuracy: 0.8706265091896057
```
We plot the loss for validation accuracy for 2 epochs.
![Screenshot (93)](https://user-images.githubusercontent.com/70371572/130334837-fe4a0ab0-584f-4b5e-8cd6-2410c5d426b8.png)

We print the confusion matrix for predicted and actual emotions. This will provide a summary for the overall model performance.
![Screenshot (94)](https://user-images.githubusercontent.com/70371572/130334892-648a439c-9684-4d4a-a893-fe4860f3a2da.png)

The diagonal indicates the correct classification of expressions and emotions.
We can also see this visually by plotting images with their actual and predicted emotions.
![Screenshot (95)](https://user-images.githubusercontent.com/70371572/130334921-ab2d4abe-9c7c-447f-a814-672822219be3.png)
Now we print the classification report using sklearn.
![Screenshot (96)](https://user-images.githubusercontent.com/70371572/130334942-53988610-6772-41b4-b363-bd3708a077c0.png)

## Part 9. Results
### Combine both models (1) FACIAL KEYPOINTS DETECTION & (2) FACIAL EXPRESSION MODEL
We combine 2 models by assigning the df-Predict of model_1 add df_emotion in a single dataframe. 
``` df_predict.head()```
![Screenshot (99)](https://user-images.githubusercontent.com/70371572/130335152-07d246ce-a79b-4b09-a97d-20cffad1e913.png)

We can see the results of this combined model which plots both predicted emotions and key facial points.
![Screenshot (97)](https://user-images.githubusercontent.com/70371572/130335181-005fcfc1-fc9c-4a85-b0a8-73da3f6101b9.png)


Note: This project was with help of online Udemy course.
