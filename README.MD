 # Self Driving Car Simulation

## Overview

In this project, we will try to learn how to train a self driving car using Convolutional Neural Network. We will be using [Udacity Car Simulator](https://github.com/udacity/self-driving-car-sim) to drive up and collect data to use later on and we will use [NVIDIA](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/) model to train, which is used by NVIDIA in real world. 

## Motivation
How do we learn to drive? How about computers? We watch other people and observe their skills, same thing for computers, they observe the collected data. What is NVIDIA’s solution? They collected 3 images from 3 different cameras(left, center,right) and recorded the angle of steering wheel for these images. Then send them over to CNN model.

## Technical Aspect
First we setup the simulation to our computer and play with it to record all the data we will need later to train our model, which includes left, center and right cameras, steering angle, throttle, brake and speed. Then by defining a method to get rid of unbalanced data, and prepare the data to load. Then split data to train and validate. We also generate some augmented images using pan, zoom, brightness and flip techniques. More data, better move! Then process images to different styles. Create batchgenerator. Then eventually create cnn model and fit the model, save it.  Then test it! 

## Methods

We created these methods to use in our project:
- Get name of the file path
- Import Data
- Balance the Data
- Load the Data
- Create Image Augmentation
- Preprocess the image 
- Generate batch
- Create Model

Then using **from sklearn.model_selection import train_test_split** , split the data to train then fit the model, and eventually Test it! 

## Technologies Used
![Python](https://www.python.org/static/img/python-logo@2x.png)
![NVIDIA](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSmtUrRp4uhS7G7qFE6ki5Z_Nj_JOMSUTs_yg&usqp=CAU)
![Tensorflow](https://www.gstatic.com/devrel-devsite/prod/v4d5d232859440be8edf63a1095b80ebe5c19605e99f3b348a30c4b0140c2eb88/tensorflow/images/lockup.svg)
![socketio](https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/Socket-io.svg/330px-Socket-io.svg.png)
![Flask](https://flask.palletsprojects.com/en/1.1.x/_images/flask-logo.png)
![Udacity](https://miro.medium.com/proxy/1*Tk8bK5S0TsvfRo4gBFe0Lw.png))
![WSGI](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTcJlaQnfiLOBTwanSbzbRJx5wKxV4ZQYWv-g&usqp=CAU)
![Keras](https://keras.io/img/logo-small.png)