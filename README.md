# Sentiment-analysis-using-visual-and-textual-data
This project consists of 2 modules:
1. Sentiment analysis using faces in a video
2. Sentiment analysis from subtitles

## Sentiment Analysis using faces in a video
First of all, this module captures the frames in the video. Then it iterates through every frame and recognizes face in each frame using pre-trained YOLOv3 weights. Then it extracts only the face region of the image with some offset around it. This offset makes face completely visible. The effect can be observed in the below figure.
<p align="center">
  <img width="500" height="140" src="https://github.com/hafizas101/Sentiment-analysis-using-visual-and-textual-data/blob/master/images/offset.png">
</p>
This face image is converted to grayscale and passed into Emotion classifier that classifies the emotion of the face in either of 7 emotions. These emotions and their labels are described as follows:<br/>
emotion_code_mapping = {0:'angry',1:'disgust',2:'fear',3:'happy', 4:'sad',5:'surprise',6:'neutral'} <br/>
The emotion classifier has been trained on well known [Fer2013 dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) .  The classifer also returns a score indicating the probability of that emotion. Each face in every frame is labelled with its emotion and finally all frames are converted to an mp4 video whose frames per second can be controlled using output_FPS variable. After that we write all the information of each face corresponding to every frame in a .csv file. Finally we find the most frequent emotion and a final score in which the weight of each emotion is bed on repetition of the emotion. <br/>

## Sentiment Analysis using subtitle file

