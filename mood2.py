#!/usr/bin/env python
# coding: utf-8

# # Import libraries and define constants

# In[83]:


import os, sys, cv2, time, json, csv, re, pysrt
from yolo_files.utils import *
from keras.models import load_model
from utils.datasets import get_labels
from utils.preprocessor import preprocess_input
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np

"""
num is a very important variable. It controls how much frames per second of the video we are extracting so
basically it is very computationally expensive and time consuming for long videos to perform frame by frame
detection and processing. Hence we can periodically skip some frames. This is controlled by num variable. Setting
num=1 would mean extract all frames. num = 2 means skip 1 frame and then extract the second and so on. num = 4
would mean skip 3 frames and then extract 1 and then again skip 3 and so on.
"""

num = 1
offset = 30
emotion_code_mapping = {0:'angry',1:'disgust',2:'fear',3:'happy', 4:'sad',5:'surprise',6:'neutral'}
input_video_path = os.path.join(os.getcwd(), "input_video.mp4")
output_video_path = os.path.join(os.getcwd(), "output.mp4")
current_path = os.getcwd()
output_FPS = 4

# stopword = stopwords.words('english')
# wordnet_lemmatizer = WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()
"""
Boundary for slighly and highly negative. Increasing it will increase number of slightly negatives
but will decrease number of highly negatives. Same rule for positives.
"""
sentiment_threshold = 0.3
num_period = 100
current_path = os.getcwd()
input_subtitle_path = os.path.join(os.getcwd(), "climate.srt")
emotions = ['Highly Negative', 'Slightly Negative', 'Neutral', 'Slightly Positive', 'Highly Positive']


# # Function definitions

# In[70]:


def draw_predict(frame, conf, emotion, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), COLOR_YELLOW, 2)

    text = emotion+' {:.2f}'.format(conf)

    # Display the label at the top of the bounding box
    label_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    top = max(top, label_size[1])
    cv2.putText(frame, text, (left, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
    return

def convert_frames_to_video(frames):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    height, width, channels = frames[0].shape
    out = cv2.VideoWriter(output_video_path, fourcc, output_FPS, (width, height))
    for i, ff in enumerate (frames):
        out.write(ff)
    out.release
    cv2.destroyAllWindows()
    return

def clean_words(text):
    words = nltk.word_tokenize(text)
#     removing_stopwords = [word for word in word_tokens if word not in stopword]
#     lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in removing_stopwords]
    wordsa=[word.lower() for word in words if word.isalpha()]
    return wordsa


def calculate_polarity_and_intensity(d, rep):
    if d['neu']==1.0:
        emotional_intensity = "Neutral"
        polarity = 0.0
        rep[2] = rep[2] + 1
    elif ((d['neg'] < sentiment_threshold) & (d['pos'] == 0.0)):
        emotional_intensity = "Slightly Negative"
        polarity = -1*d['neg']
        rep[1] = rep[1] + 1
    elif ((d['neg'] > sentiment_threshold) & (d['pos'] == 0.0)):
        emotional_intensity = "Highly Negative"
        polarity = -1*d['neg']
        rep[0] = rep[0] + 1
    elif ((d['neg'] == 0.0) & (d['pos'] < sentiment_threshold)):
        emotional_intensity = "Slightly Positive"
        polarity = d['pos']
        rep[3] = rep[3] + 1
    elif ((d['neg'] == 0.0) & (d['pos'] > sentiment_threshold)):
        emotional_intensity = "Highly Positive"
        polarity = d['pos']
        rep[4] = rep[4] + 1
    else:
        values = [d['neg'], d['neu'], d['pos']]
        dominant = max(values)
        
        if values.index(dominant) == 0:
            polarity = -1*d['neg']
            if dominant > sentiment_threshold:
                emotional_intensity = "Highly Negative"
                rep[0] = rep[0] + 1
            else:
                emotional_intensity = "Slightly Negative"
                rep[1] = rep[1] + 1
        elif values.index(dominant) == 1:
            polarity = 0.0
            emotional_intensity = "Neutral"
            rep[0] = rep[0] + 1
        else:
            polarity = d['pos']
            if dominant < sentiment_threshold:
                emotional_intensity = "Slightly Positive"
                rep[3] = rep[3] + 1
            else:
                emotional_intensity = "Highly Positive"
                rep[4] = rep[4] + 1
    return polarity, emotional_intensity, rep





def is_time_stamp(l):
    if l[:2].isnumeric() and l[2] == ':':
        return True
    return False

def has_letters(line):
    if re.search('[a-zA-Z]', line):
        return True
    return False

def has_no_text(line):
    l = line.strip()
    if not len(l):
        return True
    if l.isnumeric():
        return True
    if is_time_stamp(l):
        return True
    if l[0] == '(' and l[-1] == ')':
        return True
    if not has_letters(line):
        return True
    return False

def is_lowercase_letter_or_comma(letter):
    if letter.isalpha() and letter.lower() == letter:
        return True
    if letter == ',':
        return True
    return False

def clean_up(lines):
    """
    Get rid of all non-text lines and
    try to combine text broken into multiple lines
    """
    new_lines = []
    for line in lines[1:]:
        if has_no_text(line):
            continue
        elif len(new_lines) and is_lowercase_letter_or_comma(line[0]):
            new_lines[-1] = new_lines[-1].strip() + ' ' + line
        else:
            new_lines.append(line)
    return new_lines


# # mod-et-emo-001 ==> Video Splitting in Frames

# In[71]:



cap = cv2.VideoCapture(input_video_path)
FPS = cap.get(cv2.CAP_PROP_FPS)
print("Frames Per Second: "+str(FPS))

frames = []
start = time.time()
count = 0
frame_originals = []
while (cap.isOpened()):
    ret, frame = cap.read()
    count = count + 1
    if not ret:
        break
    if count % num == 0:        
        frames.append(frame)
#         cv2.imshow("Original", frame)
#         cv2.waitKey(0)
end = time.time()
print("Total number of frames read: "+str(len(frames)))
print("Time taken in reading the frames: {} seconds".format(end-start))
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


# # mod-et-emo-002 ==> Faces and mood detection per frame

# In[72]:


cfg = './yolo_files/yolov3-face.cfg'
weights = './yolo_files/yolov3-wider_16000.weights'

# Give the configuration and weight files for the model and load the network using them.

net = cv2.dnn.readNetFromDarknet(cfg, weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
faces = []


emotion_model_path = './models/emotion_model.hdf5'
# emotion_labels = get_labels('fer2013')
# print(emotion_labels)
emotion_classifier = load_model(emotion_model_path)
emotion_target_size = emotion_classifier.input_shape[1:3]

count_repetition = np.array([0, 0, 0, 0, 0, 0, 0])

X_position = []
Y_position = []
Frame = []
Width = []
Height = []
Emotion_code = []
Emotion_name = []
Score = []
start = time.time()
for i in range (len(frames)):
    blob = cv2.dnn.blobFromImage(frames[i], 1 / 255, (IMG_WIDTH, IMG_HEIGHT),[0, 0, 0], 1, crop=False)
    net.setInput(blob)
    outs = net.forward(get_outputs_names(net))

    # Remove the bounding boxes with low confidence
    frame_faces = post_process(frames[i], outs, CONF_THRESHOLD, NMS_THRESHOLD)
    if len(frame_faces) == 0:
        continue
    else:
        face_imgs = []
        for j, f in enumerate(frame_faces):
            Frame.append(i+1)
            faces.append(f)
            [left, top, w, h] = f
            X_position.append(left)
            Y_position.append(top)
            Width.append(w)
            Height.append(h)
            original_face = frames[i][top : top+h, left:left+w]
            cv2.imwrite(str(j+1)+".jpg", original_face)
            offset_face = frames[i][top-offset : top+h+offset , left-offset:left+w+offset]
            cv2.imwrite(str(j+1)+"_offset.jpg", offset_face)
            face_imgs.append(frames[i][top-offset : top+h+offset , left-offset:left+w+offset])
#             cv2.imshow("Faces", face_imgs[j])
#             cv2.waitKey(0)
            gray_face = cv2.cvtColor(face_imgs[j], cv2.COLOR_BGR2GRAY)
            gray_face = cv2.resize(gray_face, (emotion_target_size))
            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            Score.append(emotion_probability)
            emotion_label_arg = np.argmax(emotion_prediction)
            Emotion_code.append(emotion_label_arg)
            count_repetition[emotion_label_arg] = count_repetition[emotion_label_arg] + 1
            emotion_text = emotion_code_mapping[emotion_label_arg]
            Emotion_name.append(emotion_text)
            draw_predict(frames[i], emotion_probability, emotion_text, left, top, left+w, top+h)

#         cv2.imshow("Faces", frames[i])
#         cv2.waitKey(0)
end = time.time()
print("Time taken in processing {0:1.0f} frames: {1:2.3f} seconds".format(len(frames), end-start))
convert_frames_to_video(frames)
cv2.destroyAllWindows()
project2_dict = {'Frame': Frame, 'X_position': X_position, 'Y_position': Y_position, 'Width': Width, 'Height': Height,
                 'Emotion_code': Emotion_code, 'Emotion_name': Emotion_name, 'Score': Score}

if os.path.exists(os.path.join(current_path, "mod-et-emo-002.csv")):
    os.remove(os.path.join(current_path, "mod-et-emo-002.csv"))
df = pd.DataFrame(data=project2_dict)
df.to_csv('mod-et-emo-002.csv', index=False, encoding='utf-8')


# # mod-et-emo-003 ==> Aggregation of emotions detected

# In[26]:


# print(count_repetition)

aggregated_emotion_code = np.argmax(count_repetition)
print("Aggregated Emotion Code: "+str(aggregated_emotion_code))
aggregated_emotion_name = emotion_code_mapping[aggregated_emotion_code]
print("Most Frequent Emotion: "+str(aggregated_emotion_name))
num = len(Emotion_code)
total_rep = 0
total = 0
for i in range (num):
    total = total + Score[i]*count_repetition[Emotion_code[i]]
    total_rep = total_rep + count_repetition[Emotion_code[i]]
    
final_code = total/total_rep
print("Final result: "+str(final_code))


# # mod-et-emo-004 ==> Mood polarity detection in subtitles

# In[84]:


with open(input_subtitle_path, errors='replace') as f:
    lines = f.readlines()
    new_lines = clean_up(lines)

emotional_intensity = []
polarity = []
sentences = []
count_repitition = [0, 0, 0, 0, 0]
count = 0
for i, line in enumerate(new_lines):
    words = clean_words(line)
    data = ""
    for j, w in enumerate(words):
        data = data + " "+ w    
    
    if data == "":
        continue
    sents = nltk.sent_tokenize(data)
    for s in sents:
        sentences.append(s)
        d = analyzer.polarity_scores(s)
        p, e, count_repitition = calculate_polarity_and_intensity(d, count_repitition)
        polarity.append(p)
        emotional_intensity.append(e)

if os.path.exists(os.path.join(current_path, "mod-et-emo-004.csv")):
    os.remove(os.path.join(current_path, "mod-et-emo-004.csv"))
if len(sentences) == len(emotional_intensity):
    df = pd.DataFrame(data = {'Sentences': sentences, 'Emotional Intensity': emotional_intensity, 'Polarity': polarity})
    df.to_csv('mod-et-emo-004.csv', sep='\t', index=False)


# ## mod-et-emo-005 ==> Association of the temporary emotional intensity in a video

# In[86]:


subs = pysrt.open(input_subtitle_path)
x = subs[len(subs)-1]
[hour, minute, sec] = [x.end.hours, x.end.minutes, x.end.seconds]
total_duration = hour*3600000 + minute*60000 + sec*1000

text = ""
for i in range (len(subs)):
    a = subs[i]
    text = text +" "+ a.text
    
words = clean_words(text)
num_words = len(words)
print("Total words: "+str(num_words))
words_per_duration = num_words/total_duration
time_period = int(total_duration/num_period)
print("Time period in milliseconds: "+str(time_period))
words_period = int(time_period*words_per_duration)
print("Words per time period: "+str(words_period+1))

period_num = []
from_to = []
polarity = []
emotional_intensity = []
strings = []
starting = 0
ending = time_period

start = 0
stop = words_period+1
count_repitition = [0, 0, 0, 0, 0]
for i in range(num_period):
    all_words = words[start:stop]
    data = ""
    for j, w in enumerate(all_words):
        data = data + " "+ w    
    
    if data == "":
        continue
    period_num.append(i)
    from_to.append((starting, ending))
    strings.append(data)
    d = analyzer.polarity_scores(data)
    p, e, count_repitition = calculate_polarity_and_intensity(d, count_repitition)
    polarity.append(p)
    emotional_intensity.append(e)    
    
    start = stop
    stop = start + words_period+1
    starting = ending
    ending = starting + time_period
    
if os.path.exists(os.path.join(current_path, "mod-et-emo-005.csv")):
    os.remove(os.path.join(current_path, "mod-et-emo-005.csv"))
df = pd.DataFrame(data = {'Sentences': strings, 'Time Period':period_num, 'Time from, time to (in miliseconds)':from_to, 'Detected polarity': polarity, 'Emotional_Intensity': emotional_intensity})
df.to_csv('mod-et-emo-005.csv', sep='\t', index=False)


# # mod-et-emo-006 ==> Polarity aggregation in subtitles

# In[87]:


"""
Ranking is being performed based upon how many times an emotion appears. Most frequent will get rank 1 and least
frequent will get rank 5.
"""

ranks = [0,0,0,0,0]
rep = count_repitition.copy()
total = sum(rep)
scores = []
for i in range (len(rep)):
    scores.append(rep[i]/total)
rank = 1
for i in range (len(count_repitition)):
    value = max(count_repitition)
    ranks [rep.index(value)] = ranks [rep.index(value)] + rank
    count_repitition.remove(value)
    rank = rank + 1
    
print("Ranks: "+str(ranks))
print("Final Result: "+str(scores))

if os.path.exists(os.path.join(current_path, "mod-et-emo-006.csv")):
    os.remove(os.path.join(current_path, "mod-et-emo-006.csv"))
df = pd.DataFrame(data = {'Emotions': emotions, 'Ranking':ranks, 'Final Result':scores})
df.to_csv('mod-et-emo-006.csv', sep='\t', index=False)


# In[ ]:





# In[ ]:




