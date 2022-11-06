import calendar
import time
import streamlit as st
import pafy
import pickle
import keras
import os
import cv2
import math
import pafy
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from moviepy.editor import *
from collections import deque
import matplotlib.pyplot as plt
#matplotlib inline
from keras.models import load_model

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

image_height, image_width = 64, 64
max_images_per_class = 8000

dataset_directory = "UCF50"
classes_list = ["TennisSwing", "Basketball", "JavelinThrow", "HorseRiding"]

model_output_size = len(classes_list)

st.title("Human Activity Predictor")
tb = st.text_input(label='Enter Youtube Video Link')

#youtube_video_url = tb
video_title_temp ="abcfdfdf123"
# def append_timestamp(filename):

#      timestamp = calendar.timegm(time.gmtime())
#      human_readable = dt.datetime.fromtimestamp(timestamp).isoformat()
#      filename_with_timestamp = filename + "_" + str(human_readable)
#      return filename_with_timestamp


output_directory = "/app/human_activity_predictor"
st.text("OPFP " + output_directory)

  





def download_youtube_videos(youtube_video_url, output_directory):
    try:
        
        st.text("Downloading Video")
        # Creating a Video object which includes useful information regarding the youtube video.
        #video_title_temp = append_timestamp(video_title_temp)
        video = pafy.new(youtube_video_url)

        # Getting the best available quality object for the youtube video.
        video_best = video.getbest()

        # Constructing the Output File Path
        output_file_path = f'{output_directory}/{video_title_temp}.mp4'

        st.text("OPFP " + output_file_path)
        print("VTT : "  + video_title_temp)

        # Downloading the youtube video at the best available quality.
        video_best.download(filepath = output_file_path, quiet = True)

        # Returning Video Title
        st.text("Done Downloading")
    except Exception as err:
     print(f"Unexpected {err=}, {type(err)=}") 

    return video_title_temp

model = tf.keras.models.load_model("mymodel.h5")
def make_average_predictions(video_file_path, predictions_frames_count):
    try:
        op = []
        # Initializing the Numpy array which will store Prediction Probabilities
        predicted_labels_probabilities_np = np.zeros((predictions_frames_count, model_output_size), dtype = np.float)

        # Reading the Video File using the VideoCapture Object
        video_reader = cv2.VideoCapture(video_file_path)

        # Getting The Total Frames present in the video 
        video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculating The Number of Frames to skip Before reading a frame
        skip_frames_window = video_frames_count // predictions_frames_count

        for frame_counter in range(predictions_frames_count): 

            # Setting Frame Position
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

            # Reading The Frame
        _ , frame = video_reader.read() 
        #print('frame : ' + str(frame))
        # print('h : ' + str(image_height))
        # print('w : ' + str(image_width))
            # Resize the Frame to fixed Dimensions
        resized_frame = cv2.resize(frame, (image_height, image_width))
        
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255

        # Passing the Image Normalized Frame to the model and receiving Predicted Probabilities.
        predicted_labels_probabilities = model.predict(np.expand_dims(normalized_frame, axis = 0))[0]

        # Appending predicted label probabilities to the deque object
        predicted_labels_probabilities_np[frame_counter] = predicted_labels_probabilities

        # Calculating Average of Predicted Labels Probabilities Column Wise 
        predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis = 0)

        # Sorting the Averaged Predicted Labels Probabilities
        predicted_labels_probabilities_averaged_sorted_indexes = np.argsort(predicted_labels_probabilities_averaged)[::-1]

        # Iterating Over All Averaged Predicted Label Probabilities
        for predicted_label in predicted_labels_probabilities_averaged_sorted_indexes:

            # Accessing The Class Name using predicted label.
            predicted_class_name = classes_list[predicted_label]

            # Accessing The Averaged Probability using predicted label.
            predicted_probability = predicted_labels_probabilities_averaged[predicted_label]

            print(f"CLASS NAME: {predicted_class_name}   AVERAGED PROBABILITY: {(predicted_probability):.2}")
            op.append(f"{predicted_class_name} :-> {(predicted_probability):.2}")
    
        # Closing the VideoCapture Object and releasing all resources held by it. 
        video_reader.release()
        return op
    except Exception as err:
     print(f"Unexpected {err=}, {type(err)=}")   
try:
    if st.button('Submit'):
 
      #video_title_temp = append_timestamp(video_title_temp)
      video_title = download_youtube_videos(tb, output_directory)
      print('video : ' + video_title)
      input_video_file_path = f'{output_directory}/{video_title_temp}.mp4'
      print(tb)
      print(input_video_file_path)
      resp = make_average_predictions(input_video_file_path, 100)
      for x in resp:
        st.text(x)
     #VideoFileClip(input_video_file_path).ipython_display(width = 700)
      video_file = open(video_title_temp+".mp4",'rb')
      video_bytes = video_file.read()
      st.video(video_bytes)
     #model.summary()
      st.balloons()
except Exception as err:
     print(f"Unexpected {err=}, {type(err)=}")  
       

