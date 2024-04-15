import streamlit as st
from pytube import YouTube
import os
import tempfile
import cv2

# Load your trained model

from tensorflow import keras
model = tf.keras.models.load_model('Transfer_Learning_Model_MobileNet___Date_Time_2024_04_14__10_23_59___Loss_1.0241948366165161___Accuracy_0.4468750059604645.h5')




def download_youtube_videos(youtube_video_url, output_directory):
    # Creating a Video object which includes useful information regarding the youtube video.
    video = pafy.new(youtube_video_url)

    # Getting the best available quality object for the youtube video.
    video_best = video.getbest()

    # Constructing the Output File Path
    output_file_path = f'{output_directory}/{video.title}.mp4'

    # Downloading the youtube video at the best available quality.
    video_best.download(filepath = output_file_path, quiet = True)

    # Returning Video Title
    return video.title




from collections import deque, defaultdict
import cv2
import numpy as np

def predict_on_live_video(video_file_path, output_file_path, window_size, model, classes_list):
    # Initialize a Deque Object with a fixed size which will be used to implement moving/rolling average functionality.
    predicted_labels_probabilities_deque = deque(maxlen=window_size)

    # Reading the Video File using the VideoCapture Object
    video_reader = cv2.VideoCapture(video_file_path)
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = video_reader.get(cv2.CAP_PROP_FPS)
    frame_counts = defaultdict(int)

    # Writing the Overlayed Video Files Using the VideoWriter Object
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 24, (original_video_width, original_video_height))

    while True:
        # Reading The Frame
        status, frame = video_reader.read()
        if not status:
            break

        # Resize the Frame to fixed Dimensions
        resized_frame = cv2.resize(frame, (64, 64))  # Consider adjusting dimensions as required

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255

        # Passing the normalized frame to the model and receiving Predicted Probabilities.
        predicted_labels_probabilities = model.predict(np.expand_dims(normalized_frame, axis=0))[0]

        # Appending predicted label probabilities to the deque object
        predicted_labels_probabilities_deque.append(predicted_labels_probabilities)

        # Assuring that the Deque is completely filled before starting the averaging process
        if len(predicted_labels_probabilities_deque) == window_size:
            # Calculating Average of Predicted Labels Probabilities Column Wise
            predicted_labels_probabilities_averaged = np.array(predicted_labels_probabilities_deque).mean(axis=0)

            # Converting the predicted probabilities into labels by returning the index of the maximum value.
            predicted_label = np.argmax(predicted_labels_probabilities_averaged)
            predicted_class_name = classes_list[predicted_label]

            # Tracking frame counts for each label
            frame_counts[predicted_class_name] += 1

            # Overlaying Class Name Text Ontop of the Frame
            cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Writing The Frame
        video_writer.write(frame)

    # Closing the VideoCapture and VideoWriter objects and releasing all resources held by them.
    video_reader.release()
    video_writer.release()

    # Calculating duration for each activity
    total_frames = sum(frame_counts.values())
    total_duration = total_frames / frame_rate
    durations = {key: (count / frame_rate) for key, count in frame_counts.items()}
    duration_percentages = {key: (durations[key] / total_duration) * 100 for key in durations}

    # Displaying duration and percentage
    for activity, duration in durations.items():
        print(f"Duration of {activity} Activity: {duration:.2f} seconds ({duration_percentages[activity]:.2f}%)")

    return durations, duration_percentages






!pip install pytube
import os
from pytube import YouTube

def download_youtube_videos(url, output_directory):
    # Create YouTube object with the URL
    yt = YouTube(url)

    # Select the highest resolution progressive stream available
    video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()

    # Check if video stream is available
    if not video:
        return "Video not found or incompatible format"

    # Download the video to the output directory
    video.download(output_directory)

    # Return the title of the video for further processing
    return yt.title

# Creating the output directories if it does not exist
output_directory = 'Youtube_Videos'
os.makedirs(output_directory, exist_ok=True)

# URL of the YouTube video you want to download
youtube_url = 'https://www.youtube.com/embed/9znvgHAXQlY?rel=0&autoplay=1&start=1&end=11'

# Downloading the YouTube Video
video_title = download_youtube_videos(youtube_url, output_directory)

# Handling cases where video couldn't be downloaded
if video_title != "Video not found or incompatible format":
    # Getting the YouTube Video's path you just downloaded
    input_video_file_path = f"{output_directory}/{video_title}.mp4"
    print(f"Video downloaded and saved as: {input_video_file_path}")
else:
    print(video_title)








image_height, image_width = 64, 64
max_images_per_class = 8000
# Setting sthe Widow Size which will be used by the Rolling Averge Proces
window_size = 1

# Construting The Output YouTube Video Path
output_video_file_path = f'{output_directory}/{video_title} -Output-WSize {window_size}.mp4'


window_size = 5  # Example window size, adjust based on your needs


# Define your classes list if not already defined
classes_list = ["Background_Activity", "Cutting_Levelling_Activity", "Laying_Bricks_Or_Applying_Mortar"]

# Now call the function with the additional parameters
durations, duration_percentages = predict_on_live_video(input_video_file_path, output_video_file_path, window_size, model, classes_list)

# Optionally, print the results if needed
print("Activity Durations and Percentages:")
for activity in classes_list:
    duration = durations.get(activity, 0)
    percentage = duration_percentages.get(activity, 0)
    print(f"{activity}: Duration = {duration:.2f} seconds, Percentage of total = {percentage:.2f}%")


# Calling the predict_on_live_video method to start the Prediction.
durations, duration_percentages = predict_on_live_video(input_video_file_path, output_video_file_path, window_size, model, classes_list)

# Play Video File in the Notebook
VideoFileClip(output_video_file_path).ipython_display(width = 700)




st.title('Masonry Work Activity Detector')
video_url = st.text_input('Enter YouTube video URL')

if st.button('Process Video'):
    video_path = download_youtube_video(video_url)
    if video_path:
        # Display the video
        st.video(video_path)
        
        # Classify and display results
        result = classify_video(video_path)
        st.write(result)
    else:
        st.error("Failed to download video.")
