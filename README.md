# MTCNN FOR FACE IDENTIFICATION 
![2023-05-23-14-16-03](https://github.com/HedvaldoCosta/FaceDetection/assets/67663958/1b4bef45-bf42-498d-8c77-b39d6fab34bc)

## About this project
The application was built with the user's freedom to perform tests using their own photos and videos in mind. It can also be used for facial recognition of specific people on your own webcam and street monitoring using security cameras.

## Application
Ainda necessário gerar o link

## Functionalities
* Choose your own video, the user can upload a video directly (mp4 or avi) from their computer where it will be rendered and the program will show the video with face identification. If the user does not have a video, they can choose one of the 4 videos attached within the app.
* Choose your own image, the user can choose an image file (jpg, jpeg or png) from their own computer. When selecting the image, the model will search for a face, mark it and return it to the user.

  ![image](https://github.com/HedvaldoCosta/FaceDetection/assets/67663958/f10f1bda-93ee-431c-9700-d09d701bdca9)
  
    * In certain cases, artificial intelligence (AI) cannot identify the face, and it may be that something is covering a part of the face or even the person is on the side.

      ![image](https://github.com/HedvaldoCosta/FaceDetection/assets/67663958/ed7ca574-afb2-4701-9d0a-bde9010bfb8a)

* Using the webcam, the user can connect to their webcam, the model will identify their face and return with the user's face marked. The model still reads pixel by pixel, so the return tends to be slow.

## Tools
pycharm community Edition 2023.1

python 3.9.13

click==8.1.3

mtcnn==0.1.1

numpy==1.23.5

opencv-python==4.7.0.72

streamlit==1.22.0

google colab

## About the code
````python
# Library used to build the application and put the project into production
import streamlit as st
# Creation of the rectangle and loading of images/videos/webcam
import cv2
# face detection
from mtcnn import MTCNN
# Convert a sequence of bytes to a NumPy array.
import numpy as np
# Create a temporary file on the file system with a unique name.
import tempfile
````

````python
# Class created for using the face detection function in videos and images
# loaded by opencv
class FaceDetection:
    def __init__(self, file_image='', file_video=''):
        # Instance of a face detector object 
        self.detector = MTCNN()
        # Instance of receiving the image
        self.image = file_image
        # Instance of receiving the video
        self.video = file_video
````

````python
    # Function used to apply face detection in images
    def detecting_faces_image(self):
        # perform face detection on the image
        result = self.detector.detect_faces(self.image)
        # loop over each element (dictionary) present in the 'result' list
        for faces in result:
            # The values of the 'box' keys are extracted and assigned to the variables x, y,
            # width and height. This information represents the coordinates and dimensions 
            # of the bounding box (rectangle) around the detected face.
            x, y, width, height = faces['box']
            # Draw a rectangle on the original image
            cv2.rectangle(self.image, (x, y), (x + width, y + height), (0, 0, 255), 2)
        # Returns the image with a rectangle on the face
        return self.image
````

````python
    # Function used to apply face detection in videos
    def detecting_faces_video(self):
        # Loop to be maintained while the video is open
        while self.video.isOpened():
            # Checking if each frame of self.video was read and stored
            # ret is a boolean value that indicates whether the frame was read
            ret, frame = self.video.read()
            # Indicates that the next frame of the video could not be read.
            if not ret:
                break
            # Detect faces in picture frame.
            result = self.detector.detect_faces(frame)
            # loop over each element (dictionary) present in the 'result' list
            for face in result:
                # The values of the 'box' keys are extracted and assigned to the variables
                # x, y, width and height. This information represents the coordinates
                # and dimensions of the bounding box (rectangle) around the
                # face detected.
                x, y, w, h = face['box']
                # Draw a rectangle on the original image
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # Convert image frame from BGR format to RGB format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Generator function, which can be iterated to get each frame
            # Processed individually.
            yield frame
````

````python
    # Function to detect faces on your webcam
    def detecting_faces_webcam(self):
        # Start your webcam
        cap = cv2.VideoCapture(0)
        # Loop that stops when it is not possible to read the next frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Face detection using MTCNN
            results = self.detector.detect_faces(frame)

            # Draw rectangles around detected faces
            for result in results:
                x, y, w, h = result['box']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Show the frame with the rectangles of the detected faces
            cv2.imshow('Webcam', frame)

            # Loop exit condition (by pressing "q" key)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # Release the resources
            cap.release()
            cv2.destroyAllWindows()
````

````python
if __name__ == '__main__':
    # Creating a title for videos in the sidebar
    st.sidebar.title("ESCOLHA UM VÍDEO")
    # Function that gives user permission to upload their own videos
    uploaded_file = st.sidebar.file_uploader("", type=["mp4", "avi"])
    if uploaded_file is not None:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        # Write the string as Bytes in the temp file
        temp_file.write(uploaded_file.read())
        # Get the name of the temporary file and close it
        video_path = temp_file.name
        temp_file.close()
        # Capture video streams from video files
        video = cv2.VideoCapture(video_path)
        # Return the FaceDetection class receiving the "video" parameter
        face_detection_video = FaceDetection(file_video=video)
        # Start the function for detection in videos
        video_generator = face_detection_video.detecting_faces_video()
        # Create and update interactive display elements in apps
        stframe = st.empty()
        while True:
            try:
                # Get the next frame of the video
                frame = next(video_generator)
                # Display the frame in the interface
                stframe.image(frame, channels="RGB")
            # Indicate the end of an iteration
            except StopIteration:
                break
    # Possibility to test videos already loaded inside the application
    st.sidebar.info("Sem arquivos? escolha aqui")
    video_choice = {'PARAR': '',
                    'Vídeo 1': 'https://bit.ly/436YGiF',
                    'Vídeo 2': 'https://bit.ly/45bh5wJ',
                    'Vídeo 3': 'https://bit.ly/3Oneg5H',
                    'Vídeo 4': 'https://bit.ly/42POU4R'}
    select_video = st.sidebar.selectbox('', video_choice.keys())
    video = cv2.VideoCapture(video_choice[select_video])
    face_detection_video = FaceDetection(file_video=video)
    video_generator = face_detection_video.detecting_faces_video()
    stframe = st.empty()
    while True:
        try:
            frame = next(video_generator)
            stframe.image(frame, channels="RGB")
        except StopIteration:
            break
````

````python
    # Creating a title for images in the sidebar
    st.sidebar.title("ESCOLHA UMA IMAGEM")
    # Function to load images on your own device
    upload_image = st.sidebar.file_uploader("", type=["jpg", "jpeg", "png"])
    if upload_image is not None:
        # Decode and read the image represented by the NumPy array
        image = cv2.imdecode(np.fromstring(upload_image.read(), np.uint8), 1)
        # Call the FaceDetection class with the "image" parameter
        face_detection_image = FaceDetection(file_image=image)
        # Perform face detection function on images
        image_generator = face_detection_image.detecting_faces_image()
        # Returns the image in the application
        st.image(image_generator, channels="BGR")
````

````python
    # Creating a title for the webcam part
    st.sidebar.title("WEBCAM")
    # Button created to start the webcam
    start_webcam_button = st.sidebar.button("INICIAR WEBCAM")
    # If the button is pressed, it will call the FaceDetection class executing the function
    # for webcam face detection
    if start_webcam_button:
        face_detection_webcam = FaceDetection()
        face_detection_webcam.detecting_faces_webcam()
````
