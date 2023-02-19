import numpy as np
import cv2
import tensorflow.keras as keras
import copy
import os
import uuid
from typing import Tuple

# Create a custom layer to calculate the difference between two feature sets
class AbsDistance(keras.layers.Layer):
    """Calculates absolute distance between two tensors."""
    
    def __init__(self, name="absDistance", **kwargs):
        super().__init__(name=name, **kwargs)
    
    def call(self, inputs):
        return keras.backend.abs(inputs[0] - inputs[1])

custom_objects = {'AbsDistance': AbsDistance}

class Helper:
    @staticmethod
    def getFiles(directory: str, extension: str = 'jpg', returnAbsPath: bool = True) -> list:
        """
        This method returns the list of files with the given extension within the given directory.

        Parameters:
            directory (str): The path to the directory to scan.
            extension (str, optional): The extension of the files to count. Default is 'jpg'.
            returnAbsPath (bool, optional): If True, returns a list of absolute paths. Default is True.

        Returns:
            list: A list of file directories.

        Raises:
            ValueError: If no files are found in the given directory.
        """
        
        fileDirs = []
        for entry in os.scandir(directory):
            if not entry.is_file() or not entry.name.endswith(f'.{extension}'):
                continue

            fileName = entry.name
            if returnAbsPath:
                fileDirs.append(os.path.join(directory, fileName))

            else:
                fileDirs.append(fileName)
        
        if len(fileDirs) == 0:
            raise ValueError(f"No files found in directory {directory}")
            
        return fileDirs
    
    
    @staticmethod
    def loadImage(directory: str, toGrayscale: bool = False, newSize: tuple = None) -> Tuple[np.ndarray, tuple]:
        """
        This method loads an image from the given directory.

        Parameters:
            directory (str): The path to the image to load.
            toGrayscale (bool, optional): If True, converts the image to grayscale. Default is False.
            newSize (tuple, optional): If not None, the image will be resized to the given size.

        Returns:
            tuple: A tuple of the loaded image and image shape.

        Raises:
            FileNotFoundError: If the image is not found in the given directory.
        """
        
        img = cv2.imread(directory)
        if type(img) == type(None):
            raise FileNotFoundError(f"Image not found in {directory}.")

        imgShape = img.shape

        if toGrayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if newSize != None:
            img = cv2.resize(img, newSize, interpolation=cv2.INTER_AREA)

        return np.array(img), imgShape


class AI:
    def __init__(self, app_dir, input_shape, threshold):
        # Load the model from disk
        try:
            self.model = keras.models.load_model(os.path.join(app_dir, "../models/FINAL--ConvUnits[32, 64, 128, 256]--Kernel(5, 5)--DenseUnits[]--BatchNormState=1--BatchSize32.h5"), custom_objects=custom_objects)
        
        except:
            raise OSError("Can not open the model.")

        self.app_dir = app_dir
        self.input_shape = input_shape
        self.threshold = threshold

    def preprocess_image(self, img):
        # Convert the image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize the image to the required input shape
        img = cv2.resize(img, self.input_shape[:2])

        # Convert to numpy and reshape the array
        img = np.array(img, dtype='float32')
        img = img.reshape((1, *self.input_shape))

        # Normalize the pixel values to be between 0 and 1
        img = img / 255.0
        return img

    def predict(self, img1, img2):
        img1, img2 = self.preprocess_image(img1), self.preprocess_image(img2)
        prediction = float(self.model.predict([img1, img2]))
        print(prediction)
        # Return 1 if the prediction is above the threshold, 0 otherwise
        return 1 if prediction >= self.threshold else 0

    def match_image(self, img):
        # Get the paths of all face images in the faces directory
        faces_paths = Helper.getFiles(os.path.join(self.app_dir, "faces"), returnAbsPath=True)

        # Iterate over each face image and compare it to the input image
        for face_path in faces_paths:
            print(os.path.split(face_path)[-1])
            face, _ = Helper.loadImage(face_path)

            # If the prediction is above the threshold, return 1 (i.e., the images match)
            if self.predict(img, face):
                return 1

        # If no match is found, return 0
        return 0


class UI:
    def __init__(self, app_dir, AI_engine, webcam):
        # Initialize the face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.cap = webcam

        # Set the paths to the faces directory and the app directory
        self.app_dir = app_dir
        self.faces_path = os.path.join(self.app_dir, "faces")

        # Store a reference to the AI engine
        self.AI_engine = AI_engine

        # Create the faces directory if it does not exist
        if not os.path.exists(self.faces_path):
            os.makedirs(self.faces_path)

    def sign_up(self):
        print("Press 'q' to take a picture.")

        while True:
            # Read a frame from the video capture
            ret, frame = self.cap.read()
            frameCopy = copy.deepcopy(frame)

            # Convert the frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale frame
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            # Draw rectangles around the detected faces
            for (x, y, w, h) in faces:
                face = frame[y:y + h, x:x + w]
                cv2.rectangle(frameCopy, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the frame in a window called "Video"
            cv2.imshow("Video", frameCopy)

            # Wait for a key press and exit if 'q' is pressed
            if (cv2.waitKey(1) & 0xFF == ord('q')) and len(faces) == 1:
                save_path = os.path.join(self.faces_path, str(f"{str(uuid.uuid4())}.jpg"))
                cv2.imwrite(save_path, face)
                print("Your face has been added to the database.")
                break
            
        cv2.destroyAllWindows()
                 
                    
    def log_in(self):
        print("Press 'q' to take a picture.")

        while True:
            # Read a frame from the video capture
            ret, frame = self.cap.read()
            frameCopy = copy.deepcopy(frame)

            # Convert the frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale frame
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            # Draw rectangles around the detected faces
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                cv2.rectangle(frameCopy, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the frame in a window called "Video"
            cv2.imshow("Video", frameCopy)

            # Wait for a key press and exit if 'q' is pressed and exactly one face is detected
            if cv2.waitKey(1) & 0xFF == ord('q') and len(faces) == 1:
                # Match the detected face with the stored faces
                result = self.AI_engine.match_image(face)
                if result:
                    print("You are now logged in.")
                else:
                    print("Face does not match.")
                break # exit the while loop

        cv2.destroyAllWindows()


def main():
    webcam = cv2.VideoCapture(0)

    # Specify directory for application
    app_dir = r"D:/CS/Projects/Single_Shot_Learning/Ui"

    # Define input shape for AI model
    input_shape = (128, 128, 1)

    # Define threshold for AI model
    threshold = 0.6
    
    # Initialize AI engine
    AI_engine = AI(app_dir, input_shape, threshold)

    # Initialize user interface
    ui = UI(app_dir, AI_engine, webcam)
    
    # Continuously prompt user for command until program is exited
    while True:
        print("1) Sign Up")
        print("2) Log in")
        cmd = input("Enter a command> ")

        # If user enters 1, prompt user to sign up
        if cmd == '1':
            ui.sign_up()
        
        # If user enters 2, prompt user to log in
        elif cmd == '2':
            ui.log_in()
        
        # If user enters an invalid command, display error message
        else:
            print("Invalid command")


            
if __name__ == '__main__':
    main()