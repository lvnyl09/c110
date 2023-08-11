import cv2
import numpy as np
import tensorflow as tf

# Load the TensorFlow model
model = tf.keras.models.load_model('keras_model.h5')

# Define a list of class names
class_names = ['rock', 'paper', 'scissor']

# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(1)

# Infinite loop
while True:
    # Reading / Requesting a Frame from the Camera 
    status , frame = camera.read()

    # if we were successfully able to read the frame
    if status:
        # Flip the frame
        frame = cv2.flip(frame , 1)

        # Resize the frame
        resized_frame = cv2.resize(frame, (224, 224))

        # Expand the dimensions
        input_data = np.expand_dims(resized_frame, axis=0)

        # Normalize it before feeding to the model
        input_data = input_data / 255.0

        # Get predictions from the model
        predictions = model.predict(input_data)

        # Get the index of the predicted class
        predicted_index = np.argmax(predictions[0])

        # Get the name of the predicted class
        predicted_class = class_names[predicted_index]

        # Print the predicted class to the console
        print(predicted_class)

        # Displaying the frames captured
        cv2.imshow('feed' , frame)

        # Waiting for 1ms
        code = cv2.waitKey(1)

        # If space key is pressed, break the loop
        if code == 32:
            break

# Release the camera from the application software
camera.release()

# Close the open window
cv2.destroyAllWindows()
