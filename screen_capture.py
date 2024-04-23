import time
from mss import mss
import numpy as np
import cv2
import sys  # For debug info

# Load the template image (avatar) and get its dimensions
avatar_template_unsized = cv2.imread('assets/player.png', 0)  # 0 for grayscale
avatar_template = cv2.resize(avatar_template_unsized, (59, 58))  # Resize the template
template_height, template_width = avatar_template.shape[:2]  # Get the dimensions of the resized template

def capture_screen(saveImg=False, *, fileName=None):
    """
    Capture the screen using the mss library.

    Args:
        saveImg (bool, optional): Whether to save the captured image or not. Defaults to False.
        fileName (str, optional): The filename to use when saving the image. If not provided, a timestamp will be used.

    Returns:
        numpy.ndarray: The captured screen as a NumPy array.
    """
    with mss() as sct:
        monitor = sct.monitors[1]  # Captures the primary monitor
        sct_img = sct.grab(monitor)
        # Convert to an array that OpenCV can use
        img = np.array(sct_img)
        # Drop the alpha channel, or you can use it depending on your model
        img = img[:, :, :3]

        if saveImg:
            if fileName is None:
                # Give it a timestamp so it is unique
                fileName = f"frames/screen_capture_{int(time.time())}.png"
            else:
                fileName = f"frames/{fileName}_{int(time.time())}.png"
            cv2.imwrite(fileName, img)
        return img

def preprocess_frame(frame, expand_dims=True):
    """
    Preprocess the given frame for input to the neural network.

    Args:
        frame (numpy.ndarray): The frame to preprocess.
        expand_dims (bool, optional): Whether to add batch and channel dimensions. Defaults to True.

    Returns:
        numpy.ndarray: The preprocessed frame.
    """
    if not expand_dims:
        # Convert the frame to grayscale to reduce the number of input channels
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame
    else:
        # Resize the frame to a smaller size, e.g., 128x72. You might need to experiment with this.
        frame = cv2.resize(frame, (128, 72))
        # Convert the frame to grayscale to reduce the number of input channels
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Add a batch dimension and channel dimension
        frame = np.expand_dims(frame, axis=0)
        frame = np.expand_dims(frame, axis=0)
        return frame

def ensure_grayscale(image):
    """
    Checks if an image is grayscale, and if not, converts it to grayscale.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Grayscale image.

    Raises:
        ValueError: If the input image is in an unexpected format.
    """
    # Check if the image has 2 dimensions (indicating it's already grayscale)
    if len(image.shape) == 2:
        # Image is already grayscale
        return image
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # Image is colored (BGR or RGB) and needs to be converted
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # Image is in an unexpected format (e.g., already grayscale or has an alpha channel)
        raise ValueError("The input image is in an unexpected format.")

def find_avatar(frame, saveImg=False, *, fileName=None):
    """
    Searches for the avatar in the given frame.

    Args:
        frame (numpy.ndarray): The current game frame (screenshot).
        saveImg (bool, optional): Whether to save the image with the detected avatar. Defaults to False.
        fileName (str, optional): The filename to use when saving the image. If not provided, a timestamp will be used.

    Returns:
        bool: A boolean indicating whether the avatar was found.
    """
    if frame.ndim == 3 and frame.shape[2] == 3:
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        # The frame is already grayscale or has an unexpected format
        gray_frame = frame  # Assuming frame is already grayscale if not in BGR format

    gray_avatar_template = ensure_grayscale(avatar_template)

    try:
        gray_frame_float = np.float32(gray_frame)
        avatar_template_float = np.float32(gray_avatar_template)
        # Perform template matching
        res = cv2.matchTemplate(gray_frame_float, avatar_template_float, cv2.TM_CCOEFF_NORMED)

        # Add a red box over the avatar
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + template_width, top_left[1] + template_height)
        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)

        if saveImg:
            if fileName is None:
                # Give it a timestamp so it is unique
                fileName = f"frames/screen_capture_{int(time.time())}.png"
            else:
                fileName = f"frames/{fileName}_{int(time.time())}.png"
            cv2.imwrite(fileName, frame)

    except:
        print(sys.exc_info())
        print(gray_frame.shape)  # Should be (height, width)
        print(avatar_template.shape)  # Should be (height, width)
        exit()

    # Define a threshold for detection
    threshold = 0.3248961  # You may need to adjust this based on testing

    # If the max correlation value is greater than the threshold, we found the avatar
    if np.max(res) >= threshold:
        return True
    else:
        return False