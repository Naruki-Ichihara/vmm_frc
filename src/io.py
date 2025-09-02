import numpy as np
import cv2 as cv
import pydicom as dicom
from typing import Optional, Callable

def import_image(image_path: str,
                 cvt_control: Optional[int] = cv.COLOR_BGR2GRAY) -> np.ndarray:
    """
    Import an image from a file and convert it to a specified color space, using openCV.
    
    Args:
        image_path (str): Path to the image file.
        cvt_control (Optional[int]): OpenCV color conversion code. 
                                      Default is cv.COLOR_BGR2GRAY.

    Returns:
        np.ndarray: The imported image in the specified color space.
    """
    image = cv.imread(image_path)
    # print(f"[INFO] Importing: {image_path}")
    if cvt_control is not None:
        image = cv.cvtColor(image, cvt_control)
    return image

def import_dicom(image_path: str) -> np.ndarray:
    """
    Import a DICOM image from a file and convert it to a numpy array.
    
    Args:
        image_path (str): Path to the DICOM file.

    Returns:
        np.ndarray: The imported DICOM image as a numpy array.
    """
    ds = dicom.dcmread(image_path)
    image = ds.pixel_array
    return image

def _get_image_path(template: str,
                    index: int,
                    digit: int,
                    format: str) -> str:
    """
    Generate a formatted image path based on the template, index, digit, and format.

    Args:
        template (str): The template string for the image path.
        index (int): The index to be formatted into the path.
        digit (int): The number of digits for zero-padding.
        format (str): The format string for the image, e.g., 'png', 'jpg', 'tif'

    Returns:
        str: The formatted image path.
    """
    if digit > 0:
        index = f"{index}".zfill(digit)
    elif digit <= 0:
        raise ValueError("Digit must be greater than 0.")
    return template + index + "." + format

def import_image_sequence(template: str,
                          start_index: int,
                          end_index: int,
                          digit: int,
                          format: str,
                          save: Optional[bool] = False,
                          save_path: Optional[str] = None,
                          processing: Optional[Callable] = None,
                          cvt_control: Optional[int] = cv.COLOR_BGR2GRAY) -> np.ndarray:
    """
    Import a sequence of images from a specified template and range of indices.
    
    Args:
        template (str): The template string for the image path.
        start_index (int): The starting index for the image sequence.
        end_index (int): The ending index for the image sequence.
        digit (int): The number of digits for zero-padding.
        format (str): The format string for the image, e.g., 'png', 'jpg', 'tif'
        save (Optional[bool]): Whether to save the imported images as a numpy array.
        save_path (Optional[str]): The path to save the numpy array if save is True.
        processing (Optional[Callable]): A function to process each image after import.
        cvt_control (Optional[int]): OpenCV color conversion code. 
                                      Default is cv.COLOR_BGR2GRAY.
    Returns:
        np.ndarray: The imported image sequence as a numpy array.

    Raises:
        ValueError: If digit is less than or equal to 0.
        Exception: If there is an error importing the image sequence.
        """
    if processing is None:
        processing = lambda x: x
    print(f"[INFO] Importing sequence: {template} from {start_index} to {end_index}")
    if format == "dcm":
        volume = np.stack(
        [processing(import_dicom(_get_image_path(template, i, digit, format)))
         for i in range(start_index, end_index+1)], axis=0)
    else:
        try:
            volume = np.stack(
            [processing(import_image(_get_image_path(template, i, digit, format), cvt_control))
             for i in range(start_index, end_index+1)], axis=0)
        except Exception as e:
            print(f"Error importing image sequence: {e}")
            return None
    if save:
        if save_path is None:
            raise ValueError("Save path must be provided when save is True.")
        np.save(save_path, volume)
    print(f"[INFO] Imported {volume.shape[0]} images of shape {volume.shape[1:]}")
    return volume

def trim_image(image, start: tuple[int, int], end: tuple[int, int]) -> np.ndarray:
    """
    Trim an image to a specified region defined by start and end coordinates.

    Args:
        image (np.ndarray): The input image to be trimmed.
        start (tuple[int, int]): The starting coordinates (x, y) for the trim.
        end (tuple[int, int]): The ending coordinates (x, y) for the trim.

    Returns:
        np.ndarray: The trimmed image.
    """
    return image[start[1]:end[1], start[0]:end[0]]