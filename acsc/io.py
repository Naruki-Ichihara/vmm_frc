import numpy as np
import cv2 as cv
import pydicom as dicom
from typing import Optional, Callable, Union

def drop_edges_3D(width: int, volume: np.ndarray) -> np.ndarray:
    """Drop edges of 3D volume.

    Args:
        width (int): Width of edges.
        volume (np.ndarray): 3D volume.

    Returns:
        np.ndarray: 3D volume without edges.

    """
    return volume[width:volume.shape[0]-width, width:volume.shape[1]-width, width:volume.shape[2]-width]

def trim_image(start: tuple[int, int], end: tuple[int, int], image: np.ndarray) -> np.ndarray:
    """Crop image.

    Args:
        start (tuple[int, int]): Start coordinate.
        end (tuple[int, int]): End coordinate.
        image (np.ndarray): Image.

    Returns:
        np.ndarray: Cropped image.

    """
    return image[start[0]:end[0], start[1]:end[1]]

def import_image(path: str, cvt_control: Optional[int] = cv.COLOR_BGR2GRAY) -> np.ndarray:
    """Import image from path.
    
    Args:
        path (str): Path to the image.
        cvt_control (int): cvtColor control number.

    Returns:
        np.ndarray: Imported image.
    
    """
    image = cv.imread(path)
    if cvt_control is not None:
        image = cv.cvtColor(image, cvt_control)
    cupy_image = np.asarray(image)
    return cupy_image

def import_dicom(path: str) -> np.ndarray:
    """Import DICOM image from path.
    
    Args:
        path (str): Path to the image.
        cvt_control (int): cvtColor control number.

    Returns:
        np.ndarray: Imported image.
    """
    dicom_image = dicom.dcmread(path)
    image = dicom_image.pixel_array
    cupy_image = np.asarray(image)
    return cupy_image

def get_image_path(path_template: str, 
                   index_of_image: int, 
                   number_of_digit: int, 
                   format: str) -> str:
    """Get paths of images.

    Args:
        path_template (str): Path template of image.
        number_of_images (int): Number of images.
        number_of_digit (int): Number of digits.
        format (str): Format of image.

    Returns:
        str: Path of image.
    
    """
    digit = f'{index_of_image}'.zfill(number_of_digit)
    path_of_image = path_template + digit + '.' + format
    return path_of_image

def import_image_sequence(path_template: str, 
                          number_of_images: int, 
                          number_of_digits: int,
                          format: str,
                          initial_number: Optional[int] = 0,
                          path_for_save: Optional[str] = None,
                          process: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                          cvt_control: Optional[int] = None) -> np.ndarray:
    """ the image sequence is imported as volume. The comvection is assumed to be
    [plane, row, column], or with the direction [z, y, x].

    Args:
        sequence_path (str): Path template of image sequence.
        number_of_images (int): Number of images.
        number_of_digits (int): Number of digits.
        format (str): Format of image.
        process (Optional[Callable[[np.ndarray], np.ndarray]]): Process function of image.

    Returns:
        np.ndarray: Imported image sequence.
    
    """
    
    if process is None:
        process = lambda image: image
    if format == "dcm":
        volume = np.stack(
        [process(import_dicom(get_image_path(path_template, i, number_of_digits, format)))
         for i in range(initial_number, number_of_images)], axis=0)
    else:
        volume = np.stack(
        [process(import_image(get_image_path(path_template, i, number_of_digits, format), cvt_control))
         for i in range(initial_number, number_of_images)], axis=0)
    
    if path_for_save is not None:
        np.save(path_for_save, volume)

    return volume