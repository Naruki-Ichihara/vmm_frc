---
sidebar_position: 2
title: vmm.io
---

# vmm.io

Image and volume I/O utilities for loading CT scan data.

## Functions

### drop_edges_3D

```python
def drop_edges_3D(width: int, volume: np.ndarray) -> np.ndarray
```

Drop edges of 3D volume.

**Args:**
- `width` (int): Width of edges to remove.
- `volume` (np.ndarray): 3D volume.

**Returns:**
- `np.ndarray`: 3D volume without edges.

---

### trim_image

```python
def trim_image(start: tuple[int, int], end: tuple[int, int], image: np.ndarray) -> np.ndarray
```

Crop image.

**Args:**
- `start` (tuple[int, int]): Start coordinate.
- `end` (tuple[int, int]): End coordinate.
- `image` (np.ndarray): Image.

**Returns:**
- `np.ndarray`: Cropped image.

---

### import_image

```python
def import_image(path: str, cvt_control: Optional[int] = cv.COLOR_BGR2GRAY) -> np.ndarray
```

Import image from path.

**Args:**
- `path` (str): Path to the image.
- `cvt_control` (int, optional): cvtColor control number. Default is grayscale conversion.

**Returns:**
- `np.ndarray`: Imported image.

---

### import_dicom

```python
def import_dicom(path: str) -> np.ndarray
```

Import DICOM image from path.

**Args:**
- `path` (str): Path to the DICOM image.

**Returns:**
- `np.ndarray`: Imported image.

---

### get_image_path

```python
def get_image_path(path_template: str, index_of_image: int, number_of_digit: int, format: str) -> str
```

Get path of image with zero-padded index.

**Args:**
- `path_template` (str): Path template of image (without number and extension).
- `index_of_image` (int): Index of image.
- `number_of_digit` (int): Number of digits for zero-padding.
- `format` (str): Format of image (e.g., "tif", "png").

**Returns:**
- `str`: Full path of image.

---

### import_image_sequence

```python
def import_image_sequence(
    path_template: str,
    number_of_images: int,
    number_of_digits: int,
    format: str,
    initial_number: Optional[int] = 0,
    path_for_save: Optional[str] = None,
    process: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    cvt_control: Optional[int] = None
) -> np.ndarray
```

Import image sequence as 3D volume with convention [z, y, x].

**Args:**
- `path_template` (str): Path template of image sequence.
- `number_of_images` (int): Number of images to import.
- `number_of_digits` (int): Number of digits for zero-padding.
- `format` (str): Format of image (e.g., "tif", "png", "dcm").
- `initial_number` (int, optional): Starting index. Default is 0.
- `path_for_save` (str, optional): Path to save volume as .npy file.
- `process` (Callable, optional): Process function to apply to each image.
- `cvt_control` (int, optional): cvtColor control number.

**Returns:**
- `np.ndarray`: Imported image sequence as 3D volume.

**Example:**

```python
from vmm.io import import_image_sequence

# Import 100 TIF images starting from image0000.tif
volume = import_image_sequence(
    path_template="path/to/image",
    number_of_images=100,
    number_of_digits=4,
    format="tif"
)
print(volume.shape)  # (100, height, width)
```
