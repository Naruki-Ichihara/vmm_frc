import src.io as io

def test_import_image_sequence():
    image_sequence = io.import_image_sequence("tests/test_images/test_",
                                           0,
                                           4,
                                           4,
                                           "tif")
    assert image_sequence.shape == (5, 1024, 1024)

def test_trim_image():
    trim = lambda x: io.trim_image(x, [200, 200], [300, 300])
    image_sequence = io.import_image_sequence("tests/test_images/test_",
                                           0,
                                           4,
                                           4,
                                           "tif",
                                           processing=trim)
    assert image_sequence.shape == (5, 100, 100)
