import cv2
import os


def readImgs(directory=None):
    """
    Read images from a directory (default dir is "./images"), use cv2 to read the images and return a list of images
    as "numpy.ndarray"s.
    :param directory: String. Specifies the directory in which the images are stored.
    :return: A list of images read by cv2.imread().
    """
    if directory is None:
        directory = "./images"

    images = os.listdir(directory)
    images = [os.path.join(directory, i) for i in images]
    images = [cv2.imread(i) for i in images]
    return images


def stitch(images):
    """
    Create a stitcher, use it to stitch the images.
    :param images: A list of images read by cv2.imread().
    :return: None
    """
    stitcher = cv2.Stitcher_create(0)
    result = stitcher.stitch(images)
    if result[0]:
        raise Exception("Process failed with error code", result[0],
                        "--See https://stackoverflow.com/a/36645276/9601555 for more info")

    cv2.imwrite("./images/result.jpg", result[1])

    print("Stitch successful -- images is written to ./images/result.jpg!")


def main():
    images = readImgs("/home/haochen/Tools/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04/vision_images")
    stitch(images)


if __name__ == '__main__':
    main()
