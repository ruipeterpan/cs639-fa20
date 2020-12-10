## /Tracking

We encapsulated both algorithms (meanshift & camshift) inside the class `Tracker` in `tracker.py`.

On initialization, `Tracker` reads in a local template image for histogram calculations. It also uses OpenCV template matching to find the initial window location and size.

When `track()` is invoked, it takes in an input image as a `numpy.ndarray`, chooses a tracking algorithm to run, and returns the image and the `(x, y, width, height)` coordinates of the bounding box.

### Usage

```python
from tracker import Tracker  # before this, modify the path to the template image in Tracker
import cv2
t = Tracker()
input = cv2.imread("")  # numpy.ndarray
pts, output = t.track(input, method="camshift")  # meanshift/camshift
output  # numpy.ndarray of the same size as the input
pts  # tuple that contains (x, y, width, height) of the bounding box
```
