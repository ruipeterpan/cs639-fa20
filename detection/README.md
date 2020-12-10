## Detection

We encapsulated the detector in a class `Predictor` in `detectron_predictor.py`. On initialization, `Predictor` sets up detectron2 configurations, download a Mask-RCNN model (wrapped around ResNet50) from detectron2's model zoo, and sets up the default detectron2 predictor wrapper.

`detection.py` is used by our robotics framework -- please ignore it.

### Usage

```python
import detectron_predictor as pr
import cv2
p = pr.Predictor()
input = cv2.imread("")  # numpy.ndarray
output = p.transform(input)
output  # numpy.ndarray of the same size as the input
```
