# Object detection using ResNet50 and Mask R-CNN

The `.ipynb` contains all the shell scripts and python scripts and can be run on the cloud.

To run locally, first run `setup.sh` to download the dataset and install project dependencies.

Next, run classfy.py to fine-tune the pretrained model. Save the model as `model.pth` for future use.

It is recommended to run the IPython Notebook on Google Collab and download the model checkpoint file from there.

## predictor usage
Put the `predictor.py` in the same directory as the end-user script.

```
import predictor as pr
p = pr.Predictor()
p.transform(...)  # takes in tensors of size [3, 512, 512] and outputs masks of size [512, 512].
```