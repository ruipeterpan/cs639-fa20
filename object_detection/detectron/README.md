# detectron_predictor

# Setting up

Run `setup.sh` to install [detectron2](https://github.com/facebookresearch/detectron2) and its dependencies, mainly:

* Upgrade `torch` and `torchvision` so that they are compatible with detectron2.
* Install `pyyaml` for parsing YAML

# Usage

```python
import detectron_predictor as pr
p = pr.Predictor()
p.transform('input.png')
# output image is written to output.png by default
```

Note that the prediction latency is ~4s on a CPU and ~0.15s on a GPU in Google Colab.