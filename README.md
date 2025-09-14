# PAIR-X
Demo code for Pairwise mAtching of Intermediate Representations for eXplainability (PAIR-X).

## Citation
PAIR-X was developed by Lauren Shrack in the [BeeryLab at MIT](https://beerylab.csail.mit.edu/). 

```
@misc{shrack2025pairwisematchingintermediaterepresentations,
      title={Pairwise Matching of Intermediate Representations for Fine-grained Explainability}, 
      author={Lauren Shrack and Timm Haucke and Antoine Salaün and Arjun Subramonian and Sara Beery},
      year={2025},
      eprint={2503.22881},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.22881}, 
}
```
[The original repository is here](https://github.com/pairx-explains/pairx).

## To run the demo

1. Clone this repository

       git clone https://github.com/WildMeOrg/pairx.git
       cd pairx
2. Create a virtual environment (python 3.10 is currently recommended), and install pairx
        
       python -m venv .venv
       # or with uv, whichever is your preference ... 
       source .venv/bin/activate
       pip install -e .
3. Run the example

       cd examples 
       python demo.py
4. View the PAIR-X output in the `output` directory.

## Quickstart: Running on new datasets and models

### Imports

Clone the repository as described above. Then include the needed imports:

       import torch

       from pairx.core import explain
       from pairx.dataset import get_img_pair_from_paths
       from pairx.loaders import wildme_multispecies_miewid

       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Load the model components

To use [WildMe multispecies MiewID](https://huggingface.co/conservationxlabs/miewid-msv2), you can use:

       model, img_size, img_transforms = wildme_multispecies_miewid(device)
Otherwise, you should supply each variable for your own model:

- `model`: The trained model for your dataset. It should be callable with `model(img)`.
- `img_size`: A `(w, h)` tuple with the input image size of the model, e.g., `(440, 440)`.
- `img_transforms`: Any transforms that should be applied to the images before the model is called, as a function.

### Load the images

If you want to run ad hoc on a pair of images, you can simply use:

       img_paths = ["examples/data/cow_0_0.jpg", "examples/data/cow_0_1.jpg"]
       img_0, img_1, img_np_0, img_np_1 = get_img_pair_from_paths(device,
                                                                  f"{img_paths[0]}",
                                                                  f"{img_paths[1]}",
                                                                  img_size,
                                                                  img_transforms
                                                                  )

This loads the images with and without transforms applied, so that the untransformed images can be used for visualizations.

### Run PAIR-X

With the model and images loaded, you can run PAIR-X using:

       imgs = explain(device,
                      img_0,                     # First image, transformed
                      img_1,                     # Second image, transformed      
                      img_np_0,                  # First image, pretransform
                      img_np_1,                  # Second image, pretransform
                      model,                     # Model
                      ["backbone.blocks.3"],     # Layer keys for intermediate layers (read below about choosing this)
                      k_lines=20,                # Number of matches to visualize as lines
                      k_colors=10                # Number of matches to visualize in fine-grained color map
                      )

#### Choosing an intermediate layer:

The best intermediate layer to use may vary by model and dataset. To choose one for your problem, we suggest generating visualizations for a few layers and manually selecting one. The `explain` function allows you to supply multiple layer keys, producing a visualization for each.

You can efficiently list all the layer keys in your model using `helpers.list_layer_keys(model, max_depth=4)`. In our experiments, we typically compared across each of the higher-level blocks in a model (e.g., for an EfficientNet backbone, we tried `layer_keys = (blocks.1, blocks.2, blocks.3, blocks.4, blocks.5)`).






       
