# maskNPF

[Online pre-print paper](https://acp.copernicus.org/preprints/acp-2021-771/)

If you have any question, please open an issue or directly send an email to us.


## TODO
- ~~Upload the pre-trained model used in the manuscript.~~
- ~~Use a notebook to illustrate how to use the code step-by-step.~~
- Train the Mask R-CNN model on all NPF events from the four datasets and upload the trained model.

## Requirements
Please install the packages in the `requirements.txt`.
Download and put the [pretrained model](https://github.com/cvvsu/maskNPF/releases/download/v0.0/maskrcnn.pth) to the `checkpoints` folder. 

```
$ pip install -r requirements.txt
```
We find the versions of `PyTorch` (>=1.7) and `Python` (>=3.6) do not really affect the results. 

## Usage

```
$ python3 main.py --station hyytiala --im_size 256 --scores 0.0 --vmax 1e4 --model_name maskrcnn.pth
```

Check the `demo.ipynb` file for more information.


## Visualization 

[psd2im](https://github.com/cvvsu/maskNPF/blob/a9fba694765864962c8de1e3e7336c4d9dbb30d2/utils/utils.py#L18) is the base function used for visualization. It can be used to draw surface plots or NPF images.

[draw_subplots](https://github.com/cvvsu/maskNPF/blob/a9fba694765864962c8de1e3e7336c4d9dbb30d2/utils/utils.py#L148) is used to draw surface plots with one or several colorbars (Fig. 2 and Fig. 6).


## Masks

We currently use a simple GUI (based on the `tk` package) to select the detected masks for NPF event days. You can design a new one using Qt or other packages. Currently only one single maks can be selected. The one-day masks are used to calculate the GRs, while the two-day masks are used for determining the start and end times.

## GRs, start times, and end times

Our code can calculate GRs automatically for the size ranges: 
- 3 - 10 nm
- 10 - 25 nm
- 3 - 25 nm


## Citation
Please kindly cite our paper if you use our code.

```
@Article{acp-2021-771,
AUTHOR = {Su, P. and Joutsensaari, J. and Dada, L. and Zaidan, M. A. and Nieminen, T. and Li, X. and Wu, Y. and Decesari, S. and Tarkoma, S. and Pet\"aj\"a, T. and Kulmala, M. and Pellikka, P.},
TITLE = {New Particle Formation Events Detection with Deep Learning},
JOURNAL = {Atmospheric Chemistry and Physics Discussions},
VOLUME = {2021},
YEAR = {2021},
PAGES = {1--21},
URL = {https://acp.copernicus.org/preprints/acp-2021-771/},
DOI = {10.5194/acp-2021-771}
}
```

## License

The code is released under the [MIT License](https://github.com/cvvsu/maskNPF/blob/main/LICENSE). 
