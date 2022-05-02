# maskNPF

Official PyTorch code for [New particle formation event detection with Mask R-CNN](https://acp.copernicus.org/articles/22/1293/2022/acp-22-1293-2022.html). 

## Requirements
Please install the packages in the `requirements.txt`.
Download and put the pretrained models to the `checkpoints` folder. 
Two possible choices:
- Model trained on a subset of the Hyytiälä datasets (reproduce the results shown in the manuscript): [model](https://github.com/cvvsu/maskNPF/releases/download/v0.0/maskrcnn.pth)
- Model trained on all four datasets (recommend!): [model](https://github.com/cvvsu/maskNPF/releases/download/v0.1/maskrcnnfull.pth)
```
$ pip install -r requirements.txt
```
We find the versions of `PyTorch` (>=1.7) and `Python` (>=3.6) do not really affect the results. 

## Usage
1. Download the datasets to your local device.
```
$ python3 utils/get_datasets.py --station hyytiala --start_year 1990 --end_year 2021 --merge_df
```

2. run the code

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


***Note that the automatic mode fitting code is also released. But the automatic method may fail on some events. Be careful if you use the mode fitting code.***

## Citation
Please kindly cite our paper if you use our code.

```
@Article{acp-22-1293-2022,
AUTHOR = {Su, P. and Joutsensaari, J. and Dada, L. and Zaidan, M. A. and Nieminen, T. and Li, X. and Wu, Y. and Decesari, S. and Tarkoma, S. and Pet\"aj\"a, T. and Kulmala, M. and Pellikka, P.},
TITLE = {New particle formation event detection with Mask R-CNN},
JOURNAL = {Atmospheric Chemistry and Physics},
VOLUME = {22},
YEAR = {2022},
NUMBER = {2},
PAGES = {1293--1309},
URL = {https://acp.copernicus.org/articles/22/1293/2022/},
DOI = {10.5194/acp-22-1293-2022}
}
```

## License

The code is released under the [MIT License](https://github.com/cvvsu/maskNPF/blob/main/LICENSE). 
