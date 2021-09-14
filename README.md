# maskNPF

[Online pre-print paper](https://acp.copernicus.org/preprints/acp-2021-771/)

If you have any question, please open an issue or directly send an email to us.


## TODO
- Upload the pre-trained model used in the manuscript.
- Use a notebook to illustrate how to use the code step-by-step.
- Train the Mask R-CNN model on all NPF events from the four datasets and upload the trained model.
- Provide a fancy GUI.

## How to use the code

1. Please change the parameters in the `options.py` file. Specific parameters:

    - `im_size` : image size for processing. It is also possible to use the size 128*128. 
    - `scores`  : if you want analyze a long-term dataset, you can set a small value such as 0.2 or 0.3 to save time and effort. If you are dealing with one-year dataset, it is better to check all the detected masks (**0.0**).

2. Run the `main.py`. 

    - Drawing the NPF images does not take too much time.
    - However, the mask detection process is quite slow if you do not have a powerful GPU. [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb?utm_source=scs-index#) may provide a free GPU, which is a possible choice. After detecting the masks, you can download the masks to your local device for visualization.
    - You need to select the masks yourself (more than one detected masks for some days).
    - Once the masks are obtained, the GRs, start times, and end times can be determined automatically.

**NOTE**: make sure that your local device is powerful enough; otherwise, please modify the function [save_SE_GR](https://github.com/cvvsu/maskNPF/blob/a959edf04f794d70e7ef8979494e8f36e317326e/model.py#L285) in the `model.py` to calculate GRs one-by-one (do not use the `multiprocessing` package).


## Visualization 

`psd2im` is the base function used for visualization. It can be used to draw surface plots or NPF images.

`draw_subplots` is used to draw surface plots with one or several colorbars (Fig. 2 and Fig. 6).


## Masks

We currently use a simple GUI (based on the `tk` package) to select the detected masks for NPF event days. You can design a new one using Qt or other packages. Currently only one single maks can be selected. The one-day masks are used to calculate the GRs, while the two-day masks are used for determining the start and end times.

## GRs, start times, and end times

Our code can calculate GRs automatically for the size ranges: 

[3-10 nm](https://github.com/cvvsu/maskNPF/blob/a959edf04f794d70e7ef8979494e8f36e317326e/model.py#L249)

[10-25 nm](https://github.com/cvvsu/maskNPF/blob/a959edf04f794d70e7ef8979494e8f36e317326e/model.py#L250)

[3-25 nm](https://github.com/cvvsu/maskNPF/blob/a959edf04f794d70e7ef8979494e8f36e317326e/model.py#L253)

[size ranges determined by the detected masks](https://github.com/cvvsu/maskNPF/blob/a7c188f64e8329e0ae50ec23936158e4c89e07b0/model.py#L254)

You can change the size range to calculate other GRs, taking 3-7 nm as an example. The parameters are changed in the fuction [get_GRs](https://github.com/cvvsu/maskNPF/blob/a959edf04f794d70e7ef8979494e8f36e317326e/model.py#L239) in the `model.py`. Once other size ranges are added, please also change the [save_dict](https://github.com/cvvsu/maskNPF/blob/a959edf04f794d70e7ef8979494e8f36e317326e/model.py#L274) at the same time in the [get_SE_GR](https://github.com/cvvsu/maskNPF/blob/a959edf04f794d70e7ef8979494e8f36e317326e/model.py#L257) function.



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
