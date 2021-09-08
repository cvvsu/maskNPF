# maskNPF

The code will be released later. Please `stay tuned`.

If you have any question, please open an issue or directly send an email to us.

## Visualization 

`psd2im` is the base function used for visualization. It can be used to draw surface plots or NPF images.

`draw_subplots` is used to draw surface plots with one or several colorbars (Fig. 2 and Fig. 6).

## Get masks

We currently use a simple GUI (based on the `tk` package) to select the detected masks for NPF event days. You can design a new one using Qt or other packages.

## GRs, start times, and end times

Our code can calculate GRs automatically for the size ranges: 

    1. 3-10 nm
    2. 10-25 nm
    3. 3-25 nm

You can change the size range to calculate other GRs, taking 3-7 nm as an example.


## TODO
- The pre-trained model used in the manuscript.
- Using a notebook to illustrate how to use the code.
- Train the Mask R-CNN model on all NPF events from the four datasets and upload the trained model.
- A fancy GUI.

## License

The code is released under the [MIT License](https://github.com/cvvsu/maskNPF/blob/main/LICENSE). 
