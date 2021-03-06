# Visualization of the gradients of a neural network using three activation functions

![](https://github.com/kindlychung/visualize_gradients/blob/master/all.png)

This project is a study of effects that different activation functions have on how the gradients change in time.
A simple 7-layer fully connected neural networks was used on a simulated dataset with 3 features and a binary output.
Tested activation functions include:

* Sigmoid
* Tanh
* ReLU

## Prerequisites

### pip installable

* python 3
* pytorch
* numpy
* scipy
* scikit-image

### apt gettable

* imagemagick

## Run

Tested on Ubuntu 17.10, python 3.6:

```
python main.py && python visualize.py && ./merge_images.sh && xdg-open all.png
```

## How interpret the plot

Three sections are annotationed by activation method. Each section has 7 lanes, standing for the gradients from the 7 neural net layers. In each lane, each row is the gradient value in one backprop step, from top to bottom, in order of time (top is earliest, bottom is latest). 

## Lessons learned in this specific case

* Both `tanh` and `relu` converges better than `sigmoid`
* Both `tanh` and `relu` results in more sparse gradients than `sigmoid`, `relu` is the most sparse
* Both `tanh` and `relu` results in good accuracy, while `sigmoid` performs poorly
* The `fc6` layer seems to have the vanishing gradient problem when `sigmoid` is used


