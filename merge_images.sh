#!/usr/bin/env bash


montage sigmoidfc*.png -tile 7x1 -geometry +3+3 sigmoid.png && rm sigmoidfc*.png
montage    tanhfc*.png -tile 7x1 -geometry +3+3    tanh.png && rm    tanhfc*.png
montage    relufc*.png -tile 7x1 -geometry +3+3    relu.png && rm    relufc*.png
montage    sigmoid.png \
	      tanh.png \
	      relu.png -tile 3x1 -geometry +5+5     all.png
