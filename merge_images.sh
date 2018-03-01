#!/usr/bin/env bash


montage sigmoid*.png -tile 7x1 -geometry +3+3 sigmoid.png
montage    tanh*.png -tile 7x1 -geometry +3+3    tanh.png
montage    relu*.png -tile 7x1 -geometry +3+3    relu.png
montage  sigmoid.png \
	    tanh.png \
	    relu.png -tile 3x1 -geometry +5+5     all.png
