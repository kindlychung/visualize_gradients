#!/usr/bin/env bash



montage sigmoidfc*.png -tile 7x1 -geometry +3+3 sigmoid.png
montage    tanhfc*.png -tile 7x1 -geometry +3+3    tanh.png
montage    relufc*.png -tile 7x1 -geometry +3+3    relu.png
convert sigmoid.png  -background Khaki  label:'sigmoid' -gravity Center -append    sigmoid_anno.png
convert tanh.png     -background Khaki  label:'tanh'    -gravity Center -append       tanh_anno.png
convert relu.png     -background Khaki  label:'relu'    -gravity Center -append       relu_anno.png
montage    sigmoid_anno.png \
	      tanh_anno.png \
	      relu_anno.png -tile 3x1 -geometry +5+5     all.png
