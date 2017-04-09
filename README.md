# **ENHAIM**

Image enhancement assignment for IP subject at college :ok_hand:

>   as requested by professor Moacir Ponti

## **USAGE**

The program requires the OpenCV library for matrix operations and image generation, numpy for mathematical operations and matplotlib for histogram plotting.

It can be simply run from a properly configured Windows cmd or Linux terminal, just head to the project's folder and type:

>   python enhaim.py

## **ENHANCEMENTS**

There are 4 mathematical functions for image enhancement, each of which the program will properly work on the desired input image and display on screen, alongside with a histogram comparison between the original image and the enhanced one:

* _Logarithmic Enhancement_

>   T(f) = c * log(1 + |f|), where c = 255 / log(1 + (R)), where R is the maximum intensity value found in f

* _Gamma Correction_

>   T(f) = f ^ y, where y is the desired gamma value

* _Histogram Equalization_

>   T(f) = (255 / MN) * hc(f), where M and N are the source's height and width and hc(f) is the cummulative histogram for the value in f

* _Sharpening Filter_

>   T(f) = a * f + b (convolution - f), where convolution is the resulting matrix of the convolution of f with the kernel w = [[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]]

Values calculated in the enhancements are then assigned into the output matrix as RGB values, with the following format:

>   out[x, y] = (result)

A RMSD value is also calculated, for analysis purposes. Please notice that each result is stored as float32, whereas the image shown is clipped as uint8.
