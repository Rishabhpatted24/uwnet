#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"


// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(in);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);

    // TODO: 6.1 - iterate over the input and fill in the output with max values
    int first_center = (l.size - 1) / 2;
    int out_index = 0;
	for (int channel = 0; channel < l.channels; channel++) {
        for (int y = 0; y < l.height; y += l.stride) { // moving the kernel ahead in y direction.
            for (int x = 0; x < l.width; x += l.stride) { // moving the kernel ahead in the x direction.
                float kernal_max = 0;
                for (int y_kernel = 0; y_kernel < l.size; y_kernel++) {
                    for (int x_kernel = 0; x_kernel < l.size; x_kernel++) {
			            int x_loc = x + x_kernel - first_center;
			            int y_loc = y + y_kernel - first_center;

                        if ( !(x_loc < 0 || x_loc >= l.width || y_loc < 0 || y_loc >= l.height) ) {
                            kernal_max = kernal_max > in.data[channel*l.width*l.height + y_loc*l.width + x_loc] ? kernal_max : in.data[channel*l.width*l.height + y_loc*l.width + x_loc];
                        }

                    }
                }
                out.data[out_index] = kernal_max;
                
                out_index++;
            }
        }
    }

    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix dy: error term for the previous layer
matrix backward_maxpool_layer(layer l, matrix dy)
{
    matrix in    = *l.x;
    matrix dx = make_matrix(dy.rows, l.width*l.height*l.channels);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.
    int first_center = (l.size - 1) / 2;
    int out_index = 0;
	for (int channel = 0; channel < l.channels; channel++) {
        for (int y = 0; y < l.height; y += l.stride) { // moving the kernel ahead in y direction.
            for (int x = 0; x < l.width; x += l.stride) { // moving the kernel ahead in the x direction.
                float kernal_max = 0;
                int kernal_max_x = 0;
                int kernal_max_y = 0;
                for (int y_kernel = 0; y_kernel < l.size; y_kernel++) {
                    for (int x_kernel = 0; x_kernel < l.size; x_kernel++) {
			            int x_loc = x + x_kernel - first_center;
			            int y_loc = y + y_kernel - first_center;
                        if (!(x_loc < 0 || x_loc >= l.width || y_loc < 0 || y_loc >= l.height) ) {
                            if (in.data[channel*l.width*l.height + y_loc*l.width + x_loc] > kernal_max) {
                                kernal_max_x = x_loc;
                                kernal_max_y = y_loc;
                                kernal_max =  in.data[channel*l.width*l.height + y_loc*l.width + x_loc];
                            } 
                        }

                    }
                }
                dx.data[channel*l.width*l.height + kernal_max_y*l.width + kernal_max_x] += dy.data[out_index];
                out_index++;
            }
        }
    }

    return dx;
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay){}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.x = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

