#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "uwnet.h"

// Run an activation layer on input
// layer l: pointer to layer to run
// matrix x: input to layer
// returns: the result of running the layer y = f(x)
matrix forward_activation_layer(layer l, matrix x)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(x);

    ACTIVATION a = l.activation;
    matrix y = copy_matrix(x);

    // apply the activation function to matrix y
    // logistic(x) = 1/(1+e^(-x))
    // relu(x)     = x if x > 0 else 0
    // lrelu(x)    = x if x > 0 else .01 * x
    // softmax(x)  = e^{x_i} / sum(e^{x_j}) for all x_j in the same row 

    if (a == LOGISTIC) {
        for (int i = 0; i < x.rows * x.cols; i++) {
            y.data[i] =  1 / (1 + exp(-1.0 * x.data[i]));
        }
    } else if (a == RELU) {
        for (int i = 0; i < x.rows * x.cols; i++) {
            y.data[i] = (x.data[i] > 0) ? x.data[i] : 0;
        }
    } else if (a == LRELU) {
        for (int i = 0; i < x.rows * x.cols; i++) {
            y.data[i] = (x.data[i] > 0) ? x.data[i] : 0.01 * x.data[i];
        }
    } else if (a == SOFTMAX) {
        float tmp[x.rows];

        for (int i = 0; i <  x.rows; i++) {
            float sum = 0;
            for (int j = 0; j < x.cols; j++) {
                sum += exp(x.data[i * x.cols + j]);
            }
            tmp[i] = sum;
        }

        for (int i = 0; i < x.rows; i++) {
            for (int j = 0; j < x.cols; j++) {
                y.data[i * y.cols + j] = (exp(x.data[i * x.cols + j])) / tmp[i];
            }
        }
    }

    return y;
}

// Run an activation layer on input
// layer l: pointer to layer to run
// matrix dy: derivative of loss wrt output, dL/dy
// returns: derivative of loss wrt input, dL/dx
matrix backward_activation_layer(layer l, matrix dy)
{
    matrix x = *l.x;
    matrix dx = copy_matrix(dy);
    ACTIVATION a = l.activation;
 
    // calculate dL/dx = f'(x) * dL/dy
    // assume for this part that f'(x) = 1 for softmax because we will only use
    // it with cross-entropy loss for classification and include it in the loss
    // calculations
    // d/dx logistic(x) = logistic(x) * (1 - logistic(x))
    // d/dx relu(x)     = 1 if x > 0 else 0
    // d/dx lrelu(x)    = 1 if x > 0 else 0.01
    // d/dx softmax(x)  = 1

    if (a == LOGISTIC) {
        for (int i = 0; i < dy.rows * dy.cols; i++) {
            double val =  1/(1+exp(-1 * x.data[i]));
            dx.data[i] = (val  * (1 - val)) * dy.data[i];
           // dx.data[i] *= dy.data[i];
        }
    } else if (a == RELU) {
        for (int i = 0; i < dy.rows * dy.cols; i++) {
            dx.data[i] = (x.data[i] > 0) ? 1.0 : 0;
            dx.data[i] *= dy.data[i];
        }
    } else if (a == LRELU) {
        for (int i = 0; i < dy.rows * dy.cols; i++) {
           dx.data[i] = (x.data[i] > 0) ? 1.0 : 0.01;
           dx.data[i] *= dy.data[i];
        }
    } else if (a == SOFTMAX) {
        for (int i = 0; i < dx.rows * dy.cols; i++) {
            dx.data[i] = 1.0 * dy.data[i];
        }
    }
    return dx;
}

// Update activation layer..... nothing happens tho
// layer l: layer to update
// float rate: SGD learning rate
// float momentum: SGD momentum term
// float decay: l2 normalization term
void update_activation_layer(layer l, float rate, float momentum, float decay){}

layer make_activation_layer(ACTIVATION a)
{
    layer l = {0};
    l.activation = a;
    l.x = calloc(1, sizeof(matrix));
    l.forward = forward_activation_layer;
    l.backward = backward_activation_layer;
    l.update = update_activation_layer;
    return l;
}

