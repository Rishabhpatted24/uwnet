from uwnet import *

def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(32, 32, 8, 3, 2),
            make_convolutional_layer(16, 16, 8, 16, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 16, 3, 2),
            make_convolutional_layer(8, 8, 16, 32, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 32, 3, 2),
            make_convolutional_layer(4, 4, 32, 64, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(4, 4, 64, 3, 2),
            make_connected_layer(256, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

def connected_net():
    l = [   make_connected_layer(3072, 336),
            make_activation_layer(RELU),
            make_connected_layer(336,184),
            make_activation_layer(RELU),
            make_connected_layer(184, 64),
            make_activation_layer(RELU),
            make_connected_layer(64, 32),
            make_activation_layer(RELU),
            make_connected_layer(32, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 5000
rate = .01
momentum = .9
decay = .005

# m = conv_net()
m = connected_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# How accurate is the fully connected network vs the convnet when they use similar number of operations?
# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
# Your answer:
# 
# The number of operations for 1 forward pass through the convolutional network is: 1,108,480.
# The convolutional network's train accuracy was: 70.51799893379211%
# The convolutional network's test accuracy was: 65.75999855995178%
#
# The number of operations for 1 forward pass through the connected network is: 1,108,160.
# The connected network's train accuracy was: 53.79199981689453%
# The connected network's test accuracy was: 49.75999891757965%
#
# The convolutional network has a significantly higher accuracy than the fully connected network when using a similar number of operations.
# We believe the reason the convolutional network performs better is because it extracts the local features much better than the fully connected network.
# We believe this is because the convolutional network to identify the parts of each image that contribute most to that image's classification and use that
# when classifing an image, while the fully connected network uses each pixel equally during its prediction which is not as useful for image classification.

