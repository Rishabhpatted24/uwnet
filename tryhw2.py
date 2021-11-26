from uwnet import *
def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2),
            make_batchnorm_layer(8),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 8, 3, 2),
            make_convolutional_layer(8, 8, 8, 16, 3, 1),
            make_batchnorm_layer(16),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 16, 3, 2),
            make_convolutional_layer(4, 4, 16, 32, 3, 1),
            make_batchnorm_layer(32),
            make_activation_layer(RELU),
            make_connected_layer(512, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)


print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 500
rate = .01
momentum = .9
decay = .005

m = conv_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# 7.6 Question: What do you notice about training the convnet with/without batch normalization? How does it affect convergence? How does it affect what magnitude of learning rate you can use? Write down any observations from your experiments:
# TODO: 

# When training the convnet with batch normalization there was a significant increase (~13-14%) in training and testing accuracy. 
# Learning Rate: 0.01
#    For the convnet without batch normalization the training accuracy was 40.67800045013428% and test accuracy was 40.619999170303345%.
#    For the convnet with batch normalization the training accuracy was 54.1700005531311% and test accuracy was 53.21000218391418%.

# The convnet with batch normalization also converges much faster than the convnet as its loss decreases at a higher rate.

# Also, while the convnet with batch normalization could use learning rates between 0.01 and 0.1, using a high learning rate for a convnet without batchnormalization made it return much worse results:
# Learning rate: 0.1
#    For the convnet without batch normalization the training accuracy was 10.000000149011612% and test accuracy was 10.000000149011612%.
#    For the convnet with batch normalization the training accuracy was 52.8980016708374% and test accuracy was 52.0799994468689%.
# Learning rate: 0.07
#    For the convnet without batch normalization the training accuracy was 45.719999074935913% and test accuracy was 45.419999957084656%.
#    For the convnet with batch normalization the training accuracy was 54.39800024032593% and test accuracy was 52.96000242233276%.
# Learning rate: 0.05
#    For the convnet without batch normalization the training accuracy was 47.21600115299225% and test accuracy was 47.02000021934509%.
#    For the convnet with batch normalization the training accuracy was 55.49200177192688% and test accuracy was 54.43999767303467%.
# Learning rate: 0.03 (BEST PERFORMANCE)
#    For the convnet without batch normalization the training accuracy was 47.284001111984253% and test accuracy was 46.75000011920929%.
#    For the convnet with batch normalization the training accuracy was 55.4419994354248% and test accuracy was 54.54000234603882%.
# Thus, it seems that a convnet with batch normalization can use a more flexible learning rate value, while a convnet without batch normalization needs a smaller learning rate.
# The best performance for both models occured when using a learning rate of 0.03.