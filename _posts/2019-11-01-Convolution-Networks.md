---
layout: post
title: Convolutional Neural Networks (Goodfellow Chapter 9)
---

The idea of convolutional neural networks (CNNs) ([LeCun, 1989](http://yann.lecun.com/exdb/publis/pdf/lecun-99.pdf)) is arguable the biggest innovation in deep learning so far. Its most promenent application is image classification and object detection. In this article I will provide a summary and some thoughts about the chapter on CNNs in the deep learning book by Goodfellow.

To quote Goodfellow: "Convolutional networks are simply neural networks that use convolution in place of general matrix multiplication in at least one of their layers". Using convolutions has several advantages: the number of trainable parameters is greatly reduced compared to a fully-connected neural network with an equal number of nodes. Furthermore, as we will see later, convolutions can be seen as some prior knowledge which is highly beneficial in the case of image recognition such as *equivariance*.

## Convolution Operator

A convolution operation consists of an input $I: m \times n$ and a kernel $K: k \times l$ and outputs the convolution $V = I * K$, where the convolution operator is defined as

$$
(I * K)(i,j) = \sum_{k,l}I(i+k,j+l)K(k,l).
$$

Graphically this can be represented as
![Graphical representation of convolution operation](https://camo.githubusercontent.com/3309220c48ab22c9a5dfe7656c3f1639b6b1755d/68747470733a2f2f7777772e64726f70626f782e636f6d2f732f6e3134713930677a386138726278622f32645f636f6e766f6c7574696f6e2e706e673f7261773d31)

To see how fully-connected neural networks are linked to convolutions, imagine what would happen when $K$ would have the same dimensions as $I$ and $K$ would have trainable weights. Exactly, we would simply get a summation of all input nodes which is exactly what a fully-connected neural network represents.

### Stride

Despite the fact that traditional convolutions already provide a large computional cost cut, we might want to further reduce the computional cost of CNNs by skipping over some positions of the input matrix $I$. This techique can be thought of a kind of down-sampling. Note that we therefore also reduce the computional and memory cost of the deeper layers since the input size is reduced. Algebraicly, a stride of size $s$ is defined as

$$
(I * K)(i,j,s) = \sum_{k,l}I(i*s+k,j*s+l)K(k,l).
$$

### Zero padding

Recall that the size of the kernel $K$ is typically smaller than the input $I$, so as we have deeper and deeper models, the layer size gets smaller and smaller. One solution would be to use small kernel sizes but this diminishes the expressive power of the kernels to learn non-trivial shapes.

To solve this problem we can augment our input $I$ using *zero padding*. With this technique we add zero nodes to the input such that we can maintain the same output size as the original input. As a result we can train deeper models more effectively.

One side effect is that pixels near the border of the image lose some of its signal, since they are surrounded by zero padding inputs.

## Motivation

There are three main motivations for using convolutions, namely:

- **Sparse interactions**: Image detection and/or classification models usually attempt to recognise certain shapes in an image. These shapes such as corners or edges are found in a small region of the image, therefore it is wasteful to try to detect this by having a fully-connected node. Instead we can train a kernel of a smaller size (say 100 by 100 pixels) for detection. Using these smaller kernels reduces the number of parameters significantly.
- **Parameter sharing**: The weight of the kernel is applied to many areas of the input space. Remember that traditional neural networks only apply every weight once for computing the output layer.
- **Equivariance**: This property tells us that if the input changes, the output changes in a similar way. As an example, if we would have a kernel that detects a box, moving this box to a different location in the image would result in detecting this box in its new location with the same kernel.

In short, convolutional neural networks allow us to train very deep models with significantly less parameters than its equally deep fully-connected variant. Furthermore, we can train the model to be robust for changes in location in for example an image.

## Pooling

A convolution layer in a neural network can be thought of a three step layer. We have the input layer and sequentially apply

1. Convolutional operations with $k$ kernels;
2. Detection by applying an activation layer to the output of the convolution of step 1;
3. Apply a pooling layer.

