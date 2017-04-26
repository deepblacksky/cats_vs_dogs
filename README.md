# cats_vs_dogs
This is a tensorflow tutorials of CNN using own data.

The data is from kaggle, [cats_vs_dogs](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)
## Process:

1. read data
2. create input queue
3. create batch
4. train
5. save model architectrue and weigths
6. restore model
7. test new data

## test_version

The first version, there is not test data, only can input single image to test.

In the test version, the all data have split to train data and test data by 8:2.

So, the there will be train loss, train accuracy and test loss, test accuracy.

## the model 

![](https://github.com/deepblacksky/cats_vs_dogs/blob/master/result/graph2.png?raw=true)

## result

![](https://github.com/deepblacksky/cats_vs_dogs/blob/master/result/output1.png?raw=true)

![](https://github.com/deepblacksky/cats_vs_dogs/blob/master/result/output2.png?raw=true)

![](https://github.com/deepblacksky/cats_vs_dogs/blob/master/result/loss.png?raw=true)

![](https://github.com/deepblacksky/cats_vs_dogs/blob/master/result/accuracy.png?raw=true)

