import tensorflow as tf
import Run_All_BOW_2021csm1002



if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    # x_train = x_train[:1000]
    # y_train = y_train[:1000]




    train_feature , train_descriptor, train_desc = Run_All_BOW_2021csm1002.sift(x_train)

    test_feature, test_descriptor , test_desc = Run_All_BOW_2021csm1002.sift(x_test)

    # Run_All_BOW_2021csm1002.elbow_method(train_descriptor)

    optimal_k,kmeans,bovw,dists = Run_All_BOW_2021csm1002.CreateVisualDictionary(train_feature,train_descriptor,train_desc)

    train_histogram = Run_All_BOW_2021csm1002.ComputeHistogram(train_feature,bovw,kmeans)

    test_histogram = Run_All_BOW_2021csm1002.ComputeHistogram(test_feature, bovw)



    Xtrain_hist,Ytrain,Xtest_hist,Ytest = Run_All_BOW_2021csm1002.get_train_and_test(train_histogram, y_train,test_histogram,y_test)


    distances = Run_All_BOW_2021csm1002.MatchHistogram(Xtrain_hist,Xtest_hist,Ytrain,Ytest,train_desc,bovw)