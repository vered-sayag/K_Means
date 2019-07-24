import numpy as np
#import scipy.io as sio
#import imageio
#import matplotlib.pyplot as plt
from init_centroids import init_centroids
from scipy.misc import imread



def printIter(loop, K, centroids):
    print("iter %d: " % (loop), end='')
    for j in range(0, K):
        if np.floor(centroids[j][0]*100)/100 == 0:
            print("[0., ", end='')
        else:
            print("[{}, ".format(np.floor(centroids[j][0]*100)/100),end='')

        if np.floor(centroids[j][1] * 100) / 100 == 0:
            print("0., ", end='')
        else:
            print("{}, ".format(np.floor(centroids[j][1] * 100) / 100), end='')

        if np.floor(centroids[j][2] * 100) / 100 == 0:
            print("0.]", end='')
        else:
            print("{}]".format(np.floor(centroids[j][2] * 100) / 100), end='')
        if j != K - 1:
            print(", ", end='')
    print("")


def main():
    # data preperation (loading, normalizing, reshaping)
    path = 'dog.jpeg'

    K = 2
    while K < 17:
        A = imread(path)
        A_norm = A.astype(float) / 255.
        img_size = A_norm.shape
        X = A_norm.reshape(img_size[0] * img_size[1], img_size[2])
        centroids = init_centroids(X, K)
        print('k=%d:' % K)
        distribution = []
        # loss = []
        # for i in range(0, 11):
        #   loss.append(0)

        for i in range(0, img_size[0] * img_size[1]):
            distribution.append(0)

        printIter(0, K, centroids)

        for loop in range(1, 11):
            for i in range(0, img_size[0] * img_size[1]):
                distribution[i] = 0
                minDest = np.linalg.norm(X[i] - centroids[0])
                for j in range(0, K):
                    if minDest > np.linalg.norm(X[i] - centroids[j]):
                        distribution[i] = j
                        minDest = np.linalg.norm(X[i] - centroids[j])
                # loss[loop-1]+=minDest

            # loss[loop-1]/=img_size[0] * img_size[1]
            # print("iter ",loop-1," : loss ",loss[loop-1])

            for j in range(0, K):
                numPix = 0;
                sumPix = [0, 0, 0]
                for i in range(0, img_size[0] * img_size[1]):
                    if distribution[i] == j:
                        numPix = numPix + 1
                        sumPix = sumPix + X[i]
                if numPix != 0:
                    centroids[j] = sumPix / numPix

            printIter(loop, K, centroids)
        K = K * 2

        # plt.plot(loss)
        # plt.title('K = %d' % K)
        # plt.ylabel('loss')
        # plt.xlabel('iteration')
        # plt.show()

        # for i in range(0, img_size[0] * img_size[1]):
        #    for j in range(0, K):
        #        if distribution[i] == j:
        #            X[i] = centroids[j]

        # A_new = np.reshape(X, (img_size[0], img_size[1], img_size[2]))
        # plt.imshow(A_new)
        # plt.grid(False)
        # plt.show()


if __name__ == "__main__":
    main()
