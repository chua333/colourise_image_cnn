import numpy as np
import cv2 as cv


# https://github.com/richzhang/colorization/tree/caffe/colorization/models
# https://github.com/richzhang/colorization/blob/caffe/colorization/resources/pts_in_hull.npy
# https://github.com/opencv/opencv/blob/master/samples/dnn/colorization.py

prototxt_path = "models/colorization_deploy_v2.prototxt"
model_path = "models/colorization_release_v2.caffemodel"
kernel_path = "models/pts_in_hull.npy"
image_path = "images/bw_lion.jpg"

net = cv.dnn.readNetFromCaffe(prototxt_path, model_path)
points = np.load(kernel_path)

points = points.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype("float32")]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full((1, 313), 2.606, dtype="float32")]

bw_image = cv.imread(image_path)
normalized = bw_image.astype("float32") / 255.0
lab = cv.cvtColor(normalized, cv.COLOR_BGR2Lab)

resized = cv.resize(lab, (224, 224))
L = cv.split(resized)[0]
L -= 50

net.setInput(cv.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

ab = cv.resize(ab, (bw_image.shape[1], bw_image.shape[0]))
L = cv.split(lab)[0]

colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
colorized = cv.cvtColor(colorized, cv.COLOR_Lab2BGR)
colorized = (255 * colorized).astype("uint8")

cv.imshow("Original", bw_image)
cv.imshow("Colorized", colorized)
cv.waitKey(0)
cv.destroyAllWindows()
