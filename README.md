# resnet50

```
import tensorflow as tf
from matplotlib.image import imread
dog = imread('/Users/jhalverson/Downloads/newfoundland224.jpg')
dog = dog.reshape(1, 224, 224, 3)
mymodel = tf.keras.applications.ResNet50()
pred = mymodel.predict(dog)
tf.keras.applications.imagenet_utils.decode_predictions(pred, top=1)

Downloading data from https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json
40960/35363 [==================================] - 0s 1us/step
[[('n02111277', 'Newfoundland', 0.6345591)]]
```
