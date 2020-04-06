# resnet50

```
import tensorflow as tf
from matplotlib.image import imread
dog = imread('/Users/jhalverson/Downloads/newfoundland224.jpg')
dog = dog.reshape(1, 224, 224, 3)
mymodel = tf.keras.applications.ResNet50()
pred = mymodel.predict(dog)
tf.keras.applications.imagenet_utils.decode_predictions(pred, top=1)
```
