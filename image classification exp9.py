from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow. keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras. applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np

#ResNet
model_resnet = ResNet50(weights='imagenet')          
img_path ='orange.JPG'
img = image.load_img(img_path, target_size= (224, 224))
x = image.img_to_array(img)
x = np.expand_dims (x, axis=0)
x = preprocess_input(x)
#model. summary ()
preds_resnet = model_resnet.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('***ResNe†****')
print('Predicted:', decode_predictions (preds_resnet, top=3)[0])
                       
model_vgg = VGG16 (weights='imagenet')

img_path = 'orange.JPG'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array (img)
x = np.expand_dims (x, axis=0)
x = preprocess_input (x)
#model.summary ()
preds_vgg = model_vgg.predict(x)
print('****VGG****')
print('Predicted:', decode_predictions (preds_vgg, top=3)[0])
     
model_inception = InceptionV3 (weights='imagenet')
                              
img_path = 'orange.JPG'
img = image. load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims (x, axis=0)
x = preprocess_input (x)
                              
preds_inception = model_inception.predict(x)
print('***INCEPTION****')
print('Predicted:', decode_predictions (preds_inception, top=3)[0])
