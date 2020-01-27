# Load libraries
import flask
import tensorflow as tf
from tensorflow.keras.models import load_model as load_tf_model
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io
import numpy as np

target_size = (28, 28)
classes = [
  'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# instantiate flask 
app = flask.Flask(__name__)

sess = tf.Session()
set_session(sess)
global graph
graph = tf.get_default_graph()

model = load_tf_model('mnist_fashion.h5')

def prepare_image(image):
  # # if the image mode is not RGB, convert it
  # if image.mode != "RGB":
  #   image = image.convert("RGB")

  # resize the input image and preprocess it
  image = image.resize(target_size)
  image = img_to_array(image)
  image = image.reshape(1, 28, 28, 1)

  # return the processed image
  return image

# define a predict function as an endpoint 
@app.route("/test", methods=["GET","POST"])
def test():
  global sess
  global graph
  data = {"success": False}

  image = Image.open('./images/boot.png')
  image = image.resize(target_size)
  image = img_to_array(image) / 255.0
  image = image.reshape(1, 28, 28, 1)

  with graph.as_default():
    set_session(sess)
    data["prediction"] = model.predict(image).tolist()
    data["success"] = True

  # return a response in json format 
  return flask.jsonify(data)    

@app.route('/predict', methods=['POST'])
def predict():
  global sess
  global graph
  data = {"success": False}

  image = flask.request.files["image"].read()
  if image == None:
    return flask.jsonify({'image': 'Image is required'})

  image = Image.open(io.BytesIO(image))
  image = prepare_image(image)

  with graph.as_default():
    set_session(sess)
    prediction = model.predict(image).tolist()
    data["prediction"] = prediction
    data["success"] = True
    data['type'] = classes[np.argmax(prediction)]

  return flask.jsonify(data)

# start the flask app, allow remote connections 
app.run(host='0.0.0.0', debug=True)
