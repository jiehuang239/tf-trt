import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
def benchmark_saved_model(SAVED_MODEL_DIR,batch_size=4):
    with tf.Session(graph=tf.Graph(), config=config) as sess:

        tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING], SAVED_MODEL_DIR)
        start = time.time()
        for i in range (batch_size):
            img_path = './data/img%d.JPG'%(i%4)
            img = image.load_img(img_path,target_size=(224,224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            output = sess.run(OUTPUT_TENSOR, feed_dict={INPUT_TENSOR: x})
            print(type(output))
            print(output.shape)
            print(decode_predictions(output,top=3)[0])
        end = time.time()
        print("processing time = {}ms".format((end-start)/4*1000))
INPUT_TENSOR = 'input_1:0'
OUTPUT_TENSOR = 'fc1000/Softmax:0'
config = tf.ConfigProto()
saved_model_dir = 'resnet50_saved_model_TFTRT_FP16'
benchmark_saved_model(saved_model_dir,4)

