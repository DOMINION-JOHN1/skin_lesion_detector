from flask import Flask, render_template, request
import keras.utils as image
import numpy as np
import io
import tensorflow as tf
import base64


app = Flask(__name__)

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="model/model_float16_quant.tflite")
interpreter.allocate_tensors()

@app.route('/', methods=['GET', 'POST'])
def index():
    img_data = None
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Use in-memory image
            image_stream = io.BytesIO(uploaded_file.read())
            img = image.load_img(image_stream, target_size=(224, 224))

            # Convert PIL Image to data URL
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="PNG")
            img_data = base64.b64encode(img_buffer.getvalue()).decode()
            
            # Preprocess the image
            img_array = image.img_to_array(img)
            x_mean = img_array.mean()
            x_std = img_array.std()
            img_array = (img_array - x_mean) / x_std
            img_array = np.expand_dims(img_array, axis=0)

            # Set input tensor for the TFLite interpreter
            input_details = interpreter.get_input_details()
            interpreter.set_tensor(input_details[0]['index'], img_array)

            # Invoke the interpreter
            interpreter.invoke()

            # Get the prediction result
            output_details = interpreter.get_output_details()
            predictions = interpreter.get_tensor(output_details[0]['index'])
            predicted_class = np.argmax(predictions, axis=1)
            
            class_dict = {
                0: 'Actinic keratoses',
                1: 'Basal cell carcinoma',
                2: 'Benign keratosis-like lesions',
                3: 'Dermatofibroma_df',
                4: 'Melanocytic nevi',
                5: 'Vascular lesions',
                6: 'Dermatofibroma_mel' 
            }
            
            result = class_dict[predicted_class[0]]

            return render_template('index.html', prediction=result, img_data=img_data)
    return render_template('index.html', prediction=None, img_data=None)

if __name__ == '__main__':
    app.run(debug=True)
