import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Load model
        model_path = os.path.join(os.getcwd(), "artifacts", "training", "model.h5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        model = load_model(model_path)

        # Load and preprocess image
        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255.0  # Normalize image
        test_image = np.expand_dims(test_image, axis=0)

        # Predict
        prediction = model.predict(test_image)
        
        # Check if model uses softmax or sigmoid
        if prediction.shape[1] == 1:  # Binary classification (sigmoid)
            result = (prediction > 0.5).astype(int)
        else:  # Multi-class classification (softmax)
            result = np.argmax(prediction, axis=1)

        # Interpret results
        if result[0] == 1:
            return [{"image": "Tumor"}]
        else:
            return [{"image": "NoTumor"}]



# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import os
# # model_path = os.path.join(os.getcwd(), "artifacts", "training", "model.h5")  # Ensure correct absolute path
# # model = load_model(model_path)


# class PredictionPipeline:
#     def __init__(self,filename):
#         self.filename =filename


    
#     def predict(self):
#         # load model
#         model = load_model(os.path.join("artifacts", "training", "model.h5"))

#         imagename = self.filename
#         test_image = image.load_img(imagename, target_size = (224,224))
#         test_image = image.img_to_array(test_image)
#         test_image = np.expand_dims(test_image, axis = 0)
#         result = np.argmax(model.predict(test_image), axis=1)
#         print(result)

#         if result[0] == 1:
#             prediction = 'Tumor'
#             return [{ "image" : prediction}]
#         else:
#             prediction = 'Normal'
#             return [{ "image" : prediction}]
        
