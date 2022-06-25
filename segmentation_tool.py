from flask import Flask, render_template, request, url_for,redirect
from werkzeug.utils import secure_filename
import os,cv2
from keras.models import load_model
import numpy as np

model=load_model('D:\Python Programming\Flask\static\cat_vs_dog.h5')

def prediction(file_path):
    img=cv2.imread(file_path)
    img=cv2.resize(img, (224,224),interpolation = cv2.INTER_NEAREST)
    img=np.expand_dims(img, axis=0)
    result=model.predict(img)
    return result

# FLASK APPLICATION
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/Segmentor/', methods=['POST','GET'])
def Segmentor():
    return render_template('segment.html')

@app.route('/upload',methods=['GET','POST'])
def upload():
    if request.method == 'POST':
      f = request.files['file']
      file_path=os.path.join(r'D:\Python Programming\Flask\static',secure_filename(f.filename))
      f.save(file_path)
      result=prediction(file_path)
      if result<=0.5:
        return 'CAT'
      else:
        return 'DOG0'







if __name__=="__main__":
    app.run(debug=True)