from flask import Flask
import pickle
import werkzeug
import numpy
import scipy.misc
from PIL import Image

filename = 'ph_model.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

avg=0

@app.route('/',methods=['GET','POST'])
def predict():
    imagefile = flask.request.files['image0']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save(filename)
    im = Image.open('land.jpg')
    immat = im.load()
    (X, Y) = im.size
    #img = scipy.misc.imread(filename, mode="L")
    #img = img.reshape(784)
    l=[]
    l.append([x,y])
    l.append([x-(X//2)//2,y-(Y//2)//2])
    l.append([x+(X//2)//2,y-(Y//2)//2])
    l.append([x-(X//2)//2,y+(Y//2)//2])
    l.append([x+(X//2)//2,y+(Y//2)//2])
    image_rgb = im.convert("RGB")
    for i in l:
        rgb_pixel_value = image_rgb.getpixel((i[0],i[1]))
        avg+=model.predict([[rgb_pixel_value[0],rgb_pixel_value[1],rgb_pixel_value[2]]])
    return str(avg/5)


if __name__ == '__main__':
    app.run(debug=True)
