from flask import Flask, jsonify
import pickle
import werkzeug
import flask
import scipy.misc
from PIL import Image

filename = 'ph_model.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

def getDetails(ph_value):
    desc=''
    fertilier = {
        "list" : [
            {"name": "NPK 12:11:18",
             "image": "https://5.imimg.com/data5/CI/MF/FF/SELLER-1393371/plantis-npk-12-11-18-500x500.jpg",
             "material": "Nitrogen, Phosporus, Potassium"},
            {"name": "NPK 16:8:24",
             "image": "https://5.imimg.com/data5/FF/CA/MY-1393371/npk-16-8-24-fertilizers-500x500.jpg",
             "material": "Nitrogen, Phosporus, Potassium"},
            {"name": "Master NPK",
             "image": "https://5.imimg.com/data5/CC/PB/MY-1393371/master-npk-fetilizers-500x500.png",
             "material": "Nitrogen, Phosporus, Potassium"},
            {"name": "NPK 20:20:20",
             "image": "https://5.imimg.com/data5/JQ/LP/MY-1393371/npk-20-20-20-fertilizers-500x500.jpg",
             "material": "Nitrogen, Phosporus, Potassium"},
            {"name": "NPK 5-15-45",
             "image": "https://5.imimg.com/data5/SN/DD/MY-1393371/npk-5-15-45-fertilizers-500x500.jpg",
             "material": "Nitrogen, Phosporus, Potassium"},
            {"name": "NPK 30-10-10",
             "image": "https://5.imimg.com/data5/GN/CH/MY-1393371/npk-30-10-10-fertilizers-500x500.jpg",
             "material": "Nitrogen, Phosporus, Potassium"},
            {"name": "NPK 13:40:13",
             "image": "https://5.imimg.com/data5/AW/YX/MY-1393371/npk-13-40-13-fertilizers-500x500.jpg",
             "material": "Nitrogen, Phosporus, Potassium"},
            {"name": "NPK 19:19:19",
             "image": "https://5.imimg.com/data5/JP/NK/MY-1393371/npk-19-19-19-fertilizers-500x500.jpg",
             "material": "Nitrogen, Phosporus, Potassium"},
            {"name": "NPK 6:12:36",
             "image": "https://5.imimg.com/data5/PG/VW/MY-1393371/npk-6-12-36-fertilizers-500x500.jpg",
             "material": "Nitrogen, Phosporus, Potassium"}
                ]
    }
    if ph_value>=6.0 and ph_value<=7.5:
        desc="Your Argiculture Land is ready to grown Crops."
        fertilizer={"list":[]}
    elif ph_value<6.0:
        desc="Land is Acidic, Some nutrients such as nitrogen, phosphorus, and potassium are less available."
    elif ph_value>7.5:
        desc="Land is very Alkaline, Iron, manganese, and phosphorus are less available."
    response = {
        'status': 200,
        'message': 'OK',
        'desc': desc,
        'fertilizer': fertilier,
        'ph_value': ph_value
    }
    return jsonify(response)

@app.route('/',methods=['GET','POST'])
def predict():
    imagefile = flask.request.files['image0']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save(filename)
    im = Image.open(filename)
    immat = im.load()
    (X, Y) = im.size
    l=[]
    avg=0
    x,y=X-(X//2),Y-(Y//2)
    l.append([x,y])
    l.append([x-(X//2)//2,y-(Y//2)//2])
    l.append([x+(X//2)//2,y-(Y//2)//2])
    l.append([x-(X//2)//2,y+(Y//2)//2])
    l.append([x+(X//2)//2,y+(Y//2)//2])
    image_rgb = im.convert("RGB")
    for i in l:
        rgb_pixel_value = image_rgb.getpixel((i[0],i[1]))
        avg+=model.predict([[rgb_pixel_value[0],rgb_pixel_value[1],rgb_pixel_value[2]]])
    return getDetails((avg/5)[0])

if __name__ == '__main__':
    app.run(debug=True)
