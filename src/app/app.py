import os

from flask import Flask, render_template, request
from flask_dropzone import Dropzone
from flask import current_app as app
from app.services import s3,prediction

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)

app.config.update(
    UPLOADED_PATH=os.path.join(basedir, 'uploads'),
    # Flask-Dropzone config:
    DROPZONE_ALLOWED_FILE_TYPE='image',
    DROPZONE_MAX_FILE_SIZE=3,
    DROPZONE_MAX_FILES=130,
)

dropzone = Dropzone(app)

@app.route('/', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        for key, f in request.files.items():
            if key.startswith('file'):
                print(f.filename)
                print('Hello')
                # f.save(os.path.join(app.config['UPLOADED_PATH'], f.filename))
                s3.upload(f,'test_folder/'+f.filename)
    return render_template('index.html')

@app.route('/result', methods=['POST', 'GET'])
def result():
    c=prediction.predict_res()
    print(c)
    return render_template('tables.html',c=c)

@app.route('/custom', methods=['POST', 'GET'])
def custom():
    c=request.form.to_dict()
    imagerotation=c['imagerotation']
    zoomrange=c['zoomrange']
    widthshift=c['widthshift']
    horizantalshift=c['horizantalshift']
    verticalflip=c['verticalflip']
    model=c['model']
    batchsize=c['batchsize']
    epochs=c['epochs']
    stepsepoch=c['stepsepoch']
    validation=c['validation']
    print(c)
    return render_template('train.html',c=c)
    

if __name__ == '__main__':
    app.run(debug=True)
