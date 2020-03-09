# -*- coding: utf-8 -*-
from flask import Blueprint, render_template, request
import os
from flask import current_app as app
from app.services import s3


blueprint = Blueprint('home', __name__)

@blueprint.route('/')
def index():
    return render_template('index.html')

@blueprint.route('/', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        for key, f in request.files.items():
            if key.startswith('file'):
                print(f.filename)
                print('Hello')
                # f.save(os.path.join(app.config['UPLOADED_PATH'], f.filename))
                s3.upload(f,'test/'+f.filename)
    return render_template('index.html')