import boto3
from flask import current_app as app
import app.settings as settings


def upload(file,full_image_path):
    s3 = boto3.client('s3',aws_access_key_id =settings.ACESS_KEY ,aws_secret_access_key = settings.SECRET_ACESS_KEY)
    response = s3.put_object(ACL='public-read',Body = file,Bucket =settings.BUCKET_PATH,Key = full_image_path)
    print('Upload Done')
    return True

    
