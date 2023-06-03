import boto3
from dotenv import dotenv_values

config = dotenv_values(".env")

ACCESS_KEY = config['ACCESS_KEY']
SECRET_KEY = config['SECRET_KEY']
SESSION_TOKEN = ''

client = boto3.client(
    's3',
    endpoint_url='https://storage.yandexcloud.net',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY
)

for key in client.list_objects(Bucket='pabdcv')['Contents']:
    print(key['Key'])

client.upload_file('models/my_model.zip', 'pabdcv', '1234/model.zip')
