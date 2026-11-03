import os
from django.apps import AppConfig
class MyappConfig(AppConfig):
    name = 'myapp'
    path = os.path.dirname(__file__)
