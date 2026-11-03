
from django.db import models
class Base(models.Model):
    base_id = models.AutoField(primary_key=True)
    field_base = models.IntegerField()
    class Meta:
        app_label = 'testapp'
        db_table = 'base'
class Child(Base):
    extra = models.IntegerField(default=0)
    class Meta:
        app_label = 'testapp'
        db_table = 'child'
