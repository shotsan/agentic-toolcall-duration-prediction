from django.db import models
class Product(models.Model):
    sku = models.CharField(primary_key=True, max_length=50)
    class Meta:
        app_label = 'myapp'
class Order(models.Model):
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    class Meta:
        app_label = 'myapp'
