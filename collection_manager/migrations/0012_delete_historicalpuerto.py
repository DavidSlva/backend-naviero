# Generated by Django 5.1.3 on 2024-11-14 04:28

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('collection_manager', '0011_historicalpuerto_eslora_max_puerto_eslora_max'),
    ]

    operations = [
        migrations.DeleteModel(
            name='HistoricalPuerto',
        ),
    ]