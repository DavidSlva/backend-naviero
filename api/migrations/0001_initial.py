# Generated by Django 5.1.2 on 2024-10-13 18:24

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='BarcosRecalando',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('nombre_barco', models.CharField(max_length=255)),
                ('puerto', models.CharField(max_length=50)),
                ('tipo_barco', models.CharField(max_length=255)),
                ('eslora', models.FloatField()),
                ('bandera', models.CharField(max_length=50)),
                ('agente', models.CharField(max_length=255)),
                ('carga', models.CharField(max_length=255)),
                ('detalle_operacion', models.CharField(max_length=255)),
                ('fecha_entrada', models.DateTimeField()),
                ('fecha_salida', models.DateTimeField()),
            ],
            options={
                'verbose_name': 'Barco recalando',
                'verbose_name_plural': 'Barcos recalando',
                'ordering': ['nombre_barco'],
            },
        ),
    ]
