# Generated by Django 5.1.2 on 2024-10-13 19:35

import django.db.models.deletion
import simple_history.models
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Aduana',
            fields=[
                ('codigo', models.IntegerField(primary_key=True, serialize=False)),
                ('nombre', models.CharField(max_length=255)),
                ('longitud', models.FloatField()),
                ('latitud', models.FloatField()),
            ],
            options={
                'verbose_name': 'Aduana',
                'verbose_name_plural': 'Aduanas',
                'ordering': ['nombre'],
            },
        ),
        migrations.CreateModel(
            name='Clausula',
            fields=[
                ('codigo', models.IntegerField(primary_key=True, serialize=False)),
                ('nombre', models.CharField(max_length=255)),
                ('sigla', models.CharField(max_length=255)),
            ],
            options={
                'verbose_name': 'Clausula',
                'verbose_name_plural': 'Clausulas',
                'ordering': ['nombre'],
            },
        ),
        migrations.CreateModel(
            name='ModalidadVenta',
            fields=[
                ('codigo', models.IntegerField(primary_key=True, serialize=False)),
                ('nombre', models.CharField(max_length=255)),
                ('descripcion', models.CharField(max_length=255)),
            ],
            options={
                'verbose_name': 'Modalidad de venta',
                'verbose_name_plural': 'Modalidades de venta',
                'ordering': ['nombre'],
            },
        ),
        migrations.CreateModel(
            name='Pais',
            fields=[
                ('codigo', models.IntegerField(primary_key=True, serialize=False)),
                ('nombre', models.CharField(max_length=255)),
                ('continente', models.CharField(max_length=255)),
            ],
            options={
                'verbose_name': 'Pais',
                'verbose_name_plural': 'Paises',
                'ordering': ['nombre'],
            },
        ),
        migrations.CreateModel(
            name='RegimenImportacion',
            fields=[
                ('codigo', models.IntegerField(primary_key=True, serialize=False)),
                ('nombre', models.CharField(max_length=255)),
                ('sigla', models.CharField(max_length=255)),
                ('active', models.BooleanField()),
            ],
            options={
                'verbose_name': 'Regimen de importación',
                'verbose_name_plural': 'Regimenes de importación',
                'ordering': ['nombre'],
            },
        ),
        migrations.CreateModel(
            name='Region',
            fields=[
                ('codigo', models.IntegerField(primary_key=True, serialize=False)),
                ('nombre', models.CharField(max_length=255)),
            ],
            options={
                'verbose_name': 'Región',
                'verbose_name_plural': 'Regiones',
                'ordering': ['nombre'],
            },
        ),
        migrations.CreateModel(
            name='TipoCarga',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('codigo', models.CharField(max_length=50)),
                ('nombre', models.CharField(max_length=255)),
                ('descripcion', models.CharField(max_length=255)),
            ],
            options={
                'verbose_name': 'Tipo de carga',
                'verbose_name_plural': 'Tipos de carga',
                'ordering': ['nombre'],
            },
        ),
        migrations.CreateModel(
            name='TipoOperacion',
            fields=[
                ('codigo', models.IntegerField(primary_key=True, serialize=False)),
                ('nombre', models.CharField(max_length=255)),
                ('nomber_a_consignar', models.CharField(max_length=255)),
                ('ind_ingreso', models.BooleanField()),
                ('ind_salida', models.BooleanField()),
                ('operacion', models.CharField(max_length=255)),
            ],
            options={
                'verbose_name': 'Tipo de operación',
                'verbose_name_plural': 'Tipos de operaciones',
                'ordering': ['nombre'],
            },
        ),
        migrations.CreateModel(
            name='UnidadMedida',
            fields=[
                ('codigo', models.IntegerField(primary_key=True, serialize=False)),
                ('nombre', models.CharField(max_length=255)),
                ('unidad', models.CharField(max_length=255)),
            ],
            options={
                'verbose_name': 'Unidad de medida',
                'verbose_name_plural': 'Unidades de medida',
                'ordering': ['nombre'],
            },
        ),
        migrations.CreateModel(
            name='ViaTransporte',
            fields=[
                ('codigo', models.IntegerField(primary_key=True, serialize=False)),
                ('nombre', models.CharField(max_length=255)),
            ],
            options={
                'verbose_name': 'Via de transporte',
                'verbose_name_plural': 'Vias de transporte',
                'ordering': ['nombre'],
            },
        ),
        migrations.CreateModel(
            name='HistoricalClausula',
            fields=[
                ('codigo', models.IntegerField(db_index=True)),
                ('nombre', models.CharField(max_length=255)),
                ('sigla', models.CharField(max_length=255)),
                ('history_id', models.AutoField(primary_key=True, serialize=False)),
                ('history_date', models.DateTimeField(db_index=True)),
                ('history_change_reason', models.CharField(max_length=100, null=True)),
                ('history_type', models.CharField(choices=[('+', 'Created'), ('~', 'Changed'), ('-', 'Deleted')], max_length=1)),
                ('history_user', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='+', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name': 'historical Clausula',
                'verbose_name_plural': 'historical Clausulas',
                'ordering': ('-history_date', '-history_id'),
                'get_latest_by': ('history_date', 'history_id'),
            },
            bases=(simple_history.models.HistoricalChanges, models.Model),
        ),
        migrations.CreateModel(
            name='HistoricalModalidadVenta',
            fields=[
                ('codigo', models.IntegerField(db_index=True)),
                ('nombre', models.CharField(max_length=255)),
                ('descripcion', models.CharField(max_length=255)),
                ('history_id', models.AutoField(primary_key=True, serialize=False)),
                ('history_date', models.DateTimeField(db_index=True)),
                ('history_change_reason', models.CharField(max_length=100, null=True)),
                ('history_type', models.CharField(choices=[('+', 'Created'), ('~', 'Changed'), ('-', 'Deleted')], max_length=1)),
                ('history_user', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='+', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name': 'historical Modalidad de venta',
                'verbose_name_plural': 'historical Modalidades de venta',
                'ordering': ('-history_date', '-history_id'),
                'get_latest_by': ('history_date', 'history_id'),
            },
            bases=(simple_history.models.HistoricalChanges, models.Model),
        ),
        migrations.CreateModel(
            name='HistoricalPais',
            fields=[
                ('codigo', models.IntegerField(db_index=True)),
                ('nombre', models.CharField(max_length=255)),
                ('continente', models.CharField(max_length=255)),
                ('history_id', models.AutoField(primary_key=True, serialize=False)),
                ('history_date', models.DateTimeField(db_index=True)),
                ('history_change_reason', models.CharField(max_length=100, null=True)),
                ('history_type', models.CharField(choices=[('+', 'Created'), ('~', 'Changed'), ('-', 'Deleted')], max_length=1)),
                ('history_user', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='+', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name': 'historical Pais',
                'verbose_name_plural': 'historical Paises',
                'ordering': ('-history_date', '-history_id'),
                'get_latest_by': ('history_date', 'history_id'),
            },
            bases=(simple_history.models.HistoricalChanges, models.Model),
        ),
        migrations.CreateModel(
            name='HistoricalRegimenImportacion',
            fields=[
                ('codigo', models.IntegerField(db_index=True)),
                ('nombre', models.CharField(max_length=255)),
                ('sigla', models.CharField(max_length=255)),
                ('active', models.BooleanField()),
                ('history_id', models.AutoField(primary_key=True, serialize=False)),
                ('history_date', models.DateTimeField(db_index=True)),
                ('history_change_reason', models.CharField(max_length=100, null=True)),
                ('history_type', models.CharField(choices=[('+', 'Created'), ('~', 'Changed'), ('-', 'Deleted')], max_length=1)),
                ('history_user', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='+', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name': 'historical Regimen de importación',
                'verbose_name_plural': 'historical Regimenes de importación',
                'ordering': ('-history_date', '-history_id'),
                'get_latest_by': ('history_date', 'history_id'),
            },
            bases=(simple_history.models.HistoricalChanges, models.Model),
        ),
        migrations.CreateModel(
            name='HistoricalRegion',
            fields=[
                ('codigo', models.IntegerField(db_index=True)),
                ('nombre', models.CharField(max_length=255)),
                ('history_id', models.AutoField(primary_key=True, serialize=False)),
                ('history_date', models.DateTimeField(db_index=True)),
                ('history_change_reason', models.CharField(max_length=100, null=True)),
                ('history_type', models.CharField(choices=[('+', 'Created'), ('~', 'Changed'), ('-', 'Deleted')], max_length=1)),
                ('history_user', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='+', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name': 'historical Región',
                'verbose_name_plural': 'historical Regiones',
                'ordering': ('-history_date', '-history_id'),
                'get_latest_by': ('history_date', 'history_id'),
            },
            bases=(simple_history.models.HistoricalChanges, models.Model),
        ),
        migrations.CreateModel(
            name='HistoricalTipoCarga',
            fields=[
                ('id', models.BigIntegerField(auto_created=True, blank=True, db_index=True, verbose_name='ID')),
                ('codigo', models.CharField(max_length=50)),
                ('nombre', models.CharField(max_length=255)),
                ('descripcion', models.CharField(max_length=255)),
                ('history_id', models.AutoField(primary_key=True, serialize=False)),
                ('history_date', models.DateTimeField(db_index=True)),
                ('history_change_reason', models.CharField(max_length=100, null=True)),
                ('history_type', models.CharField(choices=[('+', 'Created'), ('~', 'Changed'), ('-', 'Deleted')], max_length=1)),
                ('history_user', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='+', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name': 'historical Tipo de carga',
                'verbose_name_plural': 'historical Tipos de carga',
                'ordering': ('-history_date', '-history_id'),
                'get_latest_by': ('history_date', 'history_id'),
            },
            bases=(simple_history.models.HistoricalChanges, models.Model),
        ),
        migrations.CreateModel(
            name='HistoricalTipoOperacion',
            fields=[
                ('codigo', models.IntegerField(db_index=True)),
                ('nombre', models.CharField(max_length=255)),
                ('nomber_a_consignar', models.CharField(max_length=255)),
                ('ind_ingreso', models.BooleanField()),
                ('ind_salida', models.BooleanField()),
                ('operacion', models.CharField(max_length=255)),
                ('history_id', models.AutoField(primary_key=True, serialize=False)),
                ('history_date', models.DateTimeField(db_index=True)),
                ('history_change_reason', models.CharField(max_length=100, null=True)),
                ('history_type', models.CharField(choices=[('+', 'Created'), ('~', 'Changed'), ('-', 'Deleted')], max_length=1)),
                ('history_user', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='+', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name': 'historical Tipo de operación',
                'verbose_name_plural': 'historical Tipos de operaciones',
                'ordering': ('-history_date', '-history_id'),
                'get_latest_by': ('history_date', 'history_id'),
            },
            bases=(simple_history.models.HistoricalChanges, models.Model),
        ),
        migrations.CreateModel(
            name='HistoricalUnidadMedida',
            fields=[
                ('codigo', models.IntegerField(db_index=True)),
                ('nombre', models.CharField(max_length=255)),
                ('unidad', models.CharField(max_length=255)),
                ('history_id', models.AutoField(primary_key=True, serialize=False)),
                ('history_date', models.DateTimeField(db_index=True)),
                ('history_change_reason', models.CharField(max_length=100, null=True)),
                ('history_type', models.CharField(choices=[('+', 'Created'), ('~', 'Changed'), ('-', 'Deleted')], max_length=1)),
                ('history_user', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='+', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name': 'historical Unidad de medida',
                'verbose_name_plural': 'historical Unidades de medida',
                'ordering': ('-history_date', '-history_id'),
                'get_latest_by': ('history_date', 'history_id'),
            },
            bases=(simple_history.models.HistoricalChanges, models.Model),
        ),
        migrations.CreateModel(
            name='HistoricalViaTransporte',
            fields=[
                ('codigo', models.IntegerField(db_index=True)),
                ('nombre', models.CharField(max_length=255)),
                ('history_id', models.AutoField(primary_key=True, serialize=False)),
                ('history_date', models.DateTimeField(db_index=True)),
                ('history_change_reason', models.CharField(max_length=100, null=True)),
                ('history_type', models.CharField(choices=[('+', 'Created'), ('~', 'Changed'), ('-', 'Deleted')], max_length=1)),
                ('history_user', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='+', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name': 'historical Via de transporte',
                'verbose_name_plural': 'historical Vias de transporte',
                'ordering': ('-history_date', '-history_id'),
                'get_latest_by': ('history_date', 'history_id'),
            },
            bases=(simple_history.models.HistoricalChanges, models.Model),
        ),
        migrations.CreateModel(
            name='HistoricalTipoMoneda',
            fields=[
                ('codigo', models.IntegerField(db_index=True)),
                ('nombre', models.CharField(max_length=255)),
                ('history_id', models.AutoField(primary_key=True, serialize=False)),
                ('history_date', models.DateTimeField(db_index=True)),
                ('history_change_reason', models.CharField(max_length=100, null=True)),
                ('history_type', models.CharField(choices=[('+', 'Created'), ('~', 'Changed'), ('-', 'Deleted')], max_length=1)),
                ('history_user', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='+', to=settings.AUTH_USER_MODEL)),
                ('pais', models.ForeignKey(blank=True, db_constraint=False, null=True, on_delete=django.db.models.deletion.DO_NOTHING, related_name='+', to='collection_manager.pais')),
            ],
            options={
                'verbose_name': 'historical Tipo de moneda',
                'verbose_name_plural': 'historical Tipos de moneda',
                'ordering': ('-history_date', '-history_id'),
                'get_latest_by': ('history_date', 'history_id'),
            },
            bases=(simple_history.models.HistoricalChanges, models.Model),
        ),
        migrations.CreateModel(
            name='HistoricalPuerto',
            fields=[
                ('codigo', models.IntegerField(db_index=True)),
                ('nombre', models.CharField(max_length=255)),
                ('tipo', models.CharField(max_length=255)),
                ('latitud', models.FloatField()),
                ('longitud', models.FloatField()),
                ('zona_geografica', models.CharField(max_length=255)),
                ('history_id', models.AutoField(primary_key=True, serialize=False)),
                ('history_date', models.DateTimeField(db_index=True)),
                ('history_change_reason', models.CharField(max_length=100, null=True)),
                ('history_type', models.CharField(choices=[('+', 'Created'), ('~', 'Changed'), ('-', 'Deleted')], max_length=1)),
                ('history_user', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='+', to=settings.AUTH_USER_MODEL)),
                ('pais', models.ForeignKey(blank=True, db_constraint=False, null=True, on_delete=django.db.models.deletion.DO_NOTHING, related_name='+', to='collection_manager.pais')),
            ],
            options={
                'verbose_name': 'historical Puerto',
                'verbose_name_plural': 'historical Puertos',
                'ordering': ('-history_date', '-history_id'),
                'get_latest_by': ('history_date', 'history_id'),
            },
            bases=(simple_history.models.HistoricalChanges, models.Model),
        ),
        migrations.CreateModel(
            name='Puerto',
            fields=[
                ('codigo', models.IntegerField(primary_key=True, serialize=False)),
                ('nombre', models.CharField(max_length=255)),
                ('tipo', models.CharField(max_length=255)),
                ('latitud', models.FloatField()),
                ('longitud', models.FloatField()),
                ('zona_geografica', models.CharField(max_length=255)),
                ('pais', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='collection_manager.pais')),
            ],
            options={
                'verbose_name': 'Puerto',
                'verbose_name_plural': 'Puertos',
                'ordering': ['nombre'],
            },
        ),
        migrations.CreateModel(
            name='TipoMoneda',
            fields=[
                ('codigo', models.IntegerField(primary_key=True, serialize=False)),
                ('nombre', models.CharField(max_length=255)),
                ('pais', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='collection_manager.pais')),
            ],
            options={
                'verbose_name': 'Tipo de moneda',
                'verbose_name_plural': 'Tipos de moneda',
                'ordering': ['nombre'],
            },
        ),
    ]
