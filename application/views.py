from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.decorators import api_view

from application.services.naves_recalando import descargar_barcos


# Create your views here.
@api_view(['GET'])
def index(request):
    return HttpResponse("Hola!")

@api_view(['GET'])
def obtener_recalando(request):
    naves = descargar_barcos()
    return HttpResponse(naves)
