from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response

from api.models import BarcosRecalando
from api.serializer import BarcosRecalandoSerializer
from api.services.naves_recalando import descargar_barcos

# Create your views here.
@api_view(['GET'])
def barcos_recalando(request):
    naves = descargar_barcos()
    # barcos = BarcosRecalando.objects.all()
    # serializer = BarcosRecalandoSerializer(barcos, many=True)
    return Response(naves)
