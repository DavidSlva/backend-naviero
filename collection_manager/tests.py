from rest_framework import status
from rest_framework.test import APITestCase
from django.urls import reverse


class RutaTests(APITestCase) :

    def test_bahias_list(self) :
        url = reverse('bahia-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_paises_list(self) :
        url = reverse('pais-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_puertos_list(self) :
        url = reverse('puerto-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_tipos_operacion_list(self) :
        url = reverse('tipo_operacion-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_aduanas_list(self) :
        url = reverse('aduana-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_tipos_carga_list(self) :
        url = reverse('tipo_carga-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_vias_transporte_list(self) :
        url = reverse('via_transporte-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_regimen_importacion_list(self) :
        url = reverse('regimen_importacion-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_modalidades_venta_list(self) :
        url = reverse('modalidad_venta-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_regiones_list(self) :
        url = reverse('region-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_unidades_medida_list(self) :
        url = reverse('unidad_medida-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_tipos_moneda_list(self) :
        url = reverse('tipo_moneda-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_clausulas_list(self) :
        url = reverse('clausula-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    # def test_cargar_codigos(self) :
    #     url = reverse('cargar_codigos')
    #     response = self.client.get(url)
    #     self.assertEqual(response.status_code, status.HTTP_200_OK)
    #
    # def test_cargar_aduanas(self) :
    #     url = reverse('cargar_aduanas')
    #     response = self.client.get(url)
    #     self.assertEqual(response.status_code, status.HTTP_200_OK)
    #
    # def test_cargar_bahias(self) :
    #     url = reverse('cargar_bahias')
    #     response = self.client.get(url)
    #     self.assertEqual(response.status_code, status.HTTP_200_OK)
