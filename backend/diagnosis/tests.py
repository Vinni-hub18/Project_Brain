from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework.test import APITestCase
from rest_framework import status
from .models import BrainScan


class BrainScanAPITests(APITestCase):
    def setUp(self):
        self.upload_url = reverse("upload_scan")
        self.list_url = reverse("scan_list")

    def test_upload_scan(self):
        file = SimpleUploadedFile("scan.jpg", b"fake image content", content_type="image/jpeg")
        data = {
            "patient_name": "Test Patient",
            "patient_id": "P001",
            "scan_file": file,
        }
        response = self.client.post(self.upload_url, data, format="multipart")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

    def test_scan_list(self):
        BrainScan.objects.create(patient_name="A", patient_id="1")
        response = self.client.get(self.list_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)