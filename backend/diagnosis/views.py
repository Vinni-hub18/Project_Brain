import io
import os
import requests

from django.conf import settings
from django.core.files import File
from django.core.files.base import ContentFile
from django.db import transaction
from django.http import FileResponse

from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response

from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from .models import BrainScan
from .serializers import BrainScanSerializer
from .services.predict import predict_mask


def build_pdf_report(scan):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 50
    p.setFont("Helvetica-Bold", 18)
    p.drawString(50, y, "Brain Tumor Analysis Report")

    y -= 40
    p.setFont("Helvetica", 12)
    p.drawString(50, y, f"Scan ID: {scan.id}")
    y -= 20
    p.drawString(50, y, f"Patient Name: {scan.patient_name or 'N/A'}")
    y -= 20
    p.drawString(50, y, f"Patient ID: {scan.patient_id or 'N/A'}")
    y -= 20
    p.drawString(50, y, f"Prediction: {scan.prediction}")
    y -= 20
    p.drawString(
        50, y,
        f"Confidence Score: {scan.confidence_score if scan.confidence_score is not None else 'N/A'}%"
    )
    y -= 20
    p.drawString(
        50, y,
        f"Validation: {scan.validation_message or 'Passed'}"
    )
    y -= 20
    p.drawString(
        50, y,
        f"Uploaded At: {scan.uploaded_at.strftime('%Y-%m-%d %H:%M:%S') if scan.uploaded_at else 'N/A'}"
    )

    y -= 35
    p.setFont("Helvetica-Bold", 13)
    p.drawString(50, y, "Insight")
    y -= 20
    p.setFont("Helvetica", 11)

    insight = scan.insight_text or "N/A"
    for i in range(0, len(insight), 90):
        p.drawString(50, y, insight[i:i + 90])
        y -= 18

    image_y = y - 20

    if scan.heatmap_image and os.path.exists(scan.heatmap_image.path):
        p.setFont("Helvetica-Bold", 13)
        p.drawString(50, image_y, "Heatmap Image")
        p.drawImage(
            ImageReader(scan.heatmap_image.path),
            50,
            image_y - 160,
            width=200,
            height=150,
            preserveAspectRatio=True,
            mask="auto"
        )

    if scan.segmentation_mask and os.path.exists(scan.segmentation_mask.path):
        p.setFont("Helvetica-Bold", 13)
        p.drawString(300, image_y, "Segmentation Mask")
        p.drawImage(
            ImageReader(scan.segmentation_mask.path),
            300,
            image_y - 160,
            width=200,
            height=150,
            preserveAspectRatio=True,
            mask="auto"
        )

    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer


def apply_prediction_to_scan(scan):
    result = predict_mask(scan.scan_file.path, scan.id)

    scan.prediction = result["prediction"]
    scan.confidence_score = result["confidence_score"]
    scan.insight_text = result["insight_text"]
    scan.validation_message = result["validation_message"]
    scan.mask_area_pixels = result["mask_area_pixels"]
    scan.mask_area_ratio = result["mask_area_ratio"]
    scan.max_probability = result["max_probability"]
    scan.mean_mask_probability = result["mean_mask_probability"]
    scan.is_processed = True

    if result["mask_path"] and os.path.exists(result["mask_path"]):
        with open(result["mask_path"], "rb") as f:
            scan.segmentation_mask.save(os.path.basename(result["mask_path"]), File(f), save=False)

    if result["overlay_path"] and os.path.exists(result["overlay_path"]):
        with open(result["overlay_path"], "rb") as f:
            scan.heatmap_image.save(os.path.basename(result["overlay_path"]), File(f), save=False)

    scan.save()

    pdf_buffer = build_pdf_report(scan)
    pdf_name = f"report_scan_{scan.id}.pdf"
    scan.report_file.save(pdf_name, ContentFile(pdf_buffer.getvalue()), save=True)

    return result


@api_view(["GET"])
def health_check(request):
    return Response({"status": "ok", "message": "Diagnosis API working"})


@api_view(["GET"])
def scan_list(request):
    scans = BrainScan.objects.all().order_by("-uploaded_at")
    serializer = BrainScanSerializer(scans, many=True, context={"request": request})
    return Response(serializer.data)


@api_view(["GET"])
def scan_detail(request, pk):
    try:
        scan = BrainScan.objects.get(pk=pk)
    except BrainScan.DoesNotExist:
        return Response({"error": "Scan not found"}, status=status.HTTP_404_NOT_FOUND)

    serializer = BrainScanSerializer(scan, context={"request": request})
    return Response(serializer.data)


@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def upload_scan(request):
    serializer = BrainScanSerializer(data=request.data, context={"request": request})
    if serializer.is_valid():
        scan = serializer.save()
        return Response(
            BrainScanSerializer(scan, context={"request": request}).data,
            status=status.HTTP_201_CREATED
        )
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def upload_multiple_scans(request):
    files = request.FILES.getlist("scan_file")

    if not files:
        return Response({"error": "No files uploaded"}, status=status.HTTP_400_BAD_REQUEST)

    created_scans = []

    for f in files:
        scan = BrainScan.objects.create(
            patient_name=request.data.get("patient_name"),
            patient_id=request.data.get("patient_id"),
            scan_file=f
        )
        created_scans.append(scan)

    serializer = BrainScanSerializer(created_scans, many=True, context={"request": request})
    return Response(serializer.data, status=status.HTTP_201_CREATED)


@api_view(["POST"])
def analyze_scan(request, pk):
    try:
        scan = BrainScan.objects.get(pk=pk)
    except BrainScan.DoesNotExist:
        return Response({"error": "Scan not found"}, status=status.HTTP_404_NOT_FOUND)

    if not scan.scan_file:
        return Response({"error": "No scan file uploaded"}, status=status.HTTP_400_BAD_REQUEST)

    apply_prediction_to_scan(scan)

    return Response(
        BrainScanSerializer(scan, context={"request": request}).data,
        status=status.HTTP_200_OK
    )


@api_view(["GET"])
def download_report(request, pk):
    try:
        scan = BrainScan.objects.get(pk=pk)
    except BrainScan.DoesNotExist:
        return Response({"error": "Scan not found"}, status=status.HTTP_404_NOT_FOUND)

    if not scan.report_file:
        return Response({"error": "Report not generated yet"}, status=status.HTTP_404_NOT_FOUND)

    return FileResponse(
        scan.report_file.open("rb"),
        as_attachment=True,
        filename=os.path.basename(scan.report_file.name)
    )


@api_view(["POST"])
def analyze_all_scans(request):
    scans = BrainScan.objects.filter(is_processed=False).order_by("uploaded_at")

    if not scans.exists():
        return Response({"message": "No unprocessed scans found"}, status=status.HTTP_404_NOT_FOUND)

    results = []

    with transaction.atomic():
        for scan in scans:
            if not scan.scan_file:
                continue

            apply_prediction_to_scan(scan)

            results.append({
                "id": scan.id,
                "prediction": scan.prediction,
                "confidence_score": scan.confidence_score,
                "report_file": request.build_absolute_uri(scan.report_file.url) if scan.report_file else None,
                "validation_message": scan.validation_message,
            })

    return Response(
        {"message": "Batch analysis complete", "results": results},
        status=status.HTTP_200_OK
    )


@api_view(["GET"])
def pending_scans(request):
    scans = BrainScan.objects.filter(is_processed=False).order_by("-uploaded_at")
    serializer = BrainScanSerializer(scans, many=True, context={"request": request})
    return Response(serializer.data)


@api_view(["POST"])
def chatbot_reply(request):
    user_message = request.data.get("message", "").strip()

    if not user_message:
        return Response({"error": "Message is required"}, status=status.HTTP_400_BAD_REQUEST)

        return Response(
            {"error": "OpenRouter API key not configured"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
            },
            json={
                "model": "openai/gpt-4.1-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful medical assistant for a brain tumor diagnosis app. Do not give a final diagnosis. Suggest users consult a doctor for medical decisions."
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],
                "max_tokens": 300,
                "temperature": 0.7
            },
            timeout=60
        )

        data = response.json()

        if response.status_code != 200:
            return Response(
                {
                    "error": "OpenRouter request failed",
                    "details": data
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        reply = data["choices"][0]["message"]["content"]
        return Response({"reply": reply}, status=status.HTTP_200_OK)

    except requests.RequestException as e:
        return Response(
            {"error": "Failed to connect to OpenRouter", "details": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    except Exception as e:
        return Response(
            {"error": "Unexpected server error", "details": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )