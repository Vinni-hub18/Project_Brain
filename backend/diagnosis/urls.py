from django.urls import path
from .views import (
    health_check,
    scan_list,
    upload_scan,
    upload_multiple_scans,
    scan_detail,
    analyze_scan,
    download_report,
    analyze_all_scans,
    pending_scans,
    chatbot_reply,
)

urlpatterns = [
    path("health/", health_check, name="health_check"),
    path("scans/", scan_list, name="scan_list"),
    path("scans/upload/", upload_scan, name="upload_scan"),
    path("scans/upload-multiple/", upload_multiple_scans, name="upload_multiple_scans"),
    path("scans/<int:pk>/", scan_detail, name="scan_detail"),
    path("scans/<int:pk>/analyze/", analyze_scan, name="analyze_scan"),
    path("scans/<int:pk>/report/", download_report, name="download_report"),
    path("scans/analyze-all/", analyze_all_scans, name="analyze_all_scans"),
    path("scans/pending/", pending_scans, name="pending_scans"),
    path("chat/", chatbot_reply, name="chatbot_reply"),
]