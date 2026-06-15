from django.contrib import admin
from .models import BrainScan

@admin.register(BrainScan)
class BrainScanAdmin(admin.ModelAdmin):
    list_display = ("id", "patient_name", "prediction", "confidence_score", "is_processed", "uploaded_at")
    list_filter = ("prediction", "is_processed", "uploaded_at")
    search_fields = ("patient_name", "patient_id")