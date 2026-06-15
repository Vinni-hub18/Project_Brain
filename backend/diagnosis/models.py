from django.db import models
from django.core.validators import FileExtensionValidator


class BrainScan(models.Model):
    PREDICTION_CHOICES = [
        ("tumor_suspected", "Tumor Suspected"),
        ("no_tumor", "No Tumor"),
        ("invalid_input", "Invalid Input"),
        ("uncertain", "Uncertain"),
        ("pending", "Pending"),
    ]

    patient_name = models.CharField(max_length=100, blank=True, null=True)
    patient_id = models.CharField(max_length=50, blank=True, null=True)

    scan_file = models.FileField(
        upload_to="scans/original/",
        validators=[FileExtensionValidator(allowed_extensions=["jpg", "jpeg", "png", "dcm"])],
        blank=True,
        null=True,
    )
    heatmap_image = models.ImageField(upload_to="scans/heatmaps/", blank=True, null=True)
    segmentation_mask = models.ImageField(upload_to="scans/masks/", blank=True, null=True)
    report_file = models.FileField(upload_to="reports/", blank=True, null=True)

    prediction = models.CharField(
        max_length=30,
        choices=PREDICTION_CHOICES,
        default="pending"
    )
    confidence_score = models.FloatField(blank=True, null=True)
    insight_text = models.TextField(default="Awaiting analysis.")
    validation_message = models.TextField(blank=True, null=True)

    mask_area_pixels = models.PositiveIntegerField(blank=True, null=True)
    mask_area_ratio = models.FloatField(blank=True, null=True)
    mean_mask_probability = models.FloatField(blank=True, null=True)
    max_probability = models.FloatField(blank=True, null=True)

    is_processed = models.BooleanField(default=False)

    uploaded_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Scan {self.id} - {self.prediction}"