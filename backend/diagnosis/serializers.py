from rest_framework import serializers
from .models import BrainScan


class BrainScanSerializer(serializers.ModelSerializer):
    class Meta:
        model = BrainScan
        fields = "__all__"
        read_only_fields = (
            "prediction",
            "confidence_score",
            "insight_text",
            "validation_message",
            "mask_area_pixels",
            "mask_area_ratio",
            "mean_mask_probability",
            "max_probability",
            "heatmap_image",
            "segmentation_mask",
            "report_file",
            "is_processed",
            "uploaded_at",
            "updated_at",
        )