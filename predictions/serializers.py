from rest_framework import serializers


class PredictInputSerializer(serializers.Serializer):
    teacher_id = serializers.IntegerField()
    role = serializers.CharField(max_length=50)
    department = serializers.CharField(max_length=200)
    experience_years = serializers.IntegerField(min_value=0)
    year = serializers.IntegerField(required=False)

    block_1 = serializers.FloatField(required=False, default=0)
    block_2 = serializers.FloatField(required=False, default=0)
    block_3 = serializers.FloatField(required=False, default=0)
    block_4 = serializers.FloatField(required=False, default=0)


class FinalizeRecordSerializer(serializers.Serializer):
    teacher_id = serializers.IntegerField()
    full_name = serializers.CharField(max_length=200, required=False, allow_blank=True)
    department = serializers.CharField(max_length=200)
    role = serializers.CharField(max_length=50)
    experience_years = serializers.IntegerField(min_value=0, required=False, default=1)

    block_1 = serializers.FloatField(required=False, default=0)
    block_2 = serializers.FloatField(required=False, default=0)
    block_3 = serializers.FloatField(required=False, default=0)
    block_4 = serializers.FloatField(required=False, default=0)

    total_kpi = serializers.FloatField()


class FinalizeInputSerializer(serializers.Serializer):
    year = serializers.IntegerField()
    records = FinalizeRecordSerializer(many=True)


class SnapshotRebuildInputSerializer(serializers.Serializer):
    base_year = serializers.IntegerField(required=False)
    target_year = serializers.IntegerField(required=False)


class SnapshotQuerySerializer(serializers.Serializer):
    year = serializers.IntegerField(required=True)
    department = serializers.CharField(max_length=200, required=False, allow_blank=True)
    role = serializers.CharField(max_length=50, required=False, allow_blank=True)
    teacher_id = serializers.IntegerField(required=False)
