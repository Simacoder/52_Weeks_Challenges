from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt  
from .forms import IrisForm
from .services import predict_iris
import json
from django.http import Http404


def home(request):
    return render(request, "predictor/predict_form.html", {"form": IrisForm()})


@require_http_methods(["POST"])
def predict_view(request):
    form = IrisForm(request.POST)
    if not form.is_valid():
        return render(request, "predictor/predict_form.html", {"form": form})
    data = form.cleaned_data
    features = [
        data["sepal_length"],
        data["sepal_width"],
        data["petal_length"],
        data["petal_width"],
    ]
    result = predict_iris(features)
    return render(
        request,
        "predictor/predict_form.html",
        {"form": IrisForm(), "result": result, "submitted": True},
    )


@csrf_exempt  # <-- add this line
@require_http_methods(["POST"])
def predict_api(request):
    # Accept JSON only (optional but recommended)
    if request.META.get("CONTENT_TYPE", "").startswith("application/json"):
        try:
            payload = json.loads(request.body or "{}")
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON."}, status=400)
    else:
        # fall back to form-encoded if you want to keep supporting it:
        payload = request.POST.dict()

    required = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    missing = [k for k in required if k not in payload]
    if missing:
        return JsonResponse({"error": f"Missing: {', '.join(missing)}"}, status=400)

    try:
        features = [float(payload[k]) for k in required]
    except ValueError:
        return JsonResponse({"error": "All features must be numeric."}, status=400)

    return JsonResponse(predict_iris(features))

import json
from pathlib import Path

def analysis(request):
    stats_path = Path(__file__).resolve().parent / "model" / "iris_stats.json"
    if not stats_path.exists():
        raise Http404("Analysis data not found. Please run train.py first.")

    with open(stats_path) as f:
        stats = json.load(f)

    # Convert class_counts from { "0": count, "1": count } to a list of (name, count)
    class_counts = [
        (stats["target_names"][int(idx)], count)
        for idx, count in stats["class_counts"].items()
    ]

    context = {
        "feature_stats": stats["feature_stats"],
        "class_counts": class_counts,
        "target_names": stats["target_names"],
    }
    return render(request, "predictor/analysis.html", context)