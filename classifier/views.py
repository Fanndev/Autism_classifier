"""
Views for classifier app - Handles HTTP requests and responses
"""
import os
import uuid
from django.shortcuts import render
from django.http import JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from .forms import ImageUploadForm
from .application.services import get_classifier_service


def home(request):
    """Home page view."""
    form = ImageUploadForm()
    return render(request, 'classifier/home.html', {'form': form})


def about(request):
    """About page view."""
    return render(request, 'classifier/about.html')


def classify(request):
    """
    Handle image classification request.
    Accepts POST with image file.
    """
    if request.method != 'POST':
        return JsonResponse({
            'success': False,
            'error': 'Method not allowed'
        }, status=405)
    
    form = ImageUploadForm(request.POST, request.FILES)
    
    if not form.is_valid():
        errors = form.errors.as_json()
        return JsonResponse({
            'success': False,
            'error': 'Validasi gagal',
            'details': errors
        }, status=400)
    
    # Save uploaded file temporarily
    image = form.cleaned_data['image']
    file_ext = os.path.splitext(image.name)[1]
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    
    # Ensure media directory exists
    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, unique_filename)
    
    # Save file
    with open(file_path, 'wb+') as destination:
        for chunk in image.chunks():
            destination.write(chunk)
    
    try:
        # Get classifier service and classify image
        service = get_classifier_service()
        result = service.classify_image(file_path)
        
        if not result.success:
            return JsonResponse({
                'success': False,
                'error': result.error_message
            })
        
        # Prepare response
        prediction = result.prediction
        response_data = {
            'success': True,
            'prediction': {
                'label': str(prediction.label),
                'confidence': float(prediction.confidence),
                'confidence_percentage': str(prediction.confidence_percentage),
                'is_autistic': bool(prediction.is_autistic),
                'message': str(prediction.result_message)
            }
        }
        
        return JsonResponse(response_data)
        
    finally:
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)


class ClassifyView(View):
    """Class-based view for classification (alternative)."""
    
    def get(self, request):
        """Render upload form."""
        form = ImageUploadForm()
        return render(request, 'classifier/home.html', {'form': form})
    
    def post(self, request):
        """Handle classification request."""
        return classify(request)
