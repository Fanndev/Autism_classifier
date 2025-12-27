"""
Django Forms for classifier app
"""
from django import forms
from django.conf import settings


class ImageUploadForm(forms.Form):
    """Form for uploading images for classification."""
    
    model = forms.ChoiceField(
        label='Pilih Model',
        choices=settings.AVAILABLE_MODELS,
        initial='autism_cnn_model(MTCNN).h5',
        widget=forms.Select(attrs={
            'class': 'form-select',
            'id': 'model-select'
        })
    )
    
    image = forms.ImageField(
        label='Pilih Gambar',
        help_text='Upload gambar wajah (JPG atau PNG)',
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': 'image/jpeg,image/png,image/jpg',
            'id': 'image-input'
        })
    )
    
    def clean_image(self):
        """Validate uploaded image."""
        image = self.cleaned_data.get('image')
        
        if image:
            # Check file size (max 10MB)
            if image.size > 10 * 1024 * 1024:
                raise forms.ValidationError(
                    'Ukuran file terlalu besar. Maksimum 10MB.'
                )
            
            # Check content type
            content_type = image.content_type
            if content_type not in ['image/jpeg', 'image/png', 'image/jpg']:
                raise forms.ValidationError(
                    'Format file tidak didukung. Gunakan JPG atau PNG.'
                )
        
        return image
