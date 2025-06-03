from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('rlor_backend.urls')),  # ✅ Include your app's URLs
]
