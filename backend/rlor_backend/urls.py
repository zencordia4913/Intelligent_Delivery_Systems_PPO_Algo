from django.urls import path
from . import views  # ✅ import views directly

urlpatterns = [
    path("api/convert_addresses/", views.convert_addresses, name="convert_addresses"),
]
