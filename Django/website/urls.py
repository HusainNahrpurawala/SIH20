from django.urls import path
from . import views

app_name = 'site'

urlpatterns = [
    path('', views.Home.as_view(), name = 'Home'),
    path('Security/', views.Security.as_view(), name = 'Security'),
    #path('allLogs/', views.allLogs.as_view(), name = 'AllLogs'),
    #path('userLog/', views.UserLog.as_view(), name = 'UserLog'),
    #path('SignUp/', views.SignUp.as_view(), name = 'SignUp'),
    #path('Guest/', views.Guest.as_view(), name = 'Guest'),
    path('logout/', views.Logout, name = 'logout'),
    
]