from django.shortcuts import render, redirect
from django.views.generic import View
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from .models import Person
from django.http import HttpResponse
from django.urls import reverse

# Create your views here.

class Home(View):

    template_name = 'website/home.html'

    def get(self, request):
        return render(request, self.template_name)

    def post(self, request):
        un = request.POST["username"]
        pw = request.POST["password"]

        u = authenticate(username = un, password = pw)
        if u is not None:
            login(request, u)
            p = Person.objects.get(user = u)
            if p.isSecurity:
                return render(request, 'website/security.html')
            else:
                return render(request, 'website/employee.html')
        else:
            return HttpResponse('Fail')

class Security(View):

    template_name = 'website/security.html'

    def get(self, request):
        u = request.user
        if u.is_authenticated:
            p = Person.objects.get(user = u)
            if p.isSecurity:
                return render(request, self.template_name)
            else:
                return HttpResponse('Fail')
        else:
            return redirect(reverse('home'))

class SignUp(View):
    
    template_name = 'website/signup.html'

    def get(self, request):
        u = request.user
        if u.is_authenticated:
            p = Person.objects.get(user = u)
            if p.isSecurity:
                return render(request, self.template_name)
            else:
                return HttpResponse('Fail')
        else:
            return redirect(reverse('home'))

    def post(self, request):
        return render(request, self.template_name)

def Logout(request):
    logout(request)
    return redirect(reverse('home'))