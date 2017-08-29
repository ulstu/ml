from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.standings, name='standings'),
    url(r'^sendgoal/$', views.sendgoal, name='sendgoal'),
    url(r'^standings/$', views.standings, name='standings'),
    url(r'^student/$', views.student, name='student')
]
