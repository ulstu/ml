# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

class StudentGoal(models.Model):
        student_name = models.CharField(max_length=200)
        upload_time = models.DateTimeField('date and time published')
        x = models.DecimalField(max_digits=5, decimal_places=2)
        y = models.DecimalField(max_digits=5, decimal_places=2)

	def __str__(self):
	        return "name: {}; x: {}; y: {}; time: {}".format(self.student_name, self.x, self.y, self.upload_time)
