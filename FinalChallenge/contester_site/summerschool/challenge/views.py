# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from challenge.models import StudentGoal
from django.http import HttpResponse
import datetime
import math
import unicodedata
import sys
import traceback
import os.path

# Radius around a real object, in which the student's answer is counted
R = 0.90
total_error = 0


class figures:
    ox = 0
    oy = 0
    name = ""

class Result:
	pos = 0
	student_name = ""
	error = 0.0
	real_object_number = 0
	answer_object_number = 0
	correct_detections = 0
	false_detections = 0
	false_negative = 0
	redundant_detections = 0

def get_answers(answers):
    answer_figures_list = []
    global line, word, figure, new_word, i
    for line in answers:
        figure = figures()
        figure.ox, figure.oy = float(line.x), float(line.y)
        new_word = ""
        answer_figures_list.append(figure)

    return answer_figures_list


def get_true(file_true):
    true_figures_list = []
    global line, word, figure, new_word, i
    for line in file_true:
        word = []
        word = line.split(" ")
        figure = figures()
        figure.ox, figure.oy = float(word[0]), float(word[1])
        new_word = ""

        true_figures_list.append(figure)

    return true_figures_list


def get_dist(p1, p2):
    diff_ox = p1.ox - p2.ox
    diff_oy = p1.oy - p2.oy
    distance = math.hypot(diff_ox, diff_oy)
    return distance


def check_radius(center, points):
    for point in points:
        distance = get_dist(center, point)
        if distance <= R:
            return False

    return True


# actions
def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")

def sendgoal(request):
	try:
		name = request.GET['name']
		rx = request.GET['x']
		ry = request.GET['y']
		g = StudentGoal(student_name = name, upload_time = datetime.datetime.now(), x = rx, y = ry)
		#g.save()
		output = "ok"
		return HttpResponse(output)
	except:
		return HttpResponseServerError(sys.exc_info()[:2])

def standings(request):
	#answers = StudentGoal.objects.all()
	students = StudentGoal.objects.order_by().values_list('student_name').distinct()

	log = []
	log.append(len(students))
	results = []
	try:
		for s in students:
			student = s[0]
			
			# load files with real coordinate and answer by student
			base_path = '/home/k.svyatov/sites/summerschool/answers/'
			answers = StudentGoal.objects.filter(student_name=student)
			log.append('Student name: {}'.format(student))
			try:
				fname = "{}{}.txt".format(base_path, str(student))
				if os.path.isfile(fname):
					file_true = open(fname)
				else:
					log.append('no file name {}'.format(fname))
					continue
			except:
				#file_true = ['0 0 red_cube']
				exc_type, exc_value, exc_traceback = sys.exc_info()
				lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
				log.append(''.join('!! ' + line for line in lines))
				continue

			# parse file and get points info (X, Y, type)
			answer_figures_list = get_answers(answers)
			log.append(len(answer_figures_list))
			true_figures_list = get_true(file_true)

			# эти метрики описаны в конце файла
			false_negative = 0
			redundant_detections = 0
			correct_detections = 0
			false_positive = answer_figures_list[:]

			total_error = 0.0
			# calc error
			for true_figures in true_figures_list:
			    error = 0.0
			    radius_empty = check_radius(true_figures, answer_figures_list)
			    if radius_empty:
			        # если радиус пустой, то все просто
			        false_negative += 1
			        error += 1.5
			    else:
			        # если в радиусе реального объекта есть обнаруженные объекты,
			        # то есть несколько сценариев в зависимости от количества объектов
			        # и их типа

			        finded_points = 0
			        for answer_figures in answer_figures_list:
			            distance = get_dist(true_figures, answer_figures)
			            if distance <= R:
			                if answer_figures in false_positive:
			                    false_positive.remove(answer_figures)

			                finded_points += 1
			                # в первую очередь ошибка для найденного объекта определяется
			                # расстоянием до центра реального объекта.
			                # За каждый следующий объект добавляется все меньшая ошибка.
			                error += float(distance) / (R + finded_points)

			                #if true_figures.name != answer_figures.name:
			                    # если тип определен не верно, то к ошибке добавляется 0.5,
			                    # при этом за каждый следующий объект добавляется все меньшая ошибка
			                    #error += (0.5 / finded_points)

			        redundant_detections += finded_points - 1
			        if finded_points != 0:
			            correct_detections += 1

			    total_error += error

			# оценка всех ложно найденных объектов
			# то есть вне радиусов вокруг реальных объектов
			# в зависимости от их количества
			# то есть если найдено очень большое лишние объектов, то
			# ошибка искусственно снижается, чтобы не слишком сильно сбивать
			# показатели ошибок для найденных объектов
			for i, fp in enumerate(false_positive):
			    if i <= 10:
			        total_error += 1.0
			    if i > 10 and i < 100:
			        total_error += 0.25
			    if i > 100:
			        total_error += 0.1
			r = Result()
			log.append(len(false_positive))
			r.student_name = student
			r.error = total_error
			r.real_object_number = len(true_figures_list)
			r.answer_object_number = len(answer_figures_list)
			r.correct_detections = correct_detections
			r.false_detections = len(false_positive)
			r.false_negative = false_negative
			r.redundant_detections = redundant_detections
			results.append(r)
	except Exception, ex:
		exc_type, exc_value, exc_traceback = sys.exc_info()
		lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
		log.append(''.join('!! ' + line for line in lines))
	results.sort(key=lambda x: x.correct_detections, reverse=True)
	for i, x in enumerate(results):
		x.pos = i + 1
	return render(request, 'standings.html', {'objects': results, 'log': log})

def student(request):
	student_res = StudentGoal.objects.filter(student_name=request.GET['name'])
	return render(request, 'student_sendings.html', {'sendings': student_res})
