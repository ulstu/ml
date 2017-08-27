import urllib2

def send_goal(nickname, x, y):
	tmpl = "http://summerschool.simcase.ru/challenge/standings/?name={}&x={}&y={}"
	surl = tmpl.format(nickname, x, y)
	content = urllib2.urlopen(surl).read()
	print content
