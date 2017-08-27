import urllib2

tmpl = "http://summerschool.simcase.ru/challenge/standings/?name={}&x={}&y={}"
surl = tmpl.format('hiber', 0, 0)
content = urllib2.urlopen(surl).read()
print content
