# -*- coding: utf-8 -*-

a=u"あ"
#print a.encode("utf8")
print type(a)
print repr(a)
print type(a.encode("utf8"))
print u'\u3042'


#a = u'\u3042'
#a = '\\'
#print ord(a)
#print unichr(ord(a) + 1)
print unichr(92)
