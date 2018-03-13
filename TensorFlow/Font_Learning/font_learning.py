# -*- coding: utf-8 -*-
'''
from fontTools.ttLib import TTFont

font = TTFont('/Library/Fonts/Arial.ttf')

print type(font)
'''

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import pandas as pd
import csv

'''
#font_path = '/Library/Fonts/Arial.ttf'
font_path = '/System/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc'
arial = ImageFont.truetype(font_path, 100, encoding='utf-8')
des_folder = './Font_Data/ヒラギノ丸ゴ ProN W4/'
'''

def draw_image(des_path, font, charater, index):
    img = Image.new('1', (28, 28), 0)
    draw = ImageDraw.Draw(img)
    draw.text( (0, 0), charater, font=font, fill='white')
    img.save(des_path+'{}.png'.format(index))


#kanji = pd.read_csv('joyo-kanji-code-u.csv')
font_path = '/Library/Fonts/Hiragino Sans GB.ttc'#Verdana.ttf'
font = ImageFont.truetype(font_path, 28, encoding= 'utf-8')
des_folder = './Font_Data/Hiragino Sans GB/'

with open('joyo-kanji-code-u.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile)
    line = reader.next()

    kanji_idx = 0
    while line:
        if not line[0].startswith('#'):# and kanji_idx < 30:
            str2unicode = unicode(line[0], 'utf-8')
            #print type(str2unicode)
            #print str2unicode.encode('utf-8')

            draw_image(des_folder, font, str2unicode, kanji_idx)
            kanji_idx += 1

        line = reader.next()
        
