from  __future__ import print_function
from xml.dom.minidom import parse
import xml.dom.minidom
from collections import defaultdict
import pandas as pd
import re
import os
import xml.etree.ElementTree as ET
# doc = minidom.parse("data/output/20467.full.tei.xml")
# DOMTree = xml.dom.minidom.parse("data/output/20467.full.tei.xml")
# print(ET.tostring(root, encoding='utf8').decode('utf8'))



def bibliography_list(listbibl):
   # listbibl = root.findall('./')[1][1][1][0]
   bibl = {}
   bibl_list = []
   for l in listbibl:
      analytic = l[0]
      # monogr = l[1]

      bibl['title'] = analytic[0].text
      authors = analytic.findall('{http://www.tei-c.org/ns/1.0}author')
      bibl['authors'] = []
      for author in authors:
         author_name = ' '.join([a.text for a in author[0]])
         bibl['authors'].append(author_name)
      if analytic.find('{http://www.tei-c.org/ns/1.0}idno'):
         bibl['doi'] = analytic.find('{http://www.tei-c.org/ns/1.0}idno').text
      else:
         bibl['doi'] = None
      bibl_list.append(bibl.copy())
   return bibl_list


def xml_parsing(file):
   file_xml = '../data_xml/' + file
   file_csv = file[:-4] + '.csv'
   file_csv = '../data_input/' + file_csv
   if os.path.isfile(file_csv):
       return True
   x = ET.parse(file_xml)
   root = x.getroot()
   # div_and_fig = root.findall('./')[1].findall('./')[0].findall('./')
   # div = root.findall('./')[1].findall('./')[0].findall('./{http://www.tei-c.org/ns/1.0}div')
   body = root.find('./{http://www.tei-c.org/ns/1.0}text')[0]
   back = root.find('./{http://www.tei-c.org/ns/1.0}text')[1]
   listbibl = back.find('.//{http://www.tei-c.org/ns/1.0}listBibl')
   div = body.findall('{http://www.tei-c.org/ns/1.0}div') 

   p = []
   for d in div:
      p += d.findall('{http://www.tei-c.org/ns/1.0}p')


   bibl_list = bibliography_list(listbibl)

   ref_final = []
   for i in p:
      ref = i.findall('{http://www.tei-c.org/ns/1.0}ref')
      print(list(i))
      try:
        txt = ''.join(i.itertext()).encode('utf-8').replace('Fig.', 'Fig').split('. ')
      except:
        txt = ''.join(i.itertext()).replace('Fig.', 'Fig').split('. ')
      target = []
      temp = defaultdict(list)
      for r in ref:
         if 'bibr' in r.attrib.values() and 'target' in r.attrib.keys():
            if int(r.attrib['target'][2:])>=0:
               target.append(int(r.attrib['target'][2:])) 
      # print('target : {}'.format(target))

      for s in txt:
         print(s)
         for t in target:
            print(t)
            temp_str = str(t+1)
            if s.startswith(temp_str+" ") or re.search(' '+temp_str+' ', s) or re.search('[\[(,]'+temp_str+'[\]),]', s):
               temp['text'] = s
               # print('txt : {}\nref :  {}\n\n'.format(s,bibl_list[t]))
               temp['ref'] = bibl_list[t]
               ref_final.append(temp.copy())

   file_csv = file[:-4] + '.csv'
   file_csv = '../data_input/' + file_csv
   df = pd.DataFrame(ref_final)
   df.to_csv(file_csv)
   return True


if __name__ == '__main__':
   file_list = os.listdir('../data_xml/')
   files = [file for file in file_list if file.endswith('.xml')]
   for file in files:
      xml_parsing(file)
