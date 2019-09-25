# -*- coding: utf-8 -*-
from __future__ import print_function
from flask import Flask, request, jsonify, render_template, redirect
from grobid_parser_pdf import grobid_client
from xml_parser import xml_parsing
import logging
import time
import requests, os, threading
app = Flask(__name__)
PORT = 8081
print('done!\nlaunching the server.')

UPLOAD_FOLDER = '/home/ubuntu/bert_hr/data_pdf'
ALLOWED_EXTENSIONS = set(['pdf',])

process_file = '/home/ubuntu/bert_hr/grobid/processed_file.txt'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

service = 'processFulltextDocument'
generateIDs = 1
consolidate_citations = 1
consolidate_header = 1
teiCoordinates = 1
force = False

class AsyncGrobitTask(threading.Thread):

    def __init__(self, filename):
        self.filename = filename
        super().__init__(daemon=True)

    def run(self):
        g_client = grobid_client()
        isfile_processed = g_client.process_pdf("../data_pdf/%s"%self.filename, "../data_xml/", service, generateIDs, consolidate_header, consolidate_citations, force, teiCoordinates)
        f = open(process_file, 'w+')
        f.write(f.read().replace("\n%s\n"%self.filename, ""))
        f.close()
        print("GROBID DONE !!")


@app.route('/gorbid_processing', methods=['GET'])
def gorbid_processing():
    filename = request.args['f']
    if os.path.isfile("../data_xml/%s"%(filename.split(".")[0] + ".tei.xml",)):
        print("File Exists")
        return "2"
    f = open(process_file, 'a+')
    f.seek(0)
    f_data = f.read()
    print(f_data)
    if filename in f_data:
        print("Already processing")
        f.close()
        return "0"
    print("New Thread Started !")
    f.write("%s\n"%filename)
    f.close()
    async_task = AsyncGrobitTask(filename=filename)
    async_task.start()
    return "1"

@app.route('/parseXML', methods=['GET'])
def parseXML():
    filename = request.args['f'].split(".")[0] + ".tei.xml"
    if os.path.isfile("../data_input/%s"%(filename.split(".")[0] + ".tei.csv",)):
        print("File Exists")
        return redirect('/bert_process?f=%s'%filename)
    is_processed = xml_parsing(filename)
    if is_processed:
        return redirect('/bert_process?f=%s'%filename)
    else:
        return "Error Processing XML"


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=PORT)
