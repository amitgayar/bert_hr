# -*- coding: utf-8 -*-
from __future__ import print_function
from flask import Flask, request, jsonify, render_template, redirect
import logging
import time
import requests, os, csv, json
from berthr import *
import threading

app = Flask(__name__) 
PORT = 8080
print('done!\nlaunching the server.')

UPLOAD_FOLDER = '/home/ubuntu/bert_hr/data_pdf'
ALLOWED_EXTENSIONS = set(['pdf',])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

process_file = '/home/ubuntu/bert_hr/flask_app/bert_processed_file.txt'

class AsyncBERTTask(threading.Thread):

    def __init__(self, filename):
        self.filename = filename
	threading.Thread.__init__(self)

    def run(self):
	runBERT(self.filename)
        f = open(process_file, 'w+')
        f.write(f.read().replace("\n%s\n"%self.filename, ""))
        f.close()
        print("BERT DONE !!")


@app.route("/")
def hello():
    return render_template('index.html')

@app.route("/help")
def help():
	return "BERT  Human Resistin classification Production Server </br></br>\
			Run the '0.0.0.0:8080/runBERT?filename=filename' in the browser with filename as the name of the '.csv' file saved in data_input \
			directory and to be predicted. </br></br>\
			For example: '0.0.0.0:8080/runBERT?filename=20467.full.tei.csv'</br></br>Results would be saved in data_output directory with the same filename."

@app.route('/uploadRA', methods=['POST'])
def uploadRA():
    if request.method == 'POST':
        print(request.files)
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect('/')
        files = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if files.filename == '':
            return redirect('/')
        if files:
            filename = str(int(round(time.time() * 1000))) + ".pdf"
            files.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("Saved")
	return redirect('/gorbid_process?f=%s'%filename)

@app.route('/gorbid_process', methods=['GET'])
def gorbid_process():
    return render_template("grobid.html")

@app.route('/bert_process', methods=['GET'])
def bert_process():
    return render_template("processbert.html")

@app.route('/runBERT', methods=['GET'])
def runBERTFunc():
        filename = request.args['f']
        output_file = filename.split(".")[0] + ".tei.csv"
        if not os.path.isfile("data_output/%s"%output_file,):
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
	    async_task = AsyncBERTTask(filename=filename)
	    async_task.start()
	    return "1"
	print("File Exists")
	return "2"

@app.route('/processed', methods=['GET'])
def BERTprocessed():
        output_rows = []
        input_rows = []
	filename = request.args['f']
	output_file = filename.split(".")[0] + ".tei.csv"
        with open("data_output/%s"%output_file,'rt')as f:
            data = csv.reader(f)
            for row in data:
                output_rows.append(row)
	with open("data_input/%s"%output_file,'rt')as f:
            data = csv.reader(f)
            for row in data:
                input_rows.append(row)
        citation_data = []
        for x in range(1, len(input_rows)):
            j_l = eval(input_rows[x][2])
            j_l = {str(x): unicode(",".join(j_l[x]), errors='ignore') if x == 'authors' else unicode(j_l[x], "utf8") if j_l[x] else "" for x in j_l}
            j_l['cite_txt'] = unicode(input_rows[x][1], "utf8")
            j_l['cite_output'] = "green" if output_rows[x][2] == 'supporting' else "red" if output_rows[x][2] == 'contradicting' \
                    else "blue"
            citation_data.append(j_l)
        print(citation_data)
    	return render_template("bert.html", data=citation_data)


if __name__ == '__main__':
	app.run(host="0.0.0.0", debug=True, port=PORT)
