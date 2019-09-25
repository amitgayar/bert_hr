#First Install Grobid by following instructions from "https://grobid.readthedocs.io/en/latest/Install-Grobid/"  and 
# "https://grobid.readthedocs.io/en/latest/Grobid-service/"
# cd grobid
# ./gradlew clean install
# ./gradlew run
# Now run the script "grobid_parser_pdf.py" with all to-be-processed-pdf-files stored in "./data/input/" directory
# Results would be stored in "./data/output/" directory
# source : https://github.com/kermitt2/grobid-client-python
import sys
import os
import io
import json
import argparse
import time
import concurrent.futures
from client import ApiClient
import ntpath
import requests

class grobid_client(ApiClient):

    def __init__(self, config_path='config.json'):
        self.config = None
        self._load_config(config_path)

    def _load_config(self, path='config.json'):
        """
        Load the json configuration 
        """
        print(path)
        config_json = open(path).read()
        self.config = json.loads(config_json)

        # test if the server is up and running...
        the_url = 'http://'+self.config['grobid_server']
        if len(self.config['grobid_port'])>0:
            the_url += ":"+self.config['grobid_port']
        the_url += "/api/isalive"
        r = requests.get(the_url)
        status = r.status_code

        if status != 200:
            print('GROBID server does not appear up and running ' + str(status))
        else:
            print("GROBID server is up and running")

    def process(self, input, output, n, service, generateIDs, consolidate_header, consolidate_citations, force, teiCoordinates):
        batch_size_pdf = self.config['batch_size']
        pdf_files = []
        
        for (dirpath, dirnames, filenames) in os.walk(input):
            for filename in filenames:
                if filename.endswith('.pdf') or filename.endswith('.PDF'): 
                    pdf_files.append(os.sep.join([dirpath, filename]))

                    if len(pdf_files) == batch_size_pdf:
                        self.process_batch(pdf_files, output, n, service, generateIDs, consolidate_header, consolidate_citations, force, teiCoordinates)
                        pdf_files = []

        # last batch
        if len(pdf_files) > 0:
            self.process_batch(pdf_files, output, n, service, generateIDs, consolidate_header, consolidate_citations, force, teiCoordinates)

    def process_batch(self, pdf_files, output, n, service, generateIDs, consolidate_header, consolidate_citations, force, teiCoordinates):
        print(len(pdf_files), "PDF files to process")
        #with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n) as executor:
            for pdf_file in pdf_files:
                executor.submit(self.process_pdf, pdf_file, output, service, generateIDs, consolidate_header, consolidate_citations, force, teiCoordinates)

    def process_pdf(self, pdf_file, output, service, generateIDs, consolidate_header, consolidate_citations, force, teiCoordinates):
        # check if TEI file is already produced 
        # we use ntpath here to be sure it will work on Windows too
        pdf_file_name = ntpath.basename(pdf_file)
        if output is not None:
            filename = os.path.join(output, os.path.splitext(pdf_file_name)[0] + '.tei.xml')
        else:
            filename = os.path.join(ntpath.dirname(pdf_file), os.path.splitext(pdf_file_name)[0] + '.tei.xml')

        if not force and os.path.isfile(filename):
            print(filename, "already exist, skipping... (use --force to reprocess pdf input files)")
            return True

        print(pdf_file)
        files = {
            'input': (
                pdf_file,
                open(pdf_file, 'rb'),
                'application/pdf',
                {'Expires': '0'}
            )
        }
        
        the_url = 'http://'+self.config['grobid_server']
        if len(self.config['grobid_port'])>0:
            the_url += ":"+self.config['grobid_port']
        the_url += "/api/"+service

        # set the GROBID parameters
        the_data = {}
        if generateIDs:
            the_data['generateIDs'] = '1'
        if consolidate_header:
            the_data['consolidateHeader'] = '1'
        if consolidate_citations:
            the_data['consolidateCitations'] = '1'   
        if teiCoordinates:
            the_data['teiCoordinates'] = self.config['coordinates'] 

        res, status = self.post(
            url=the_url,
            files=files,
            data=the_data,
            headers={'Accept': 'text/plain'}
        )

        if status == 503:
            time.sleep(self.config['sleep_time'])
            return self.process_pdf(pdf_file, output)
        elif status != 200:
            print('Processing failed with error ' + str(status))
            return False
        else:
            # writing TEI file
            try:
                with io.open(filename,'w',encoding='utf8') as tei_file:
                    tei_file.write(res.text)
                return True
            except OSError:  
               print ("Writing resulting TEI XML file %s failed" % filename)
               pass



# ==================================================================================================================
input_path = '../data_pdf/'
output_path = '../data_xml/'
service = 'processFulltextDocument'
generateIDs = 1
consolidate_citations = 1
consolidate_header = 1
teiCoordinates = 1
force = False
n =1
config_path = './config.json'
#client = grobid_client(config_path=config_path)

start_time = time.time()

#client.process(input_path, output_path, n, service, generateIDs, consolidate_header, consolidate_citations, force, teiCoordinates)

runtime = round(time.time() - start_time, 3)
print("runtime: %s seconds " % (runtime))
