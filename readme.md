- apt-get install default-jre
- First Install Grobid by following instructions from "https://grobid.readthedocs.io/en/latest/Install-Grobid/"  and         "https://grobid.readthedocs.io/en/latest/Grobid-service/"
  1. To Install : cd grobid  && ./gradlew clean install
  2. Run virtual env : in ~/bert_hr/
  3. To Run : ./gradlew run to run the server for grobid _parser_pdf.py

- Run :  script "~/bert_hr/grobid/grobid_parser_pdf.py" with all to-be-processed-pdf-files stored in "../data_pdf/" directory
  Results would be stored in "../data_xml/" directory
- Run :  “~/bert_hr/grobid/xml_parser.py” to process data in “../data_xml/” and saving in “../data_input/” directory for BERT prediction.
- ‘berthr.py’ would take csv files from “bert_hr/data_input/” to do the predictions while saving data in “bert_hr/data_output/”    directory.
- Run : ‘~/bert_hr/flask_app/berthr.py’ for predictions whose results would be saved in "data_output" directory.



- "flask_app/templates/" directory contains the required UI files.


- "data" directory contains the checkpoints of the custom trained BERT model (note: files can be found in the shared Drive folder )
