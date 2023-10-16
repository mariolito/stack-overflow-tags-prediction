import os.path

from flask import Flask, jsonify, request
from gevent.pywsgi import WSGIServer
from logging import config as log_conf
from predict.predict_tags import predict, read_models
import logging
log_conf.dictConfig({'version': 1, 'disable_existing_loggers': False, 'formatters': {
    'simple': {'class': 'logging.Formatter',
               'format': '%(asctime)s - %(levelname)s - %(threadName)s - %(name)s  - %(message)s'}}, 'handlers': {
    'console': {'class': 'logging.StreamHandler', 'formatter': 'simple', 'stream': 'ext://sys.stdout'}},
                     'loggers': {'sl_utils_tagger': {'handlers': ['console'], 'level': 'INFO', 'propagate': False},
                                 'requests': {'level': 'ERROR'}}})
_logger = logging.getLogger('ds.' + __name__)
app = Flask(__name__)
_logger.info('starting')

model_results_path = os.path.join("data", "results", "codebert_model_results")

read_models(model_results_path)


def serve(content):
    result = {}
    predicted_label_names, scores = predict(
        Title=content['Title'], Body=content['Body']
    )
    result['labels'] = predicted_label_names
    result['scores'] = scores
    return result


@app.route('/predict', methods=['GET', 'POST'])
def serve_utils():
    result = {}
    if request.method == 'POST':
        content = request.json
        try:
            result = serve(content=content)
        except:
            _logger.warning('Error with data: '+str(content))
    return jsonify(result)


if __name__ == '__main__':
    http_server = WSGIServer(("127.0.0.1", 5000), app)
    http_server.serve_forever()
