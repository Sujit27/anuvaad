from src.utilities.model_response import CheckingResponse
from src.utilities.model_response import CustomResponse
from src.utilities.model_response import Status
from src.utilities.utils import FileOperation
from src.Kafka_module.producer import Producer
from src.Kafka_module.consumer import Consumer
import time
import config
import logging
from logging.config import dictConfig
log = logging.getLogger('file')

def process_tokenization_kf():
    file_ops = FileOperation()
    DOWNLOAD_FOLDER =file_ops.file_download(config.download_folder)
    task_id = str("TOK-" + str(time.time()).replace('.', ''))
    task_starttime = str(time.time()).replace('.', '')
    try:
        consumer_class = Consumer(config.tok_input_topic, config.bootstrap_server)
        consumer = consumer_class.consumer_instantiate()
        log.info("--- consumer running -----")
    except:
        response = Status.ERR_Consumer.value
        producer_tok = Producer(config.bootstrap_server) 
        producer = producer_tok.producer_fn()
        producer.send(config.tok_output_topic, value = response)
        producer.flush()
        log.error("error in kafka opertation while listening to consumer on topic %s"%(config.tok_input_topic))
        log.info("response send to topic %s"%(config.tok_output_topic))
    try:
        log.info("trying to receive value from consumer ")
        for msg in consumer:
            log.info("value received from consumer")
            data = msg.value
            task_id = str("TOK-" + str(time.time()).replace('.', ''))
            task_starttime = str(time.time()).replace('.', '')
            checking_response = CheckingResponse(data, task_id, task_starttime, DOWNLOAD_FOLDER)
            file_value_response = checking_response.main_response_wf()
            producer_tokenise = Producer(config.bootstrap_server) 
            producer = producer_tokenise.producer_fn()
            producer.send(config.tok_output_topic, value = file_value_response)
            producer.flush()
            log.info("producer flushed value on topic %s"%(config.tok_output_topic))
    except Exception as e:
        log.error("error occured during consumer running or flushing data to another queue %s"%e)
        task_end_time = str(time.time()).replace('.', '')
        output_file_response = ""
        for msg in consumer:
            log.info("value received from consumer")
            data = msg.value
            input_files, workflow_id, jobid, tool_name, step_order = file_ops.json_input_format(data)
            task_id = str("TOK-" + str(time.time()).replace('.', ''))
            task_starttime = str(time.time()).replace('.', '')
            response = CustomResponse(Status.ERR_Producer.value, jobid, workflow_id, tool_name, step_order, task_id, task_starttime, task_end_time, output_file_response)
            producer_tokenise = Producer(config.bootstrap_server) 
            producer = producer_tokenise.producer_fn()
            producer.send(config.tok_output_topic, value = response.status_code)
            producer.flush()
            log.info("error in kafka opertation producer flushed value on topic %s"%(config.tok_output_topic))
        

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] {%(filename)s:%(lineno)d} %(threadName)s %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {
        'info': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'default',
            'filename': 'info.log'
        },
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'default',
            'stream': 'ext://sys.stdout',
        }
    },
    'loggers': {
        'file': {
            'level': 'DEBUG',
            'handlers': ['info', 'console'],
            'propagate': ''
        }
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['info', 'console']
    }
})