import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import sys
import time
sys.path.append("src")
import ctranslate2
import services.translate as anuvaad_services
import services.document_translate as anuvaad_services_batch
import utilities.sentence_processor as sentence_processor
import utilities.sentencepiece_util as sp


# model path
# CTRANSLATE_MODEL = "ctranslate2_pt_file"
TRANSLATOR_DIR = "translator_end"

ANUVAAD_MODEL_PATH = "src/available_nmt_models/nmt/56"
ANUVAAD_ENCODER = "src/available_nmt_models/sentencepiece_models/en_exp-5.6-2019-12-09-24k.model"
ANUVAAD_DECODER = "src/available_nmt_models/sentencepiece_models/hi_exp-5.6-2019-12-09-24k.model"

# hyperparameters
DEVICE = "cpu"
QUANTIZATION = None
COMPUTE_TYPE = "default"
INTER_THREADS = 1
INTRA_THREADS = 4
MAX_BATCH_SIZE = 32
BEAM_SIZE = 5

M = 10 # number of times a translator is run to get the average execution time
N = 1  # number of times the anuvaad pipeline is run

# input for translation
INPUT_A = [['▁This', '▁being', '▁a', '▁simple', '▁sentence', '▁,', '▁it', '▁should', '▁be', '▁fairly', '▁easy',
 '▁to', '▁translate'], ['▁It', '▁might', '▁be', '▁that', '▁to', '▁surrender', '▁to', '▁happiness', '▁was', '▁to', 
 '▁accept', '▁defeat', '▁,', '▁but', '▁it', '▁was', '▁a', '▁defeat', '▁better', '▁than', '▁many', '▁vict', 'ories',
  '▁.'],['▁Ins', 'ide', '▁us', '▁there', '▁is', '▁something', '▁that', '▁has', '▁no', '▁name', '▁,', '▁that', 
  '▁something', '▁is', '▁what', '▁we', '▁are', '▁.'],['▁The', '▁creatures', '▁outside', '▁looked', '▁from', '▁pig',
   '▁to', '▁man', '▁,', '▁and', '▁from', '▁man', '▁to', '▁pig', '▁,', '▁and', '▁from', '▁pig', '▁to', '▁man', '▁again',
    '▁;', '▁but', '▁already', '▁it', '▁was', '▁impossible', '▁to', '▁say', '▁which', '▁was', '▁which', '▁.']]

# NUM_TOKENS_A = len([token for sentence in INPUT_A for token in sentence])
# print(NUM_TOKENS_A)

INPUT_B = [["['▁This',", "'▁being',", "'▁a',", "'▁simple',", "'▁sentence',", "'▁,',", "'▁it',", "'▁should',", 
"'▁be',", "'▁fairly',", "'▁easy',", "'▁to',", "'▁translate']"], ["['▁It',", "'▁might',", "'▁be',", "'▁that',", 
"'▁to',", "'▁surrender',", "'▁to',", "'▁happiness',", "'▁was',", "'▁to',", "'▁accept',", "'▁defeat',", 
"'▁,',", "'▁but',", "'▁it',", "'▁was',", "'▁a',", "'▁defeat',", "'▁better',", "'▁than',", "'▁many',",
"'▁vict',", "'ories',", "'▁.']"], ["['▁Ins',", "'ide',", "'▁us',", "'▁there',", "'▁is',", "'▁something',",
"'▁that',", "'▁has',", "'▁no',", "'▁name',", "'▁,',", "'▁that',", "'▁something',", "'▁is',", "'▁what',",
"'▁we',", "'▁are',", "'▁.']"], ["['▁The',", "'▁creatures',", "'▁outside',", "'▁looked',", "'▁from',", 
"'▁pig',", "'▁to',", "'▁man',", "'▁,',", "'▁and',", "'▁from',", "'▁man',", "'▁to',", "'▁pig',", "'▁,',",
"'▁and',", "'▁from',", "'▁pig',", "'▁to',", "'▁man',", "'▁again',", "'▁;',", "'▁but',", "'▁already',",
"'▁it',", "'▁was',", "'▁impossible',", "'▁to',", "'▁say',", "'▁which',", "'▁was',", "'▁which',", "'▁.']"]]
# NUM_TOKENS_B = len([token for sentence in INPUT_B for token in sentence])
# print(NUM_TOKENS_B)

INPUT_SAMPLE = {'id': 56, 'src': "On offering to help the blind man, the man who then stole his car,\
 had not, at that precise moment, had any evil intention, quite the contrary, what he did was nothing \
     more than obey those feelings of generosity and altruism which, as everyone knows, are the two best \
         traits of human nature and to be found in much more hardened criminals than this one, a simple \
             car-thief without any hope of advancing in his profession, exploited by the real owners of \
                 this enterprise, for it is they who take advantage of the needs of the poor."}

INPUT_FILE = "wmt14-en-de.src"

def file_to_input(input_file, return_sample=False):
    '''
    Reads in a file with one sentence in each line and
    returns a dict type input for anuvaad pipeline
    '''
    with open(input_file,'r') as f:
        input_text_array = f.readlines()

    input_text_array = [sent[:-1] for sent in input_text_array]
    # input_text = " ".join(input_text_array)
    input_for_anuvaad = [{'id': 56, 'src': input_text} for input_text in input_text_array]
    
    if return_sample:
        return input_for_anuvaad[:32]
    else:
        return input_for_anuvaad

def file_to_batch_input(input_file, return_sample=False):
    '''
    Reads in a file with one sentence in each line and
    returns a dict type input for anuvaad batch pipeline
    '''
    with open(input_file,'r') as f:
        input_text_array = f.readlines()

    input_text_array = [sent[:-1] for sent in input_text_array]
    # input_text = " ".join(input_text_array)
    input_for_anuvaad = {'id': 56, 'src_list': input_text_array} 
    
    if return_sample:
        input_for_anuvaad['src_list'] = input_for_anuvaad['src_list'][:32]
        return input_for_anuvaad
    else:
        return input_for_anuvaad

def eval_ctranslation(input_for_translation):
    '''
    Shows time taken by the plain ctranslate2 env to translate the input
    '''

    # converter = ctranslate2.converters.OpenNMTPyConverter(CTRANSLATE_MODEL)
    
    # output_dir = converter.convert(TRANSLATOR_DIR, "TransformerBase", quantization=QUANTIZATION, force=True)
    translator = ctranslate2.Translator(TRANSLATOR_DIR, compute_type=COMPUTE_TYPE, inter_threads=INTER_THREADS, intra_threads=INTRA_THREADS)
    start = time.process_time()
    for _ in range(M):
        output = translator.translate_batch(input_for_translation, max_batch_size=MAX_BATCH_SIZE, beam_size=BEAM_SIZE)
    
    time_taken = (time.process_time() - start)/M
    num_tokens =  len([token for sentence in input_for_translation for token in sentence])
    time_taken_per_token = time_taken/num_tokens

    return time_taken_per_token, output

def eval_anuvaad_translation(input_for_translation):
    '''
    Shows time taken by the anuvaad translator to translate the input
    '''
    translator = ctranslate2.Translator(ANUVAAD_MODEL_PATH, compute_type=COMPUTE_TYPE, inter_threads=INTER_THREADS, intra_threads=INTRA_THREADS)   
    start = time.process_time()
    for _ in range(M):
        output = translator.translate_batch(input_for_translation, max_batch_size=MAX_BATCH_SIZE, beam_size=BEAM_SIZE)
    
    time_taken = (time.process_time() - start)/M
    num_tokens =  len([token for sentence in input_for_translation for token in sentence])
    # print(num_tokens)
    time_taken_per_token = time_taken/num_tokens

    return time_taken_per_token, output

def eval_anuvaad_pipeline(input_dict,mode):
    '''
    Shows time taken by anuvaad pipeline in
    1. Tokenization
    2. Encoding
    3. Translation
    4. Decoding
    5. Detokenization
    '''
    translator = ctranslate2.Translator(ANUVAAD_MODEL_PATH,compute_type=COMPUTE_TYPE, inter_threads=INTER_THREADS, intra_threads=INTRA_THREADS)
    
    ###
    if input_dict['src'].isupper():
        input_dict['src'] = input_dict['src'].title()
    start = time.process_time()
    for _ in range(N):
        input_tokenized = sentence_processor.moses_tokenizer(input_dict['src'])
    time_tokenizing = (time.process_time() - start)/N

    ###
    start = time.process_time()
    for _ in range(N):
        input_encoded = str(sp.encode_line(ANUVAAD_ENCODER,input_tokenized))
        input_final = anuvaad_services.format_converter(input_encoded)
    time_encoding = (time.process_time() - start)/N
    
    ###
    start = time.process_time()
    for _ in range(N):
        output = translator.translate_batch([input_final], max_batch_size=MAX_BATCH_SIZE, beam_size = BEAM_SIZE)
    time_translation = (time.process_time() - start)/N

    ###
    # translations = list()
    output = " ".join(output[0][0]['tokens'])
    start = time.process_time()
    for _ in range(N):
        output_decoded = sp.decode_line(ANUVAAD_DECODER,output)
    time_decoding = (time.process_time() - start)/N

    ###
    start = time.process_time()
    for _ in range(N):
        translation = sentence_processor.indic_detokenizer(output_decoded)
    time_detokenzing = (time.process_time() - start)/N
    
    num_tokens = len(input_final)
    # print(num_tokens)
    # print(translation)
    if mode == 0:
        return time_tokenizing / num_tokens, time_encoding / num_tokens, time_translation / num_tokens, time_decoding / num_tokens, time_detokenzing / num_tokens
    elif mode == 1:
        return translation, time_translation / num_tokens
    else:
        return input_final

def find_number_of_subwords(input_for_translation_list):
    '''
    Finds number of subwords, the input is a list of dictionaries
    with 'id' and 'src' key
    '''
    num_tokens = 0
    for input_for_translation in input_for_translation_list:
        if input_for_translation['src'].isupper():
            input_for_translation['src'] = input_for_translation['src'].title()
        input_tokenized = sentence_processor.moses_tokenizer(input_for_translation['src'])
        input_encoded = str(sp.encode_line(ANUVAAD_ENCODER,input_tokenized))
        input_final = anuvaad_services.format_converter(input_encoded)

        num_tokens += len(input_final)

    return num_tokens

def eval_anuvaad_batch_translation(input_for_translation):
    '''
    Evaluates performance of document translate (batch)
    '''
    translator_service = anuvaad_services_batch.NMTTranslateService()
    start = time.time()
    for _ in range(M):
        output = translator_service.batch_translator(input_for_translation)
    
    time_taken = (time.time() - start)/M
    return output, time_taken

def main():
    print("Checking performance...")

    # print("Checking german translation performance")
    # time_taken_per_token_german, german_translation = eval_ctranslation(INPUT_A)
    # print("German translation: ", german_translation)
    # print("Time taken for german translation per token: ", time_taken_per_token_german)

    # print("\n")

    # print("Checking hindi translation performance")
    # time_taken_per_token_hindi, hindi_translation = eval_anuvaad_translation(INPUT_B)
    # print("Hindi translation: ", hindi_translation)
    # print("Time taken for hindi translation per token: ", time_taken_per_token_hindi)

    # print("\n")

    # print("Checking anuvaad pipeline performance")
    # pipeline_times_per_token = eval_anuvaad_pipeline(INPUT_SAMPLE, 0)
    # print("\n".join(map(str,pipeline_times_per_token)))

    # print("Checking anuvaad pipeline performance")
    # translation, _ = eval_anuvaad_pipeline(INPUT_SAMPLE, 1)
    # print(translation)

    # print("\n")

    # print("Checking anuvaad translation performance, taking one line input each time")
    # input_dict_list = file_to_input(INPUT_FILE, return_sample=True)
    # time_translation_per_token_list = []
    # for input_dict in input_dict_list:
    #     _, time_translation_per_token = eval_anuvaad_pipeline(input_dict, 1)
    #     time_translation_per_token_list.append(time_translation_per_token)
    # avg_time_translation_per_token = sum(time_translation_per_token_list) / len(time_translation_per_token_list)
    # print(avg_time_translation_per_token)

    # print("\n")

    # print("Checking anuvaad translation performance, taking batch input")
    # input_dict_list = file_to_input(INPUT_FILE,return_sample=True)
    # formatted_input_array = []
    # for input_dict in input_dict_list:
    #     formatted_input_array.append(eval_anuvaad_pipeline(input_dict,2))
    # time_taken_per_token_hindi, _ = eval_anuvaad_translation(formatted_input_array)
    # print(time_taken_per_token_hindi)

    # print("Checking anuvaad translation batch sequence order")
    # input_dict_list = file_to_input(INPUT_FILE,return_sample=True)
    # formatted_input_array = []
    # for input_dict in input_dict_list:
    #     formatted_input_array.append(eval_anuvaad_pipeline(input_dict,2))
    # _, output_array = eval_anuvaad_translation(formatted_input_array)
    # for i in range(len(output_array)):
    #     output = " ".join(output_array[i][0]['tokens'])
    #     output_decoded = sp.decode_line(ANUVAAD_DECODER,output)
    #     translation = sentence_processor.indic_detokenizer(output_decoded)
    #     print(input_dict_list[i]['src'] + " : " + translation)

    print("Checking anuvaad document(batch) translator")
    num_input_tokens = find_number_of_subwords(file_to_input(INPUT_FILE, return_sample=True))
    batch_input = file_to_batch_input(INPUT_FILE, return_sample=True)
    _, time_taken = eval_anuvaad_batch_translation(batch_input)
    print("Time taken per token : ", time_taken/num_input_tokens)




if __name__ == "__main__":
    main()