import os
import pandas as pd
import base64
import config
from src.services import get_xml
from anuvaad_auditor.loghandler import log_info
from anuvaad_auditor.loghandler import log_error
from src.errors.errors_exception import ServiceError
from anuvaad_auditor.loghandler import log_exception
from src.services.preprocess import prepocess_pdf_regions
from src.services.get_tables import  get_text_table_line_df
from src.services.child_text_unify_to_parent import ChildTextUnify
from src.services.get_response import process_image_df,  process_table_df, df_to_json, process_line_df

def doc_pre_processing(filename):
    '''
        Preprocessing on input pdf to get:
            - xml files
            - images 
            - background images 
            - header and footer regions

    '''
    log_info("Service main", "document preprocessing started  ===>", None)

    img_dfs,bg_files,xml_dfs, page_width, page_height,working_dir  = get_xml.process_input_pdf(filename)
    multiple_pages = False
    pages          = len(xml_dfs)
    if pages > 1:
        multiple_pages =True
    try:
        header_region, footer_region = prepocess_pdf_regions(xml_dfs, page_height)
    except Exception as e :
            log_error("Service prepocess", "Error in finding footer and header region", None, e)

    log_info("Service main", "document preprocessing successfully completed", None)

    return img_dfs,bg_files,xml_dfs, pages, working_dir, header_region , footer_region, multiple_pages, page_width, page_height

def doc_structure_analysis(pages,xml_dfs,img_dfs,working_dir,header_region , footer_region, multiple_pages):
    
    '''
        Document structure analysis to get:
            - in_dfs
            - table_dfs
            - line_dfs
            - h_dfs
            - v_dfs
            - p_dfs
            - text_block_dfs

    '''
    log_info("Service main", "document structure analysis started  ===>", None)
    
    text_merger = ChildTextUnify()
    in_dfs, table_dfs, line_dfs = get_text_table_line_df(pages,working_dir, xml_dfs,img_dfs)
    h_dfs          = get_xml.get_hdfs(pages, in_dfs, config.DOCUMENT_CONFIGS,header_region , footer_region, multiple_pages)
    v_dfs          = get_xml.get_vdfs(pages, h_dfs, config.DOCUMENT_CONFIGS)
    p_dfs          = get_xml.get_pdfs(pages, v_dfs, config.DOCUMENT_CONFIGS, config.BLOCK_CONFIGS)
    text_block_dfs = text_merger.unify_child_text_blocks(pages, p_dfs, config.DROP_TEXT)

    log_info("Service main", "document structure analysis successfully completed", None)

    return text_block_dfs, table_dfs, line_dfs

def doc_structure_response(pages,img_dfs, text_block_dfs,table_dfs,line_dfs,page_width, page_height):

    '''
        To build required response in json format;
            -  page level information:
                    - page_no
                    - page_width
                    - page_height
                    - images
                    - tables
                    - text_blocks
            -  convert dataframe into proper json format:
                    - img_df
                    - text_df
                    - tabel_df
    '''
    log_info("Service main", "document structure response started  ===>", None)

    response = { 'result' : [] }
    for page_index in range(pages):
        img_df     = img_dfs[page_index]
        text_df    = text_block_dfs[page_index]
        table_df   = table_dfs[page_index]
        line_df    = line_dfs[page_index]
        page_json  = response_per_page(text_df, img_df,table_df,line_df, page_index, page_width, page_height)
        response['result'].append(page_json)
    
    log_info("Service main", "document structure response successfully completed", None)

    return response

def response_per_page(p_df,img_df,table_df,line_df,page_no,page_width,page_height):

    p_df['block_id'] = range(len(p_df))
    myDict           = {'page_no': page_no,'page_width': page_width,'page_height':page_height,'lines':[],'tables':[],'images':[],'text_blocks':[]}
    image_data       = process_image_df(myDict, img_df)
    table_data       = process_table_df(myDict, table_df)
    line_data        = process_line_df(myDict,line_df)
    text_data        = df_to_json(p_df)
    myDict['images'] = image_data
    myDict['tables'] = table_data
    myDict['lines']  = line_data
    myDict['text_blocks'] = text_data

    return myDict

def DocumentStructure(jobid, file_name, base_dir = config.BASE_DIR):
    try:
        img_dfs,bg_files, xml_dfs, pages, working_dir, header_region , footer_region, multiple_pages, page_width, page_height = doc_pre_processing(file_name)

        text_block_dfs, table_dfs, line_dfs = doc_structure_analysis(pages,xml_dfs,img_dfs,working_dir,header_region , footer_region, multiple_pages)

        response   =  doc_structure_response(pages, img_dfs, text_block_dfs, table_dfs,line_dfs,page_width, page_height)
        log_info("DocumentStructure","successfully received blocks in json response", jobid)
        return response

    except:
        log_exception("DocumentStructure","Error occured during pdf to blocks conversion", jobid, None)
        raise ServiceError(400, "documentstructure failed. Something went wrong during pdf to blocks conversion.")