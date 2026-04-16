import os
import json
import argparse
from src.agents.outline_writer import outlineWriter
from src.agents.writer import subsectionWriter
from src.database import database, database_survey
from src.rag import GeneralRAG_langchain
from tqdm import tqdm
import time
import re


def remove_descriptions_subquery(text):
    lines = text.split('\n')
    
    filtered_lines = [line for line in lines if line.strip().startswith("#")]
    
    result = '\n'.join(filtered_lines)
    
    return result

def write(topic, model, section_num, subsection_len, rag_num, refinement):
    outline, outline_wo_description = write_outline(topic, model, section_num)

    if refinement:
        raw_survey, raw_survey_with_references, raw_references, refined_survey, refined_survey_with_references, refined_references = write_subsection(topic, model, outline, subsection_len = subsection_len, rag_num = rag_num, refinement = True)
        return refined_survey_with_references
    else:
        raw_survey, raw_survey_with_references, raw_references = write_subsection(topic, model, outline, subsection_len = subsection_len, rag_num = rag_num, refinement = False)
        return raw_survey_with_references

def write_outline(args, topic, model, ckpt, section_num, outline_reference_num, db, api_key, api_url):
    outline_writer = outlineWriter(args=args, model=model, ckpt=ckpt, api_key=api_key, api_url = api_url, database=db)
    outline = outline_writer.draft_outline(topic, outline_reference_num, 30000, section_num)
    outline_writer.print_token_usage()
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_1 = f"{args.saving_path}/outlines_with_des_{timestamp}.txt"
    with open(filename_1, "w") as f:
        f.write(outline + '\n\n')
    filename_2 = f"{args.saving_path}/outlines_without_des_{timestamp}.txt"
    with open(filename_2, "w") as f:
        f.write(remove_descriptions_subquery(outline) + '\n\n')
        
    outline_writer.print_token_usage()
    
    print(outline)

    def duplicate_first_last_sections(markdown_content):

        pattern = r'(## \d+\.?\s*.*?(?=\n##|\Z))'
        sections = re.findall(pattern, markdown_content, re.DOTALL)
        
        if len(sections) < 2:
            return markdown_content  
        
        first_section = sections[0]
        last_section = sections[-1]
        

        first_section_number = re.search(r'## (\d+)', first_section).group(1)
        first_title = first_section.split('\n')[0].strip()
        first_content = '\n'.join(first_section.split('\n')[1:]).strip()
        new_first_section = (f"{first_title}\n{first_content}\n\n"
                            f"### {first_section_number}.1 {first_title.split(maxsplit=2)[-1]}\n"
                            f"Description: {first_content}\n\n")
        

        last_section_number = re.search(r'## (\d+)', last_section).group(1)
        last_title = last_section.split('\n')[0].strip()
        last_content = '\n'.join(last_section.split('\n')[1:]).strip()
        new_last_section = (f"{last_title}\n{last_content}\n\n"
                            f"### {last_section_number}.1 {last_title.split(maxsplit=2)[-1]}\n"
                            f"Description: {last_content}\n")
        

        markdown_content = markdown_content.replace(first_section, new_first_section)
        markdown_content = markdown_content.replace(last_section, new_last_section)
        
        return markdown_content

    outline = duplicate_first_last_sections(outline)

    return outline, remove_descriptions_subquery(outline)

def write_subsection(args, topic, model, ckpt, outline, subsection_len, rag_num, rag_max_out, db, api_key, api_url, refinement = True):
    def remove_first_last_subsection_titles(markdown_content):
        subsection_pattern = r'\n(### \d+\.\d+[^\n]*)\n'
        subsections = re.findall(subsection_pattern, markdown_content)
        
        if len(subsections) < 2:
            return markdown_content
        
        first_subsection = subsections[0]
        last_subsection = subsections[-1]

        new_content = re.sub(r'\n' + re.escape(first_subsection) + r'\n', '\n', markdown_content)

        new_content = re.sub(r'\n' + re.escape(last_subsection) + r'\n', '\n', new_content)

        new_content = re.sub(r'\n\n\n+', '\n\n', new_content)
        
        return new_content
    
    subsection_writer = subsectionWriter(args=args, model=model, ckpt=ckpt, api_key=api_key, api_url = api_url, database=db)
    if refinement:
        raw_survey, raw_survey_with_references, raw_references, refined_survey, refined_survey_with_references, refined_references = subsection_writer.write(topic, outline, subsection_len = subsection_len, rag_num = rag_num, rag_max_out=rag_max_out, refining = True)
        subsection_writer.print_token_usage()
        return raw_survey, raw_survey_with_references, raw_references, remove_first_last_subsection_titles(refined_survey), remove_first_last_subsection_titles(refined_survey_with_references), refined_references
    else:
        raw_survey, raw_survey_with_references, raw_references = subsection_writer.write(topic, outline, subsection_len = subsection_len, rag_num = rag_num, rag_max_out=rag_max_out, refining = False)
        subsection_writer.print_token_usage()
        return remove_first_last_subsection_titles(raw_survey), remove_first_last_subsection_titles(raw_survey_with_references), raw_references
    

def paras_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu',default='0', type=str, help='Specify the GPU to use')
    parser.add_argument('--saving_path',default='./output/', type=str, help='Directory to save the output survey')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--model',default='gpt-4o-mini-2024-07-18', type=str, help='Model to use')
    parser.add_argument('--ckpt',default='', type=str, help='Checkpoint to use')
    parser.add_argument('--topic',default='Multimodal Large Language Models', type=str, help='Topic to generate survey for')
    parser.add_argument('--section_num',default=6, type=int, help='Number of sections in the outline')
    parser.add_argument('--subsection_len',default=500, type=int, help='Length of each subsection')
    parser.add_argument('--outline_reference_num',default=1500, type=int, help='Number of references for outline generation')
    parser.add_argument('--rag_num',default=100, type=int, help='Number of references to use for RAG')
    parser.add_argument('--rag_max_out',default=60, type=int, help='Number of references to use for RAG')
    parser.add_argument('--api_url',default='https://api.openai.com/v1/chat/completions', type=str, help='url for API request')
    parser.add_argument('--api_key',default='', type=str, help='API key for the model')
    parser.add_argument('--db_path',default='./database', type=str, help='Directory of the database.')
    parser.add_argument('--survey_outline_path',default='', type=str, help='Directory of the outline database of survey.')
    parser.add_argument('--embedding_model',default='./gte-large-en-v1.5', type=str, help='Embedding model for retrieval.')
    args = parser.parse_args()
    return args

def main(args):
    print(args)
    print("########### Loading database and RAG Index... ###########")
    db_paper = database(db_path = args.db_path, embedding_model = args.embedding_model)
    db_survey = database_survey(db_path = args.db_path, embedding_model = args.embedding_model)

    abs_index_db_path = f'{args.db_path}/faiss_paper_title_abs_embeddings_FROM_2012_0101_TO_240926.bin'
    title_index_db_path = f'{args.db_path}/faiss_paper_title_embeddings_FROM_2012_0101_TO_240926.bin'
    doc_db_path = f'{args.db_path}/arxiv_paper_db_with_cc.json'
    arxivid_to_index_path = f'{args.db_path}/arxivid_to_index_abs.json'
    
    rag_abstract4outline = GeneralRAG_langchain(args=args,
                                                retriever_type='vectorstore',
                                                index_db_path=abs_index_db_path,
                                                doc_db_path=doc_db_path,
                                                arxivid_to_index_path=arxivid_to_index_path,
                                                embedding_model=args.embedding_model)

    rag_abstract4suboutline = rag_abstract4outline
        
    rag_abstract4subsection = rag_abstract4outline

    rag_title4citation = GeneralRAG_langchain(args=args,
                                              retriever_type='vectorstore',
                                              index_db_path=title_index_db_path,
                                              doc_db_path=doc_db_path,
                                              arxivid_to_index_path=arxivid_to_index_path,
                                              embedding_model=args.embedding_model)

    if not os.path.exists(args.saving_path):
        os.mkdir(args.saving_path)
    db = {
        "paper": db_paper, 
        "survey": db_survey,
        "rag_outline": rag_abstract4outline, 
        "rag_suboutline": rag_abstract4suboutline,
        "rag_subsection": rag_abstract4subsection,
        "rag_title4citation": rag_title4citation
    }
    
    print("########### Writing outline... ###########")
    
    outline_with_description, outline_wo_description = \
        write_outline(args, args.topic, args.model, args.ckpt, args.section_num, args.outline_reference_num, db, args.api_key, args.api_url)
    
    print("########### Writing content... ###########")

    raw_survey, raw_survey_with_references, raw_references, refined_survey, refined_survey_with_references, refined_references = \
        write_subsection(args, args.topic, args.model, args.ckpt, outline_with_description, args.subsection_len, args.rag_num, args.rag_max_out, db, args.api_key, args.api_url)

    with open(f'{args.saving_path}/{args.topic}.md', 'a+') as f:
        f.write(refined_survey_with_references)
    with open(f'{args.saving_path}/{args.topic}.json', 'a+') as f:
        save_dic = {}
        save_dic['survey'] = refined_survey_with_references
        save_dic['reference'] = refined_references
        f.write(json.dumps(save_dic, indent=4))

if __name__ == '__main__':

    args = paras_args()

    main(args)
