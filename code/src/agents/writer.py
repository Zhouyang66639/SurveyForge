import os
import re
import threading
import jsonlines
import numpy as np
from tqdm import trange,tqdm
import torch
from src.model import APIModel, LocalModel
import time
from src.utils import tokenCounter, get_index_filter
import copy
import json
from src.database import database
from src.prompt import SUBSECTION_WRITING_PROMPT, LCE_PROMPT, CHECK_CITATION_PROMPT
import sys
import concurrent.futures

class subsectionWriter():
    
    def __init__(self, args, model:str, ckpt:str, api_key:str, api_url:str,  database) -> None:
        self.args = args
        self.model, self.ckpt, self.api_key, self.api_url = model, ckpt, api_key, api_url 
        if self.model == 'local':
            self.api_model = LocalModel(self.ckpt)
        else:
            self.api_model = APIModel(self.model, self.api_key, self.api_url)

        self.db = database
        self.token_counter = tokenCounter()
        self.input_token_usage, self.output_token_usage = 0, 0

    def print_token_usage(self):
        print(f"SubsectionWriter Input token usage: {self.input_token_usage}")
        print(f"SubsectionWriter Output token usage: {self.output_token_usage}")

    def write(self, topic, outline, rag_num = 30, rag_max_out = 60 ,subsection_len = 500, refining = True, reflection=True):
        # HACK: Database subset for outline generation
        arxivid_list = list(self.db["rag_outline"].id_to_index.keys())
        arxivid_period = [arxivid for arxivid in arxivid_list if arxivid.split('.')[0] <= '2412']
        rag_outline_subset_index_filter = get_index_filter(self.db["rag_outline"].id_to_index, 
                                                           arxivid_period)
        rag_outline_subset_ids = self.db["rag_outline"].retrieve_id([topic], 
                                                                    top_k=1500,
                                                                    **rag_outline_subset_index_filter)
        if self.args.debug:
            with open(f"{self.args.saving_path}/rag_outline_subset_ids.jsonl", 'w') as f:
                writer = jsonlines.Writer(f)
                line = {"references_ids": rag_outline_subset_ids}
                writer.write(line)

        writer_subsection_index_filter = get_index_filter(self.db["rag_outline"].id_to_index, 
                                                          rag_outline_subset_ids)
            
        # Get database
        parsed_outline = self.parse_outline(outline=outline)
        section_content = [[]] * len(parsed_outline['sections'])

        section_paper_texts = [[]] *  len(parsed_outline['sections'])
        
        total_ids = []
        section_references_ids = [[]] * len(parsed_outline['sections'])
        
        for i in range(len(parsed_outline['sections'])):
            subtitles = parsed_outline['subsections'][i]
            descriptions = parsed_outline['subsection_descriptions'][i]
            subsection_query = parsed_outline['subsection_query'][i]
            for t, d, q in zip(subtitles, descriptions, subsection_query):
                if q == []:
                    q = [f'{d}']
                # TODO: combine query
                pure_t = re.sub(r'\d+\.\d+\s*', '', t)
                if 'Introduction' in pure_t or 'Conclusion' in pure_t:
                    pure_t = ''
                q = [f'{pure_t} {sub_q}' for sub_q in q]
                # print(f"description:{d} \n sub_queries:{q}")
                # references_ids = self.db.get_ids_from_query(d, num = rag_num, shuffle = False)
                references_ids = self.db["rag_subsection"].retrieve_id(q, 
                                                                       search_type='similarity', 
                                                                       rerank='citation', 
                                                                       top_k=rag_num,
                                                                       max_out=rag_max_out,
                                                                       **writer_subsection_index_filter)

                total_ids += references_ids
                section_references_ids[i].append(references_ids)
        
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_total_ids = list(set(total_ids))
   
        if self.args.debug:
            with open(f"{self.args.saving_path}/total_ids.txt", "w") as f:
                for id in unique_total_ids:
                    f.write(f"{id}\n")
            with open(f"{self.args.saving_path}/section_references_ids.json", "w") as f:
                json.dump(section_references_ids, f)
                
        self.writer_rag_results = unique_total_ids
          
        total_references_infos = self.db["paper"].get_paper_info_from_ids(list(set(total_ids)))
        temp_title_dic = {p['id']:p['title'] for p in total_references_infos}
        temp_abs_dic = {p['id']:p['abs'] for p in total_references_infos}
        
        # temp_citation_dic = {p['id']:p['citation_count'] for p in total_references_infos}
        for i in range(len(parsed_outline['sections'])):
            for references_ids in section_references_ids[i]:
                
                references_titles = [temp_title_dic[_] for _ in references_ids]
                references_papers = [temp_abs_dic[_] for _ in references_ids]
                # references_citations = [temp_citation_dic[_] for _ in references_ids]
                paper_texts = '' 
                # for t, c, p in zip(references_titles, references_citations, references_papers):
                for t, p in zip(references_titles, references_papers):
                    paper_texts += f'---\n\npaper_title: {t}\n\npaper_content:\n\n{p}\n'
                paper_texts+='---\n'
                
                if self.args.debug:
                    with open(f"{self.args.saving_path}/paper_texts.txt", 'a') as f:
                        f.write(paper_texts + '\n\n')
    
                section_paper_texts[i].append(paper_texts)

        max_section_threads = 100
        try:
            max_section_threads = max(1, int(os.getenv("SF_MAX_SECTION_THREADS", "100")))
        except ValueError:
            max_section_threads = 100

        def write_subsection_wrapper(args):
            try:
                self.write_subsection_with_reflection(*args)
            except Exception as e:
                print(f"An error occurred while writing subsection: {e}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_section_threads) as executor:
            futures = []
            for i in range(len(parsed_outline['sections'])):
                args = (section_paper_texts[i], topic, outline, parsed_outline['sections'][i],
                        parsed_outline['subsections'][i], parsed_outline['subsection_descriptions'][i],
                        section_content, i, rag_num, str(subsection_len))
                futures.append(executor.submit(write_subsection_wrapper, args))

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='Section Threading'):
                future.result()  
        raw_survey = self.generate_document(parsed_outline, section_content)
        raw_survey_with_references, raw_references = self.process_references(raw_survey)
        
        if self.args.debug:
            with open(f'{self.args.saving_path}/raw_survey.txt', 'a') as f:
                f.write(raw_survey)
            with open(f'{self.args.saving_path}/raw_survey_with_references.jsonl', 'a') as f:
                writer = jsonlines.Writer(f)
                line = [{"raw_survey_with_references": raw_survey_with_references},
                        {"raw_references": raw_references}]
                writer.write(line)
                
        if refining:
            final_section_content = self.refine_subsections(topic, outline, section_content)
            refined_survey = self.generate_document(parsed_outline, final_section_content)
            refined_survey_with_references, refined_references = self.process_references(refined_survey)
        
            if self.args.debug:
                with open(f'{self.args.saving_path}/refined_survey.txt', 'a') as f:
                    f.write(refined_survey)
                with open(f'{self.args.saving_path}/refined_survey_with_references.jsonl', 'a') as f:
                    writer = jsonlines.Writer(f)
                    line = [{"refined_survey_with_references": refined_survey_with_references},
                            {"refined_references": refined_references}]
                    writer.write(line)

            return raw_survey.replace("---", "")+'\n', raw_survey_with_references.replace("---", "")+'\n', raw_references, refined_survey.replace("---", "")+'\n', refined_survey_with_references.replace("---", "")+'\n', refined_references
        else:
            return raw_survey.replace("---", "")+'\n', raw_survey_with_references.replace("---", "")+'\n', raw_references

    def compute_price(self):
        return self.token_counter.compute_price(input_tokens=self.input_token_usage, output_tokens=self.output_token_usage, model=self.model)

    def refine_subsections(self, topic, outline, section_content):
        section_content_even = copy.deepcopy(section_content)
        final_section_content = copy.deepcopy(section_content_even)

        max_section_threads = 100
        try:
            max_section_threads = max(1, int(os.getenv("SF_MAX_SECTION_THREADS", "100")))
        except ValueError:
            max_section_threads = 100

        def lce_wrapper(args):
            try:
                self.lce(*args)
            except Exception as e:
                print(f"An error occurred in lce: {e}")

        def process_sections(section_content, target_content, parity):
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_section_threads) as executor:
                futures = []
                for i in range(len(section_content)):
                    for j in range(len(section_content[i])):
                        if j % 2 == parity:
                            if j == 0 and parity == 0:
                                contents = [''] + section_content[i][:2] if len(section_content[i]) > 1 else [''] + section_content[i] + ['']
                            elif j == (len(section_content[i]) - 1):
                                contents = section_content[i][-2:] + ['']
                            else:
                                contents = section_content[i][j-1:j+2]
                            
                            args = (topic, outline, contents, target_content[i], j)
                            futures.append(executor.submit(lce_wrapper, args))

                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f'Section Refining Threading (parity {parity})'):
                    future.result() 

        # Process even indices
        process_sections(section_content, section_content_even, 0)

        # Process odd indices
        process_sections(section_content_even, final_section_content, 1)
        
        return final_section_content

    def write_subsection_with_reflection(self, paper_texts_l, topic, outline, section, subsections, subdescriptions, res_l, idx, rag_num = 20, subsection_len = 1000, citation_num = 8):
        
        prompts = []
        for j in range(len(subsections)):
            subsection = subsections[j]
            description = subdescriptions[j]

            prompt = self.__generate_prompt(SUBSECTION_WRITING_PROMPT, paras={'OVERALL OUTLINE': outline, 'SUBSECTION NAME': subsection,\
                                                                          'DESCRIPTION':description,'TOPIC':topic,'PAPER LIST':paper_texts_l[j], 'SECTION NAME':section, 'WORD NUM':str(subsection_len),\
                                                                            'CITATION NUM':str(citation_num)})
            prompts.append(prompt)

        self.input_token_usage += self.token_counter.num_tokens_from_list_string(prompts)
        start = time.time()
        contents = self.api_model.batch_chat(prompts, temperature=1)
        end = time.time()
        period = end - start
        with open(f"{self.args.saving_path}/time_cost.log", "a") as f:
            f.write(f"Content API: {period}\n")
        # print(f"##########Content API Time taken#########: {period}")
        self.output_token_usage += self.token_counter.num_tokens_from_list_string(contents)
        contents = [c.replace('<format>','').replace('</format>','') for c in contents]

        prompts = []
        for content, paper_texts in zip(contents, paper_texts_l):
            prompts.append(self.__generate_prompt(CHECK_CITATION_PROMPT, paras={'SUBSECTION': content, 'TOPIC':topic, 'PAPER LIST':paper_texts}))
        self.input_token_usage += self.token_counter.num_tokens_from_list_string(prompts)
        start = time.time()
        contents = self.api_model.batch_chat(prompts, temperature=1)
        end = time.time()
        period = end - start
        with open(f"{self.args.saving_path}/time_cost.log", "a") as f:
            f.write(f"Content Check API: {period}\n")
        # print(f"##########Content Check API Time taken#########: {period}")
        self.output_token_usage += self.token_counter.num_tokens_from_list_string(contents)
        contents = [c.replace('<format>','').replace('</format>','') for c in contents]
    
        res_l[idx] = contents
        return contents
    
    def __generate_prompt(self, template, paras):
        prompt = template
        for k in paras.keys():
            prompt = prompt.replace(f'[{k}]', paras[k])
        return prompt
    
    def generate_prompt(self, template, paras):
        prompt = template
        for k in paras.keys():
            prompt = prompt.replace(f'[{k}]', paras[k])
        return prompt
    
    def lce(self, topic, outline, contents, res_l, idx):
        prompt = self.__generate_prompt(LCE_PROMPT, paras={'OVERALL OUTLINE': outline,'PREVIOUS': contents[0],\
                                                                          'FOLLOWING':contents[2],'TOPIC':topic,'SUBSECTION':contents[1]})
        self.input_token_usage += self.token_counter.num_tokens_from_string(prompt)
        refined_content = self.api_model.chat(prompt, temperature=1).replace('<format>','').replace('</format>','')
        self.output_token_usage += self.token_counter.num_tokens_from_string(refined_content)
        res_l[idx] = refined_content
        return refined_content.replace('Here is the refined subsection:\n','')

    def parse_outline(self, outline):
        result = {
            "title": "",
            "sections": [],
            "section_descriptions": [],
            "subsections": [],
            "subsection_descriptions": [],
            "subsection_query": []
        }

        # Split the outline into lines
        lines = outline.split('\n')
        SUB_FLAG = False
        for i, line in enumerate(lines):
            # Match title, sections, subsections and their descriptions
            if line.startswith('# '):
                result["title"] = line[2:].strip()
            elif line.startswith('## '):
                SUB_FLAG = False
                result["sections"].append(line[3:].strip())
                # Extract the description in the next line
                if i + 1 < len(lines) and lines[i + 1].startswith('Description:'):
                    result["section_descriptions"].append(lines[i + 1].split('Description:', 1)[1].strip())
                    result["subsections"].append([])
                    result["subsection_descriptions"].append([])
                    result["subsection_query"].append([])
                    
            elif line.startswith('### '):
                SUB_FLAG = True
                if result["subsections"]:
                    result["subsections"][-1].append(line[4:].strip())
                    # Extract the description in the next line
                    if i + 1 < len(lines) and lines[i + 1].startswith('Description:'):
                        result["subsection_descriptions"][-1].append(lines[i + 1].split('Description:', 1)[1].strip())
                        result["subsection_query"][-1].append([])
                        
            elif line.split('. ', 1)[0].isdigit():
                if SUB_FLAG:
                    if result["subsection_query"][-1]:
                        result["subsection_query"][-1][-1].append(line.split('. ', 1)[1].strip())
        return result
    
    def process_references(self, survey):

        citations = self.extract_citations(survey)
        
        if self.args.debug:
            with open(f'{self.args.saving_path}/citations.txt', 'w') as f:
                for item in citations:
                    f.write("%s\n" % item)
                    
        return self.replace_citations_with_numbers(citations, survey)

    def generate_document(self, parsed_outline, subsection_contents):
        document = []
        
        # Append title
        title = parsed_outline['title']
        document.append(f"# {title}\n")
        
        # Iterate over sections and their content
        for i, section in enumerate(parsed_outline['sections']):
            document.append(f"## {section}\n")

            # Append subsections and their contents
            for j, subsection in enumerate(parsed_outline['subsections'][i]):
                document.append(f"### {subsection}\n")
                if i < len(subsection_contents) and j < len(subsection_contents[i]):
                    document.append(subsection_contents[i][j] + "\n")
        
        return "\n".join(document)

    def extract_citations(self, markdown_text):
        # 正则表达式匹配方括号内的内容
        pattern = re.compile(r'\[(.*?)\]')
        matches = pattern.findall(markdown_text)
        # 分割引用，处理多引用情况，并去重
        citations = list()
        for match in matches:
            # 分割各个引用并去除空格
            parts = match.split(';')
            for part in parts:
                cit = part.strip()
                if cit not in citations:
                    citations.append(cit)
        return citations

    def replace_citations_with_numbers(self, citations, markdown_text):

        index_filter = get_index_filter(self.db["rag_title4citation"].id_to_index, 
                                        self.writer_rag_results)
        ids = self.db["rag_title4citation"].retrieve_id4citation(citations, 
                                                        search_type='similarity', 
                                                        top_k=1,
                                                        **index_filter)

        citation_to_ids = {citation: idx for citation, idx in zip(citations, ids)}

        paper_infos = self.db["paper"].get_paper_info_from_ids(ids)
        temp_dic = {p['id']:p['title'] for p in paper_infos}

        titles = [temp_dic[_] for _ in tqdm(ids)]

        ids_to_titles = {idx: title for idx, title in zip(ids, titles)}
        titles_to_ids = {title: idx for idx, title in ids_to_titles.items()}

        title_to_number = {title: num+1 for  num, title in enumerate(titles)}


        title_to_number = {title: num+1 for  num, title in enumerate(title_to_number.keys())}

        number_to_title = {num: title for  title, num in title_to_number.items()}
        number_to_title_sorted =  {key: number_to_title[key] for key in sorted(number_to_title)}

        def replace_match(match):

            citation_text = match.group(1)

            individual_citations = citation_text.split(';')
            numbered_citations = []
            for citation in individual_citations:
                c = citation.strip()
                mapped_id = citation_to_ids.get(c)
                if mapped_id is None:
                    continue
                mapped_title = ids_to_titles.get(mapped_id)
                if mapped_title is None:
                    continue
                mapped_num = title_to_number.get(mapped_title)
                if mapped_num is None:
                    continue
                numbered_citations.append(str(mapped_num))

            if not numbered_citations:
                return ''
            return '[' + '; '.join(numbered_citations) + ']'
        

        updated_text = re.sub(r'\[(.*?)\]', replace_match, markdown_text)

        references_section = "\n\n## References\n\n"

        references = {num: titles_to_ids[title] for num, title in number_to_title_sorted.items()}
        for idx, title in number_to_title_sorted.items():
            t = title.replace('\n','')
            references_section += f"[{idx}] {t}\n\n"

        return updated_text + references_section, references

