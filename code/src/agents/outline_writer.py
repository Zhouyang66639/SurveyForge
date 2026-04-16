import os
import numpy as np
import tiktoken
from tqdm import trange,tqdm
import time
import torch
from src.model import APIModel, LocalModel
from src.database import database
from src.utils import tokenCounter, get_index_filter
from src.prompt import ROUGH_OUTLINE_WITH_SURVEY_PROMPT, MERGING_OUTLINE_WITH_SURVEY_PROMPT, SUBSECTION_OUTLINE_WITH_SURVEY_PROMPT, EDIT_FINAL_OUTLINE_PROMPT_NEW
import random
import json
import re

class outlineWriter():
    
    def __init__(self, args, model:str, ckpt:str, api_key:str, api_url:str, database) -> None:
        self.args = args
        self.model, self.ckpt, self.api_key, self.api_url = model, ckpt, api_key, api_url 
        if self.model == 'local':
            self.api_model = LocalModel(self.ckpt)
        else:
            self.api_model = APIModel(self.model, self.api_key, self.api_url)

        self.db= database
        self.token_counter = tokenCounter()
        self.input_token_usage, self.output_token_usage = 0, 0

    def print_token_usage(self):
        print(f"OutlineWriter Input token usage: {self.input_token_usage}")
        print(f"OutlineWriter Output token usage: {self.output_token_usage}")

    def draft_outline(self, topic, reference_num = 600, chunk_size = 30000, section_num = 6):
        # Get database
        # Time Filter
        arxivid_list = list(self.db["rag_outline"].id_to_index.keys())
        arxivid_period = [arxivid for arxivid in arxivid_list if arxivid.split('.')[0] <= '2412']
        rag_outline_subset_index_filter = get_index_filter(self.db["rag_outline"].id_to_index, 
                                                           arxivid_period)
        rag_outline_subset_ids = self.db["rag_outline"].retrieve_id(topic, 
                                                                    top_k=reference_num,
                                                                    **rag_outline_subset_index_filter)
        references_infos = self.db["paper"].get_paper_info_from_ids(rag_outline_subset_ids)

        references_titles = [r['title'] for r in references_infos]
        references_date = [r['date'] for r in references_infos]
        references_abs = [r['abs'] for r in references_infos]

        references_survey_ids = self.db["survey"].get_ids_from_query(topic, num = 20, shuffle = False)
        references_survey_infos = self.db["survey"].get_paper_info_from_ids(references_survey_ids)


        references_survey_titles = [r['title'] for r in references_survey_infos]
        references_survey_date = [r['date'].split(" ")[0] for r in references_survey_infos]
        references_survey_abs = [r['abs'] for r in references_survey_infos]
        references_survey_ids = [r['id'] for r in references_survey_infos]

        if self.args.debug:
            with open(f'{self.args.saving_path}/1-Total_1500_papers.txt', 'w') as f:
                for i in references_titles:
                    f.write(i + '\n\n')
            with open(f'{self.args.saving_path}/1-Total_1500_papers_ids.txt', 'w') as f:
                for i in references_survey_ids:
                    f.write(i + '\n\n')

        abs_chunks, titles_chunks, date_chunks = self.chunking(references_abs, references_titles, references_date, chunk_size=chunk_size)
        survey_abs_chunks, survey_titles_chunks, survey_date_chunks, survey_ids_chunks = self.survey_chunking(references_survey_ids, references_survey_abs, references_survey_titles, references_survey_date, chunk_num=len(abs_chunks), ref_num=5)
        
        if self.args.debug:
            with open(f"{self.args.saving_path}/Survey_titles_rough.txt", "w") as f:
                for i in references_survey_titles:
                    f.write(i + '\n\n')

        # generate rough section-level outline

        outlines = self.generate_rough_outlines_with_survey(topic=topic, papers_chunks = abs_chunks, titles_chunks = titles_chunks, date_chunks = date_chunks, survey_ids_chunks=survey_ids_chunks, survey_abs_chunks=survey_abs_chunks, survey_titles_chunks=survey_titles_chunks, survey_date_chunks=survey_date_chunks, section_num=section_num)
        
        if self.args.debug:
            with open(f'{self.args.saving_path}/1-Chunk_outlines.json', 'w') as f:
                json.dump(outlines, f)

        # merge outline
        references_survey_ids = self.db["survey"].get_ids_from_query(topic, num = 10, shuffle = False)
        references_survey_infos = self.db["survey"].get_paper_info_from_ids(references_survey_ids)

        # choose the first 5 survey papers
        references_survey_infos = sorted(references_survey_infos, key=lambda x: x['date'].split(" ")[0], reverse=True)[:5]

        references_survey_titles = [r['title'] for r in references_survey_infos]
        references_survey_abs = [r['abs'] for r in references_survey_infos]
        references_survey_date = [r['date'].split(" ")[0] for r in references_survey_infos]
        references_survey_ids = [r['id'] for r in references_survey_infos]

        references_survey_outlines = []
        for id_tmp in references_survey_ids:
            with open(f"{self.args.survey_outline_path}/Final_outline_First/{id_tmp}.md", "r") as f:
                references_survey_outlines.append(f.read().strip())

        if self.args.debug:
            with open(f"{self.args.saving_path}/Survey_titles_high.txt", "w") as f:
                for i in references_survey_titles:
                    f.write(i + '\n\n')
                
        section_outline = self.merge_outlines_with_survey(topic=topic, outlines=outlines, references_survey_titles=references_survey_titles, references_survey_abs=references_survey_abs,references_survey_date=references_survey_date, references_survey_outlines=references_survey_outlines, section_num=section_num)
        
        if self.args.debug:
            with open(f"{self.args.saving_path}/2-Merged_outlines.txt", "w") as f:
                f.write(section_outline + '\n\n')
        
        # generate subsection-level outline
        self.print_token_usage()
        references_survey_outlines = []
        for id_tmp in references_survey_ids:
            with open(f"{self.args.survey_outline_path}/Final_outline/{id_tmp}.md", "r") as f:
                references_survey_outlines.append(f.read().strip())

        if self.args.debug:
            with open(f"{self.args.saving_path}/2-Merged_outlines.txt", "r") as f:
                section_outline = f.read().strip()

        subsection_outlines = self.generate_subsection_outlines_with_survey(topic=topic, section_outline= section_outline,rag_num= 50, references_survey_titles=references_survey_titles, references_survey_abs=references_survey_abs,references_survey_date=references_survey_date, references_survey_outlines=references_survey_outlines)
        
        # Process introduction and conclusion
        subsection_outlines.insert(0, [])
        subsection_outlines.append([])

        if self.args.debug:
            with open(f"{self.args.saving_path}/3-Merged_Sub_outline_wo_process.txt", "w") as f:
                f.write(merged_outline + '\n\n')
        
        merged_outline = self.process_outlines_points(section_outline, subsection_outlines)

        if self.args.debug:
            with open(f"{self.args.saving_path}/3-Merged_Sub_outline.txt", "w") as f:
                f.write(merged_outline + '\n\n')

        final_outline = merged_outline

        return final_outline

    def compute_price(self):
        return self.token_counter.compute_price(input_tokens=self.input_token_usage, output_tokens=self.output_token_usage, model=self.model)

    def generate_rough_outlines_with_survey(self, topic, papers_chunks, titles_chunks, date_chunks, survey_ids_chunks, survey_abs_chunks, survey_titles_chunks, survey_date_chunks, section_num = 8):

        prompts = []
        for idx in trange(len(papers_chunks)):
            titles = titles_chunks[idx]
            papers = papers_chunks[idx]
            date = date_chunks[idx]
            paper_texts = '' 
            for i, t, p, d in zip(range(len(papers)), titles, papers, date):
                paper_texts += f'---\npaper_title: {t}\n\npublish_date: {d}\n\npaper_abstract:\n\n{p}\n'
            paper_texts+='---\n'

            survey_titles = survey_titles_chunks[idx]
            survey_papers = survey_abs_chunks[idx]
            survey_date = survey_date_chunks[idx]
            survey_ids = survey_ids_chunks[idx]
            
            survey_paper_texts = '' 
            for i, id, t, p, d in zip(range(len(survey_papers)), survey_ids, survey_titles, survey_papers, survey_date):
                with open(f"{self.args.survey_outline_path}/Final_outline_First/{id}.md", "r") as f:
                    first_o = f.read().strip()
                survey_paper_texts += f'---\npaper_title: {t}\n\npublish_date: {d}\n\npaper_abstract:\n\n{p}\n\npaper_first_outline:\n\n{first_o}\n'
            survey_paper_texts+='---\n'

            prompt = self.__generate_prompt(ROUGH_OUTLINE_WITH_SURVEY_PROMPT, paras={'SURVEY LIST': survey_paper_texts, 'PAPER LIST': paper_texts, 'TOPIC': topic, 'SECTION NUM': str(section_num)})
            prompts.append(prompt)
        self.input_token_usage += self.token_counter.num_tokens_from_list_string(prompts)
        print("rouge outline input token :%d" %self.token_counter.num_tokens_from_list_string(prompts))
        start = time.time()
        outlines = self.api_model.batch_chat(text_batch=prompts, temperature=1)
        end = time.time()
        period = end - start
        with open(f"{self.args.saving_path}/time_cost.log", "a") as f:
            f.write(f"Outline API: {period}\n")

        self.output_token_usage += self.token_counter.num_tokens_from_list_string(outlines)
        return outlines
    
    def merge_outlines_with_survey(self, topic, outlines, references_survey_titles, references_survey_abs,references_survey_date, references_survey_outlines, section_num):

        outline_texts = '' 
        for i, o in zip(range(len(outlines)), outlines):
            outline_texts += f'---\noutline_id: {i}\n\noutline_content:\n\n{o}\n'
        outline_texts+='---\n'

        survey_paper_texts = '' 
        for i, t, p, d, o in zip(range(len(references_survey_abs)), references_survey_titles, references_survey_abs, references_survey_date, references_survey_outlines):
            survey_paper_texts += f'---\npaper_title: {t}\n\npublish_date: {d}\n\npaper_abstract:\n\n{p}\n\npaper_first_outline:\n\n{o}\n'
        survey_paper_texts+='---\n'

        prompt = self.__generate_prompt(MERGING_OUTLINE_WITH_SURVEY_PROMPT, paras={'SURVEY LIST':survey_paper_texts, 'OUTLINE LIST': outline_texts, 'TOPIC':topic, 'SECTION NUM':str(section_num)})
        self.input_token_usage += self.token_counter.num_tokens_from_string(prompt)
        outline = self.api_model.chat(prompt, temperature=1)
        self.output_token_usage += self.token_counter.num_tokens_from_string(outline)
        if self.args.debug:
            with open(f"{self.args.saving_path}/2-Merged_outlines_Input.txt", "w") as f:
                f.write(prompt + '\n\n')
        return outline.replace('<format>\n','').replace('</format>','')

    def generate_subsection_outlines_with_survey(self, topic, section_outline, rag_num, references_survey_titles, references_survey_abs,references_survey_date, references_survey_outlines):

        survey_title, survey_sections, survey_section_descriptions = self.extract_title_sections_descriptions(section_outline)

        prompts = []

        for section_name, section_description in zip(survey_sections[1:-1], survey_section_descriptions[1:-1]):
            if isinstance(section_description, list):
                references_ids = []
                for des_tmp in section_description:
                    # print(f"{section_name}:" + des_tmp)
                    references_ids_tmp = self.db["rag_suboutline"].retrieve_id([f"{topic}:" + des_tmp], 
                                                                        search_type='similarity', 
                                                                        rerank='raw', 
                                                                        top_k=int(rag_num // len(section_description) * 5))
                    references_ids.extend(references_ids_tmp)
            else:
                section_description = [section_description]
                references_ids = self.db["rag_suboutline"].retrieve_id(section_description, 
                                                                    search_type='similarity', 
                                                                    rerank='raw', 
                                                                    top_k=rag_num)

            references_infos = self.db["paper"].get_paper_info_from_ids(references_ids)
            
            # remove "Survey" papers
            references_infos = [r for r in references_infos if 'survey' not in r['title'].lower()]
            if len(references_infos) > rag_num:
                references_infos = random.sample(references_infos, rag_num)

            references_titles = [r['title'] for r in references_infos]
            references_papers = [r['abs'] for r in references_infos]
            references_date = [r['date'] for r in references_infos]
            if self.args.debug:
                with open(f'{self.args.saving_path}/3-Total_subsection_papers.txt', 'a+') as f:
                    f.write(f'{section_name}\n{section_description}\n\n')
                    for i in references_titles:
                        f.write(i + '\n\n')

            paper_texts = '' 
            for i, t, p, d in zip(range(len(references_papers)), references_titles, references_papers, references_date):
                paper_texts += f'---\npaper_title: {t}\n\npublish_date: {d}\n\npaper_content:\n\n{p}\n'
            paper_texts+='---\n'

            survey_paper_texts = '' 
            for i, t, p, d, o in zip(range(len(references_survey_abs)), references_survey_titles, references_survey_abs, references_survey_date, references_survey_outlines):
                survey_paper_texts += f'---\npaper_title: {t}\n\npublish_date: {d}\n\npaper_abstract:\n\n{p}\n\npaper_outline:\n\n{o}\n'
            survey_paper_texts+='---\n'

            prompt = self.__generate_prompt(SUBSECTION_OUTLINE_WITH_SURVEY_PROMPT, paras={'RAG NUM':str(rag_num),  'OVERALL OUTLINE': section_outline, 'OVERALL OUTLINE': section_outline,'SECTION NAME': section_name,\
                                                                          'SECTION DESCRIPTION':'\n'.join(section_description),'TOPIC':topic,'PAPER LIST':paper_texts, 'SURVEY LIST':survey_paper_texts})
            prompts.append(prompt)

        self.input_token_usage += self.token_counter.num_tokens_from_list_string(prompts)

        start = time.time()
        sub_outlines = self.api_model.batch_chat(prompts, temperature=1)
        end = time.time()
        period = end - start
        with open(f"{self.args.saving_path}/time_cost.log", "a") as f:
            f.write(f"SubOutline API: {period}\n")
        # print(f"##########SubOutline API Time taken#########: {period}")

        self.output_token_usage += self.token_counter.num_tokens_from_list_string(sub_outlines)
        if self.args.debug:
            with open(f"{self.args.saving_path}/3-Merged_Sub_outline_Input.txt", "w") as f:
                f.write(prompt + '\n\n')
        return sub_outlines


    def edit_final_outline(self, outline):
        prompt = self.__generate_prompt(EDIT_FINAL_OUTLINE_PROMPT_NEW, paras={'OVERALL OUTLINE': outline})
        self.input_token_usage += self.token_counter.num_tokens_from_string(prompt)
        outline = self.api_model.chat(prompt, temperature=1)
        self.output_token_usage += self.token_counter.num_tokens_from_string(outline)
        return outline.replace('<format>\n','').replace('</format>','')
 
    def __generate_prompt(self, template, paras):
        prompt = template
        for k in paras.keys():
            prompt = prompt.replace(f'[{k}]', paras[k])
        return prompt
    
    def extract_title_sections_descriptions(self, outline):
        outline = (outline or "").replace("<format>", "").replace("</format>", "").strip()

        title_match = re.search(r'(?im)^\s*Title\s*[:：]\s*(.+?)\s*$', outline)
        if title_match:
            title = title_match.group(1).strip()
        else:
            md_title_match = re.search(r'(?m)^\s*#\s+(.+?)\s*$', outline)
            title = md_title_match.group(1).strip() if md_title_match else "Generated Survey"

        sections, descriptions = [], []
        section_head_pattern = re.compile(r'(?im)^\s*Section\s*(\d+)\s*[:：\-]\s*(.+?)\s*$')
        section_matches = list(section_head_pattern.finditer(outline))
        if section_matches:
            for idx, m in enumerate(section_matches):
                section_idx = int(m.group(1))
                section_title = m.group(2).strip()
                start = m.end()
                end = section_matches[idx + 1].start() if idx + 1 < len(section_matches) else len(outline)
                block = outline[start:end]

                desc_lines = []
                d = re.search(
                    rf'(?im)^\s*Description\s*{section_idx}\s*[:：]\s*(.+?)\s*$',
                    block,
                )
                if not d:
                    d = re.search(r'(?im)^\s*Description(?:\s*\d+)?\s*[:：]\s*(.+?)\s*$', block)
                if d:
                    first_desc = d.group(1).strip()
                    if first_desc:
                        desc_lines.append(first_desc)

                bullets = re.findall(r'(?m)^\s*\d+\.\s*(.+?)\s*$', block)
                desc_lines.extend([b.strip() for b in bullets if b.strip()])
                if not desc_lines:
                    desc_lines = [section_title]

                sections.append(section_title)
                descriptions.append(desc_lines)
            return title, sections, descriptions

        # markdown fallback
        md_head_pattern = re.compile(r'(?im)^\s*##\s+(.+?)\s*$')
        md_matches = list(md_head_pattern.finditer(outline))
        for idx, m in enumerate(md_matches):
            section_title = re.sub(r'^\d+(?:\.\d+)*\s*', '', m.group(1)).strip()
            start = m.end()
            end = md_matches[idx + 1].start() if idx + 1 < len(md_matches) else len(outline)
            block = outline[start:end]
            desc_lines = []
            d = re.search(r'(?im)^\s*Description(?:\s*\d+)?\s*[:：]\s*(.+?)\s*$', block)
            if d:
                desc_lines.append(d.group(1).strip())
            bullets = re.findall(r'(?m)^\s*\d+\.\s*(.+?)\s*$', block)
            desc_lines.extend([b.strip() for b in bullets if b.strip()])
            if not desc_lines:
                desc_lines = [section_title]
            sections.append(section_title)
            descriptions.append(desc_lines)

        return title, sections, descriptions
    
    def extract_subsections_subdescriptions_points(self, outline):
        def extract_desc_from_block(block_text, subsection_idx):
            desc_lines = []
            # Prefer exact "Description {idx}: ..."
            m = re.search(
                rf'(?im)^\s*Description\s*{subsection_idx}\s*[:：]\s*(.+?)\s*$',
                block_text,
            )
            if not m:
                # Fallback to generic "Description: ..."
                m = re.search(r'(?im)^\s*Description(?:\s*\d+)?\s*[:：]\s*(.+?)\s*$', block_text)
            if m:
                first_desc = m.group(1).strip()
                if first_desc:
                    desc_lines.append(first_desc)

            bullets = re.findall(r'(?m)^\s*\d+\.\s*(.+?)\s*$', block_text)
            desc_lines.extend([b.strip() for b in bullets if b.strip()])
            if not desc_lines:
                # Last fallback: keep the first non-empty line in the block.
                for line in block_text.splitlines():
                    line = line.strip()
                    if line and not line.lower().startswith("subsection"):
                        desc_lines.append(line)
                        break
            return desc_lines

        outline = (outline or "").replace("<format>", "").replace("</format>", "").strip()
        if not outline:
            return [], []

        subsections, subdescriptions = [], []

        # Pattern A: "Subsection i: Title" (allow ":" / "：" / "-")
        subsection_head_pattern = re.compile(
            r'(?im)^\s*(?:#+\s*)?Subsection\s*(\d+)\s*[:：\-]\s*(.+?)\s*$'
        )
        matches = list(subsection_head_pattern.finditer(outline))
        if matches:
            for idx, m in enumerate(matches):
                start = m.end()
                end = matches[idx + 1].start() if idx + 1 < len(matches) else len(outline)
                block = outline[start:end]
                subsection_idx = int(m.group(1))
                title = m.group(2).strip()
                if title:
                    subsections.append(title)
                    subdescriptions.append(extract_desc_from_block(block, subsection_idx))
            if subsections:
                return subsections, subdescriptions

        # Pattern B fallback: markdown heading "### 2.1 xxx" / "### xxx"
        md_head_pattern = re.compile(r'(?im)^\s*###\s+(.+?)\s*$')
        md_matches = list(md_head_pattern.finditer(outline))
        if md_matches:
            for idx, m in enumerate(md_matches):
                title = re.sub(r'^\d+(?:\.\d+)*\s*', '', m.group(1)).strip()
                start = m.end()
                end = md_matches[idx + 1].start() if idx + 1 < len(md_matches) else len(outline)
                block = outline[start:end]
                if title:
                    subsections.append(title)
                    subdescriptions.append(extract_desc_from_block(block, idx + 1))
            if subsections:
                return subsections, subdescriptions

        # Keep running without crashing if model output is off-format.
        return [], []
    
    def chunking(self, papers, titles, dates, chunk_size = 14000):
        paper_chunks, title_chunks, date_chunks = [], [], []
        total_length = self.token_counter.num_tokens_from_list_string(papers)
        num_of_chunks = int(total_length / chunk_size) + 1
        avg_len = int(total_length / num_of_chunks) + 1
        split_points = []
        l = 0
        for j in range(len(papers)):
            l += self.token_counter.num_tokens_from_string(papers[j])
            if l > avg_len:
                l = 0
                split_points.append(j)
                continue
        start = 0
        for point in split_points:
            paper_chunks.append(papers[start:point])
            title_chunks.append(titles[start:point])
            date_chunks.append(dates[start:point])
            start = point
        paper_chunks.append(papers[start:])
        title_chunks.append(titles[start:])
        date_chunks.append(dates[start:])
        return paper_chunks, title_chunks, date_chunks

    def survey_chunking(self, ids, papers, titles, dates, chunk_num=10, ref_num=10):
        assert len(papers) == len(titles), "papers and titles must have the same length"
        
        paper_chunks, title_chunks, date_chunks, id_chunks = [], [], [], []
        total_papers = len(papers)
        
        all_indices = list(range(total_papers))
        
        for i in range(chunk_num):
            chunk_indices = []
            if all_indices:
                num_to_select = ref_num
                selected = random.sample(all_indices, num_to_select)
                chunk_indices.extend(selected)
                for idx in selected:
                    all_indices.remove(idx)
            
            # random select
            while len(chunk_indices) < ref_num:
                chunk_indices.append(random.randint(0, total_papers - 1))
            
            # make chunk of paper and title
            paper_chunk = [papers[idx] for idx in chunk_indices]
            title_chunk = [titles[idx] for idx in chunk_indices]
            date_chunk = [dates[idx] for idx in chunk_indices]
            id_chunk = [ids[idx] for idx in chunk_indices]
            
            paper_chunks.append(paper_chunk)
            title_chunks.append(title_chunk)
            date_chunks.append(date_chunk)
            id_chunks.append(id_chunk)
        
        return paper_chunks, title_chunks, date_chunks, id_chunks
    
    def process_outlines_points(self, section_outline, sub_outlines):
        res = ''
        survey_title, survey_sections, survey_section_descriptions = self.extract_title_sections_descriptions(outline=section_outline)
        res += f'# {survey_title}\n\n'
        for i in range(len(survey_sections)):
            section = survey_sections[i]
            section_desc_item = survey_section_descriptions[i]
            if isinstance(section_desc_item, list):
                description = "\n".join(section_desc_item)
            else:
                description = str(section_desc_item)
            res += f'## {i+1} {section}\nDescription: {description}\n\n'
            raw_sub_outline = sub_outlines[i] if i < len(sub_outlines) else ""
            subsections, subsection_descriptions = self.extract_subsections_subdescriptions_points(raw_sub_outline)
            for j in range(len(subsections)):
                subsection = subsections[j]
                sub_desc_item = subsection_descriptions[j] if j < len(subsection_descriptions) else []
                if isinstance(sub_desc_item, list):
                    sub_description = "\n".join(sub_desc_item)
                else:
                    sub_description = str(sub_desc_item)
                res += f'### {i+1}.{j+1} {subsection}\nDescription: {sub_description}\n\n'
        return res

