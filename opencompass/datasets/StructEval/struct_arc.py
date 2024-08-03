import csv
import json
import os.path as osp

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from ..base import BaseDataset

@LOAD_DATASET.register_module()
class StructARC_V1(BaseDataset):
    @staticmethod
    def load(path: str):
        dataset = DatasetDict()
        for split in ['test']:
            raw_data = []
            filename = osp.join(path, "{}_v1_gpt-4o-mini.json".format(split))
            with open(filename, "r", encoding="utf-8") as f:
                all_lines = json.loads(f.read())
                for line in all_lines:
                    bloom_questions = line["bloom_questions"]  
                    concept_questions = line["concept_questions"] 
                    for question in bloom_questions+concept_questions:
                        raw_data.append({
                            "input": question["question"],
                            "A": question["A"],
                            "B": question["B"],
                            "C": question["C"],
                            "D": question["D"],
                            "target": question["answer"]
                        })
                
            dataset[split] = Dataset.from_list(raw_data)
        
        for split in ["dev"]:
            raw_data = []
            filename = osp.join(path, "{}.json".format(split))
            with open(filename, "r", encoding="utf-8") as f:
                all_lines = json.loads(f.read())
                CHOICES = ["A", "B", "C", "D"]
                for line in all_lines:
                    raw_data.append({
                        "input": line["question"],
                        "A": line["choices"][0],
                        "B": line["choices"][1],
                        "C": line["choices"][2],
                        "D": line["choices"][3],
                        "target": CHOICES[line["answer"]]
                    })
            
            dataset[split] = Dataset.from_list(raw_data)
        
        return dataset