import os
import json
import argparse
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str, required=True)
    parser.add_argument("--pred-file", type=str, required=True)
    parser.add_argument("--save-file", type=str, required=True)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    df = pd.read_table(args.annotation_file)

    cur_df = df.copy()
    cur_df = cur_df.drop(columns=['hint', 'category', 'source', 'image', 'comment', 'l2-category'])
    cur_df.insert(6, 'prediction', None)
    for pred in open(args.pred_file):
        pred = json.loads(pred)
        if 'text' not in pred:
            pred['text'] = pred['prediction']
        cur_df.loc[df['index'] == pred['question_id'], 'prediction'] = pred['text']

    cur_df.to_excel(args.save_file, index=False, engine='openpyxl')
