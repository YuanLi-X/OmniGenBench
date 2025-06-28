# -*- coding: utf-8 -*-
# file: SSP_Demo.py
# time: 17:04 18/04/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# 描述: 在同一视图展示 Ground Truth、ViennaRNA 与模型预测结构（三图水平排列）

import os
import time
import base64
import tempfile
from pathlib import Path
import json
import numpy as np
import gradio as gr
import RNA
from omnigenome import ModelHub

# 加载模型
print("Available files:", os.listdir('.'))
ssp_model = ModelHub.load("OmniGenome-186M-SSP")

# 临时 SVG 存储目录
TEMP_DIR = Path(tempfile.mkdtemp())
print(f"Using temporary directory: {TEMP_DIR}")


def ss_validity_loss(rna_strct: str) -> float:
    """
    Calculate a validity loss score for an RNA secondary structure in dot-bracket notation.

    The score reflects the number of unmatched parentheses normalized by
    the length of the structure excluding dots. A lower score means more valid structure.

    Args:
        rna_strct (str): RNA secondary structure string using dot-bracket notation.

    Returns:
        float: Normalized count of unmatched '(' or ')' characters indicating structural invalidity.
    """
    left = right = 0
    dots = rna_strct.count('.')
    for c in rna_strct:
        if c == '(':
            left += 1
        elif c == ')':
            if left:
                left -= 1
            else:
                right += 1
        elif c != '.':
            raise ValueError(f"Invalid char {c}")
    return (left + right) / (len(rna_strct) - dots + 1e-8)


def find_invalid_positions(struct: str) -> list:
    """
    Identify positions of invalid parentheses in a dot-bracket RNA structure.

    Finds unmatched '(' or ')' positions by using a stack to track matching pairs.

    Args:
        struct (str): RNA secondary structure string in dot-bracket notation.

    Returns:
        list: List of integer indices corresponding to unmatched parentheses.
    """
    stack, invalid = [], []
    for i, c in enumerate(struct):
        if c == '(': stack.append(i)
        elif c == ')':
            if stack:
                stack.pop()
            else:
                invalid.append(i)
    invalid.extend(stack)
    return invalid


def generate_svg_datauri(rna_seq: str, struct: str) -> str:
    """
    Generate an SVG image of the RNA secondary structure and encode it as a Base64 data URI.

    Uses ViennaRNA's SVG plotting functionality and caches SVG files in a temporary directory.
    In case of error, returns an SVG image displaying the error message.

    Args:
        rna_seq (str): RNA sequence string.
        struct (str): RNA secondary structure in dot-bracket notation.

    Returns:
        str: Base64-encoded data URI of the SVG image for embedding in HTML.
    """
    try:
        path = TEMP_DIR / f"{hash(rna_seq+struct)}.svg"
        RNA.svg_rna_plot(rna_seq, struct, str(path))
        time.sleep(0.1)
        svg_bytes = path.read_bytes()
        b64 = base64.b64encode(svg_bytes).decode('utf-8')
    except Exception as e:
        err = ('<svg xmlns="http://www.w3.org/2000/svg" width="400" height="200">'
               f'<text x="50" y="100" fill="red">Error: {e}</text></svg>')
        b64 = base64.b64encode(err.encode()).decode('utf-8')
    return f"data:image/svg+xml;base64,{b64}"


def fold(rna_seq: str, gt_struct: str):
    """
    Predict and compare RNA secondary structures from ground truth, ViennaRNA, and model predictions.

    Performs the following:
    - Uses ViennaRNA to fold the input RNA sequence.
    - Uses the loaded deep learning model (OmniGenome-186M-SSP) to predict structure.
    - Validates and fixes the predicted structure.
    - Generates SVG visualizations of all three structures.
    - Computes matching accuracy between predictions and ground truth / ViennaRNA.
    - Returns structures and combined visualization HTML.

    Args:
        rna_seq (str): RNA sequence string input by user.
        gt_struct (str): Optional ground truth RNA secondary structure.

    Returns:
        tuple: (ground_truth_str, vienna_struct, predicted_struct, stats_text, combined_svg_html)
    """
    if not rna_seq.strip():
        return "", "", "", ""
    # Ground Truth: 用户输入优先
    ground = gt_struct.strip() if gt_struct and gt_struct.strip() else ""
    gt_uri = generate_svg_datauri(rna_seq, ground) if ground else ""

    # ViennaRNA 预测
    vienna_struct, vienna_energy = RNA.fold(rna_seq)
    vienna_uri = generate_svg_datauri(rna_seq, vienna_struct)

    # 模型预测
    result = ssp_model.inference(rna_seq)
    pred = "".join(result.get('predictions', []))
    if ss_validity_loss(pred):
        for i in find_invalid_positions(pred):
            pred = pred[:i] + '.' + pred[i+1:]
    pred_uri = generate_svg_datauri(rna_seq, pred)

    # 统计信息
    match_gt = (sum(a==b for a,b in zip(ground, pred)) / len(ground)) if ground else 0
    match_vienna = sum(a==b for a,b in zip(vienna_struct, pred)) / len(vienna_struct)
    stats = (
        f"GT↔Pred Match: {match_gt:.2%}" + (" | " if ground else "") +
        f"Vienna↔Pred Match: {match_vienna:.2%}"
    )

    # 合并 HTML：三图水平排列
    combined = (
        '<div style="display:flex;justify-content:space-around;">'
        f'{f"<div><h4>Ground Truth</h4><img src=\"{gt_uri}\" style=\"max-width:100%;height:auto;\"/></div>" if ground else ""}'
        f'<div><h4>ViennaRNA</h4><img src=\"{vienna_uri}\" style=\"max-width:100%;height:auto;\"/></div>'
        f'<div><h4>Prediction</h4><img src=\"{pred_uri}\" style=\"max-width:100%;height:auto;\"/></div>'
        '</div>'
    )
    return ground, vienna_struct, pred, stats, combined


def sample_rna_sequence():
    """
    Randomly sample an RNA sequence and its ground truth structure from a test dataset file.

    Attempts to load 'toy_datasets/Archive2/test.json' and returns a random example.
    If loading fails, returns an error message and empty structure.

    Returns:
        tuple: (sampled RNA sequence string, ground truth structure string)
    """
    try:
        exs = [json.loads(l) for l in open('toy_datasets/Archive2/test.json')]
        ex = exs[np.random.randint(len(exs))]
        return ex['seq'], ex.get('label','')
    except Exception as e:
        return f"加载样本出错: {e}", ""

# Gradio UI
with gr.Blocks(css="""
.heading {text-align:center;color:#2a4365;}
.controls {display:flex;gap:10px;margin:20px 0;}
.status {padding:10px;background:#f0f4f8;border-radius:4px;white-space:pre;}
""") as demo:
    gr.Markdown("# RNA 结构预测对比", elem_classes="heading")
    with gr.Row():
        rna_input = gr.Textbox(label="RNA 序列", lines=3)
        structure_input = gr.Textbox(label="Ground Truth 结构 (可选)", lines=3)
    with gr.Row(elem_classes="controls"):
        sample_btn = gr.Button("抽取样本")
        run_btn = gr.Button("预测并对比", variant="primary")
    stats_out    = gr.Textbox(label="统计信息", interactive=False, elem_classes="status")
    gt_out       = gr.Textbox(label="Ground Truth", interactive=False)
    vienna_out   = gr.Textbox(label="ViennaRNA 结构", interactive=False)
    pred_out     = gr.Textbox(label="Prediction 结构", interactive=False)
    combined_view= gr.HTML(label="三图对比视图")

    run_btn.click(
        fold,
        inputs=[rna_input, structure_input],
        outputs=[gt_out, vienna_out, pred_out, stats_out, combined_view]
    )
    sample_btn.click(
        sample_rna_sequence,
        outputs=[rna_input, structure_input]
    )

    demo.launch(share=True)
