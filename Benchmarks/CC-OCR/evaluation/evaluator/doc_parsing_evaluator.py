import nltk
import re
from tqdm import tqdm
from collections import deque
from apted.helpers import Tree
from apted import APTED, Config
from concurrent.futures import ProcessPoolExecutor, as_completed

# local import
from .common import BaseMetric, convert_to_halfwidth


# 移除指定的LaTeX命令
patterns = [
    r'\\documentclass\{.*?\}',
    r'\\usepackage\[.*?\]\{.*?\}',
    r'\\usepackage\{.*?\}',
    r'\\geometry\{.*?\}',
    r'\\begin\{document\}',
    r'\\end\{document\}',
    r'\\noindent'
]


def extract_and_clean_tables(text):
    if '</table>' not in text:
        text += '</table>'

    # Use regular expressions to find all table parts
    tables = re.findall(r'<table.*?>.*?</table>', text, re.DOTALL)

    clean_tables = []
    for table in tables:
        # Remove extra information from the table header, keeping only <table>...</table>.
        table_content = re.sub(r'<table.*?>', '<table>', table)

        # Remove line breaks and excessive spaces between tags without affecting the information inside the tags, such as attributes.
        table_content = re.sub(r'>\s+<', '><', table_content)

        # Eliminate line breaks and redundant spaces within tags (i.e., between '>' and '<').
        table_content = re.sub(r'>(.*?)<', lambda m: '>' + m.group(1).replace('\n', '').replace(' ', '') + '<', table_content, flags=re.DOTALL)

        # Flatten the table content by removing all line breaks.
        table_content = table_content.replace('\n', '').strip()
        clean_tables.append(table_content)

    flat_table = ''.join(clean_tables)
    return flat_table


class TableTree(Tree):
    """
    # Copyright 2020 IBM
    # Author: peter.zhong@au1.ibm.com
    # License:  Apache 2.0 License.
    """
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        """Show tree using brackets notation"""
        if self.tag == "td":
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % (
                self.tag,
                self.colspan,
                self.rowspan,
                self.content,
            )
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)


class CustomConfig(Config):
    """
    # Copyright 2020 IBM
    # Author: peter.zhong@au1.ibm.com
    # License:  Apache 2.0 License.
    """
    def rename(self, node1, node2):
        """Compares attributes of trees"""
        # print(node1.tag)
        if (
            (node1.tag != node2.tag)
            or (node1.colspan != node2.colspan)
            or (node1.rowspan != node2.rowspan)
        ):
            return 1.0
        if node1.tag == "td":
            if node1.content or node2.content:
                return nltk.edit_distance(node1.content, node2.content) / max(len(node1.content), len(node2.content))
        return 0.0


class TEDS(object):
    """Tree Edit Distance basead Similarity
    # Copyright 2020 IBM
    # Author: peter.zhong@au1.ibm.com
    # License:  Apache 2.0 License.
    """
    def __init__(self, structure_only=False, n_jobs=1, ignore_nodes=None):
        assert isinstance(n_jobs, int) and (
            n_jobs >= 1
        ), "n_jobs must be an integer greather than 1"
        self.structure_only = structure_only
        self.n_jobs = n_jobs
        self.ignore_nodes = ignore_nodes
        self.__tokens__ = []

    def tokenize(self, node):
        """Tokenizes table cells"""
        self.__tokens__.append("<%s>" % node.tag)
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in node.getchildren():
            self.tokenize(n)
        if node.tag != "unk":
            self.__tokens__.append("</%s>" % node.tag)
        if node.tag != "td" and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        """Converts HTML tree to the format required by apted"""
        global __tokens__
        if node.tag == "td":
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            new_node = TableTree(
                node.tag,
                int(node.attrib.get("colspan", "1")),
                int(node.attrib.get("rowspan", "1")),
                cell,
                *deque(),
            )
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != "td":
            for n in node.getchildren():
                self.load_html_tree(n, new_node)
        if parent is None:
            return new_node

    def evaluate(self, pred, true):
        """Computes TEDS score between the prediction and the ground truth of a
        given sample
        """
        # try_import("lxml")
        from lxml import etree, html
        if (not pred) or (not true):
            return 0.0

        parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
        pred = html.fromstring(pred, parser=parser)
        true = html.fromstring(true, parser=parser)
        if pred.xpath("body/table") and true.xpath("body/table"):
            pred = pred.xpath("body/table")[0]
            true = true.xpath("body/table")[0]
            if self.ignore_nodes:
                etree.strip_tags(pred, *self.ignore_nodes)
                etree.strip_tags(true, *self.ignore_nodes)
            n_nodes_pred = len(pred.xpath(".//*"))
            n_nodes_true = len(true.xpath(".//*"))
            n_nodes = max(n_nodes_pred, n_nodes_true)
            tree_pred = self.load_html_tree(pred)
            tree_true = self.load_html_tree(true)
            distance = APTED(
                tree_pred, tree_true, CustomConfig()
            ).compute_edit_distance()
            return 1.0 - (float(distance) / n_nodes)
        else:
            return 0.0


class ParsingEvaluator(BaseMetric):
    def response_post_func(self, response_text, **kwargs):
        return response_text

    def evaluate(self, response_info, gt_info, **kwargs):
        op = kwargs['op']
        if op == 'doc':
            score = self.eval_doc(response_info, gt_info)
        elif op == 'table':
            score = self.eval_table(response_info, gt_info)
        elif op in ['molecular', "formula"]:
            score = self.eval_formula(response_info, gt_info, op_name=op)
        else:
            raise ValueError(f'doc parsing unsupported op: {op}')

        # summary info
        eval_info = {"summary": {"score": score}}
        return eval_info

    def parallel_process(self, response_info, gt_info, eval_func, op_name='formula'):
        results = []
        with ProcessPoolExecutor() as executor:
            futures = []
            for img_name, gt in gt_info.items():
                if img_name not in response_info:
                    results.append(0)
                    continue

                pred = response_info[img_name]
                futures.append(executor.submit(eval_func, gt, pred, op_name=op_name))

            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
                results.append(future.result())

        score = sum(results) / len(results) if results else 0
        return score

    def evaluate_single_doc_sample(self, gt, pred, op_name=''):
        for pattern in patterns:
            pred = re.sub(pattern, '', pred)

        try:
            pattern = r'```latex(.+?)```'
            pred = re.search(pattern, pred, re.DOTALL).group(1)
        except:
            if '```latex' in pred:
                pred = pred.split('```latex')[1]

        pred = pred.replace(' ', '').replace('\n', '')
        gt = gt.replace(' ', '').replace('\n', '')

        edit_dist = nltk.edit_distance(pred, gt) / max(len(pred), len(gt))
        return 1 - edit_dist

    def eval_doc(self, response_info, gt_info):
        score = self.parallel_process(response_info, gt_info, self.evaluate_single_doc_sample)
        return score

    def evaluate_single_table_sample(self, gt, pred, op_name=''):
        teds = TEDS(structure_only=False, n_jobs=1)
        try:
            pattern = r'```html(.+?)```'
            pred = re.search(pattern, pred, re.DOTALL).group(1)
        except:
            if '```html' in pred:
                pred = pred.split('```html')[1]

        pred = extract_and_clean_tables(pred)
        pred = convert_to_halfwidth(pred)
        gt = extract_and_clean_tables(gt)
        gt = convert_to_halfwidth(gt)

        pred_html = '<html><body>{}</body></html>'.format(pred)
        gt_html = '<html><body>{}</body></html>'.format(gt)
        return teds.evaluate(pred_html, gt_html)

    def eval_table(self, response_info, gt_info):
        score = self.parallel_process(response_info, gt_info, self.evaluate_single_table_sample)
        return score

    def evaluate_single_formula_sample(self, gt, pred, op_name='formula'):
        if op_name == 'formula':
            pred = pred.replace("\n", " ").replace("```latex", "").replace("```", "").replace("\t", " ").replace(" ", "")
            gt = gt.replace(" ", "")
        elif op_name == 'molecular':
            pred = pred.replace("\n", "").replace(" ", "").replace("<smiles>", "").replace("</smiles>", "")
            gt = gt.replace(" ", "")
        edit_dist = nltk.edit_distance(pred, gt) / max(len(pred), len(gt))
        return 1 - edit_dist

    def eval_formula(self, response_info, gt_info, op_name='formula'):
        score = self.parallel_process(response_info, gt_info, self.evaluate_single_formula_sample, op_name=op_name)
        return score


if __name__ == '__main__':
    pass
