import math
from bs4.element import Comment
from bs4 import BeautifulSoup
import bs4


def get_elements(html):
    soup = BeautifulSoup(html, "html.parser")
    elements = soup.body.find_all(class_=lambda x: x != "cls-1")
    return elements


def get_all_element_lis(html):
    res_lis = []
    for element in html.descendants:
        res_lis.append(element)
    for res in res_lis:
        if type(res) != bs4.element.NavigableString and type(res) != bs4.element.Tag:
            res.extract()
    return res_lis


def html_to_text_list(h, ele_lis):
    tag_num, text_list = 0, []
    for element in ele_lis:
        if (type(element) == bs4.element.NavigableString) and (element.strip()):
            text_list.append(element.strip())
        if type(element) == bs4.element.Tag:
            tag_num += 1
    return text_list, tag_num + 2


def get_e_id_to_text_dict(html, ele_lis):
    res_dict = {}
    for element in ele_lis:
        if type(element) == bs4.element.Tag:
            if element.attrs is not None and "class" in element.attrs:
                res_dict[element["class"][0]] = element.get_text()
    return res_dict


def get_e_id_to_t_id_dict(html, ele_lis):
    res_dict = {}
    t_id = 0
    for element in ele_lis:
        if type(element) == bs4.element.NavigableString and element.strip():
            t_id += 1
        if type(element) == bs4.element.Tag:
            if element.attrs is not None and "class" in element.attrs:
                res_dict[element.attrs["class"][0]] = t_id
    return res_dict


def get_doc_tokens(page_text):
    doc_tokens = []
    prev_is_whitespace = True
    for c in page_text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
    return doc_tokens


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def subtoken_tag_offset(html, s_tok, orig_to_tok_index, ele_lis):
    w_t, t_w = word_tag_offset(html, ele_lis)
    s_t = [] 
    unique_tids = set() 
    for i in range(len(s_tok)):
        s_t.append(w_t[s_tok[i]])
        unique_tids.add(w_t[s_tok[i]])

    for ele in t_w.keys():
        t_w[ele]["start"] = orig_to_tok_index[t_w[ele]["start"]]
        if t_w[ele]["end"] + 1 < len(orig_to_tok_index):
            t_w[ele]["end"] = orig_to_tok_index[t_w[ele]["end"] + 1] - 1 
        else:
            t_w[ele]["end"] = len(s_tok) - 1
    return s_t, unique_tids, t_w


def word_tag_offset(html, ele_lis):
    cnt, w_t, t_w, tags, tags_tids = (
        0,
        [],
        {},
        [],
        [],
    )
    for element in ele_lis:
        if type(element) == bs4.element.Tag:

            doc_tokens = []
            for string in element.strings:
                if string.strip():
                    doc_tokens.extend(get_doc_tokens(string.strip()))

            if len(doc_tokens) == 0:
                continue

            start = cnt
            end = cnt + len(doc_tokens) - 1

            if end < start:
                start = end

            t_w[element["class"][0]] = {"start": start, "end": end}
            tags_tids.append(element["class"][0])

        elif type(element) == bs4.element.NavigableString and element.strip():
            doc_tokens = get_doc_tokens(element.strip())
            tid = element.parent["class"][0]
            for _ in doc_tokens:
                w_t.append(tid)
                cnt += 1
            assert cnt == len(w_t)

    return w_t, t_w


def get_xpath_and_treeid4tokens(html_code, unique_tids, max_depth):
    unknown_tag_id = len(tags_dict)
    pad_tag_id = unknown_tag_id + 1
    max_width = 1000
    width_pad_id = 1001

    pad_x_tag_seq = [pad_tag_id] * max_depth
    pad_x_subs_seq = [width_pad_id] * max_depth

    def xpath_soup(element):

        xpath_tags = []
        xpath_subscripts = []
        tree_index = []
        child = element if element.name else element.parent
        for parent in child.parents: 
            siblings = parent.find_all(child.name, recursive=False)
            para_siblings = parent.find_all(True, recursive=False)
            xpath_tags.append(child.name)
            xpath_subscripts.append(
                0
                if 1 == len(siblings)
                else next(i for i, s in enumerate(siblings, 1) if s is child)
            )

            tree_index.append(
                next(i for i, s in enumerate(para_siblings, 0) if s is child)
            )
            child = parent
        xpath_tags.reverse()
        xpath_subscripts.reverse()
        tree_index.reverse()
        return xpath_tags, xpath_subscripts, tree_index

    xpath_tag_map = {}
    xpath_subs_map = {}

    for tid in unique_tids:
        element = html_code.find(attrs={"class": tid})
        if element is None:
            xpath_tags = pad_x_tag_seq
            xpath_subscripts = pad_x_subs_seq

            xpath_tag_map[tid] = xpath_tags
            xpath_subs_map[tid] = xpath_subscripts
            continue

        xpath_tags, xpath_subscripts, tree_index = xpath_soup(
            element
        )  

        assert len(xpath_tags) == len(xpath_subscripts)
        assert len(xpath_tags) == len(tree_index)

        if len(xpath_tags) > max_depth:
            xpath_tags = xpath_tags[-max_depth:]
            xpath_subscripts = xpath_subscripts[-max_depth:]

        xpath_tags = [tags_dict.get(name, unknown_tag_id) for name in xpath_tags]
        xpath_subscripts = [min(i, max_width) for i in xpath_subscripts]

        xpath_tags += [pad_tag_id] * (max_depth - len(xpath_tags))
        xpath_subscripts += [width_pad_id] * (max_depth - len(xpath_subscripts))

        xpath_tag_map[tid] = xpath_tags
        xpath_subs_map[tid] = xpath_subscripts

    return xpath_tag_map, xpath_subs_map


tags_dict = {
    "a": 0,
    "abbr": 1,
    "acronym": 2,
    "address": 3,
    "altGlyph": 4,
    "altGlyphDef": 5,
    "altGlyphItem": 6,
    "animate": 7,
    "animateColor": 8,
    "animateMotion": 9,
    "animateTransform": 10,
    "applet": 11,
    "area": 12,
    "article": 13,
    "aside": 14,
    "audio": 15,
    "b": 16,
    "base": 17,
    "basefont": 18,
    "bdi": 19,
    "bdo": 20,
    "bgsound": 21,
    "big": 22,
    "blink": 23,
    "blockquote": 24,
    "body": 25,
    "br": 26,
    "button": 27,
    "canvas": 28,
    "caption": 29,
    "center": 30,
    "circle": 31,
    "cite": 32,
    "clipPath": 33,
    "code": 34,
    "col": 35,
    "colgroup": 36,
    "color-profile": 37,
    "content": 38,
    "cursor": 39,
    "data": 40,
    "datalist": 41,
    "dd": 42,
    "defs": 43,
    "del": 44,
    "desc": 45,
    "details": 46,
    "dfn": 47,
    "dialog": 48,
    "dir": 49,
    "div": 50,
    "dl": 51,
    "dt": 52,
    "ellipse": 53,
    "em": 54,
    "embed": 55,
    "feBlend": 56,
    "feColorMatrix": 57,
    "feComponentTransfer": 58,
    "feComposite": 59,
    "feConvolveMatrix": 60,
    "feDiffuseLighting": 61,
    "feDisplacementMap": 62,
    "feDistantLight": 63,
    "feFlood": 64,
    "feFuncA": 65,
    "feFuncB": 66,
    "feFuncG": 67,
    "feFuncR": 68,
    "feGaussianBlur": 69,
    "feImage": 70,
    "feMerge": 71,
    "feMergeNode": 72,
    "feMorphology": 73,
    "feOffset": 74,
    "fePointLight": 75,
    "feSpecularLighting": 76,
    "feSpotLight": 77,
    "feTile": 78,
    "feTurbulence": 79,
    "fieldset": 80,
    "figcaption": 81,
    "figure": 82,
    "filter": 83,
    "font-face-format": 84,
    "font-face-name": 85,
    "font-face-src": 86,
    "font-face-uri": 87,
    "font-face": 88,
    "font": 89,
    "footer": 90,
    "foreignObject": 91,
    "form": 92,
    "frame": 93,
    "frameset": 94,
    "g": 95,
    "glyph": 96,
    "glyphRef": 97,
    "h1": 98,
    "h2": 99,
    "h3": 100,
    "h4": 101,
    "h5": 102,
    "h6": 103,
    "head": 104,
    "header": 105,
    "hgroup": 106,
    "hkern": 107,
    "hr": 108,
    "html": 109,
    "i": 110,
    "iframe": 111,
    "image": 112,
    "img": 113,
    "input": 114,
    "ins": 115,
    "kbd": 116,
    "keygen": 117,
    "label": 118,
    "legend": 119,
    "li": 120,
    "line": 121,
    "linearGradient": 122,
    "link": 123,
    "main": 124,
    "map": 125,
    "mark": 126,
    "marker": 127,
    "marquee": 128,
    "mask": 129,
    "math": 130,
    "menu": 131,
    "menuitem": 132,
    "meta": 133,
    "metadata": 134,
    "meter": 135,
    "missing-glyph": 136,
    "mpath": 137,
    "nav": 138,
    "nobr": 139,
    "noembed": 140,
    "noframes": 141,
    "noscript": 142,
    "object": 143,
    "ol": 144,
    "optgroup": 145,
    "option": 146,
    "output": 147,
    "p": 148,
    "param": 149,
    "path": 150,
    "pattern": 151,
    "picture": 152,
    "plaintext": 153,
    "polygon": 154,
    "polyline": 155,
    "portal": 156,
    "pre": 157,
    "progress": 158,
    "q": 159,
    "radialGradient": 160,
    "rb": 161,
    "rect": 162,
    "rp": 163,
    "rt": 164,
    "rtc": 165,
    "ruby": 166,
    "s": 167,
    "samp": 168,
    "script": 169,
    "section": 170,
    "select": 171,
    "set": 172,
    "shadow": 173,
    "slot": 174,
    "small": 175,
    "source": 176,
    "spacer": 177,
    "span": 178,
    "stop": 179,
    "strike": 180,
    "strong": 181,
    "style": 182,
    "sub": 183,
    "summary": 184,
    "sup": 185,
    "svg": 186,
    "switch": 187,
    "symbol": 188,
    "table": 189,
    "tbody": 190,
    "td": 191,
    "template": 192,
    "text": 193,
    "textPath": 194,
    "textarea": 195,
    "tfoot": 196,
    "th": 197,
    "thead": 198,
    "time": 199,
    "title": 200,
    "tr": 201,
    "track": 202,
    "tref": 203,
    "tspan": 204,
    "tt": 205,
    "u": 206,
    "ul": 207,
    "use": 208,
    "var": 209,
    "video": 210,
    "view": 211,
    "vkern": 212,
    "wbr": 213,
    "xmp": 214,
}
