class DocSpaceLayout:
    def __init__(self, use_advanced_space_layout=False):
        if use_advanced_space_layout:
            self.use_py_space = False
        else:
            self.use_py_space = True

        self.use_advanced_space_layout = use_advanced_space_layout

    
    @staticmethod
    def box4point_to_box2point(box4point):
        # bounding box = [x0, y0, x1, y1, x2, y2, x3, y3]
        all_x = [box4point[2 * i] for i in range(4)]
        all_y = [box4point[2 * i + 1] for i in range(4)]
        box2point = [min(all_x), min(all_y), max(all_x), max(all_y)]
        return box2point
    
    @staticmethod
    def is_same_line(box1, box2):
        """
        Params:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
        """
        
        box1_midy = (box1[1] + box1[3]) / 2
        box2_midy = (box2[1] + box2[3]) / 2

        if (box1_midy < box2[3] and box1_midy > box2[1] and
            box2_midy < box1[3] and box2_midy > box1[1]):
            return True
        else:
            return False
    
    @staticmethod
    def union_box(box1, box2):
        """
        Params:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
        """
        x1 = min(box1[0], box2[0])
        y1 = min(box1[1], box2[1])
        x2 = max(box1[2], box2[2])
        y2 = max(box1[3], box2[3])

        return [x1, y1, x2, y2]

    @staticmethod
    def boxes_sort(box1, box2):
        """
        Params:
            boxes: [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
        """
        sorted_id = sorted(
            range(len(boxes)), key=lambda x: (boxes[x][1], boxes[x][0]))

        return sorted_id

    def space_layout(self, texts, boxes):
        """
        Params:
            texts: ocr 文本行string [text1, text2, ...]
            boxes: ocr 文本行坐标 [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
        """

        line_boxes = []
        line_texts = []
        max_line_char_num = 0
        line_width = 0
        # print(f"len_boxes: {len(boxes)}")
        while len(boxes) > 0:
            line_box = [boxes.pop(0)]
            line_text = [texts.pop(0)]
            char_num = len(line_text[-1])
            line_union_box = line_box[-1]
            while len(boxes) > 0 and self.is_same_line(line_box[-1], boxes[0]):
                line_box.append(boxes.pop(0))
                line_text.append(texts.pop(0))
                char_num += len(line_text[-1])
                line_union_box = self.union_box(line_union_box, line_box[-1])
            line_boxes.append(line_box)
            line_texts.append(line_text)
            if char_num >= max_line_char_num:
                max_line_char_num = char_num
                line_width = line_union_box[2] - line_union_box[0]
        
        # print(line_width)

        char_width = line_width / max_line_char_num
        # print(char_width)
        if char_width == 0:
            char_width = 1

        space_line_texts = []
        for i, line_box in enumerate(line_boxes):
            space_line_text = ""
            for j, box in enumerate(line_box):
                left_char_num = int(box[0] / char_width)
                space_line_text += " " * (left_char_num - len(space_line_text))
                space_line_text += line_texts[i][j]
            space_line_texts.append(space_line_text)


        doc_str = "\n".join(space_line_texts)
        return doc_str, space_line_texts