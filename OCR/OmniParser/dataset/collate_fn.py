import torch
import random
import cv2
import numpy as np
from utils.nested_tensor import nested_tensor_from_tensor_list
from utils.misc import sample_bezier_curve

class SeqConstructor(object):
    def __init__(self, args):
        self.num_bins = args.num_bins 
        self.num_chars = len(args.chars)
        self.pt_seq_length = args.pt_seq_length 
        self.global_prob = args.global_prob
        self.use_char_window_prompt = args.use_char_window_prompt
        self.train_vie = args.train_vie

        self.rec_eos_index = args.rec_eos_index

        self.padding_index = args.padding_index
        self.padding_tensor = torch.tensor([self.padding_index]*2,  dtype=torch.long)
        
        if self.use_char_window_prompt:
            self.padding_pt_tensor = torch.tensor([self.padding_index]*6,  dtype=torch.long)
        else:
            self.padding_pt_tensor = torch.tensor([self.padding_index]*4,  dtype=torch.long)

        self.pt_sos_tensor = torch.tensor([args.pt_sos_index], dtype=torch.long)
        self.poly_sos_tensor = torch.tensor([args.poly_sos_index], dtype=torch.long)
        self.rec_sos_tensor = torch.tensor([args.rec_sos_index], dtype=torch.long)

        self.pt_eos_tensor = torch.tensor([args.pt_eos_index],  dtype=torch.long)
        self.poly_eos_tensor = torch.tensor([args.poly_eos_index],  dtype=torch.long)
        self.rec_eos_tensor = torch.tensor([self.rec_eos_index],  dtype=torch.long)

    def process_seq(self, input_seqs_, output_seqs_):
        max_seq_length = max([len(seq) for seq in input_seqs_])
        input_seqs = torch.ones(len(input_seqs_), max_seq_length, dtype=torch.long) * self.padding_index
        output_seqs = torch.ones(len(output_seqs_), max_seq_length, dtype=torch.long) * self.padding_index

        for i in range(len(input_seqs_)):
            input_seqs[i, :len(input_seqs_[i])].copy_(input_seqs_[i])
            output_seqs[i, :len(output_seqs_[i])].copy_(output_seqs_[i])

        return input_seqs, output_seqs

    def get_spatial_window_prompt(self, pt_label, global_step=0.4, n_bins=1000):
        # prob for different mode
        # 40% all area, 30% rect_pattern, 30% random
        prob = random.uniform(0, 1)
        if prob < global_step:
            # default window
            start_x, start_y, end_x, end_y = [0, 0, n_bins - 1, n_bins - 1]
        elif prob < 0.7:
            # x-axis and y-axis are partitioned into varying numbers of blocks.
            num_xs = [3, 3, 1, 3, 2, 2, 2, 1]
            num_ys = [3, 1, 3, 2, 3, 2, 1, 2]

            total_windows = []
            for num_x, num_y in zip(num_xs, num_ys):
                inter_x = min(int(n_bins / num_x), n_bins - 1)
                inter_y = min(int(n_bins / num_y), n_bins - 1)

                for i in range(num_x):
                    for j in range(num_y):
                        start_x = i*inter_x
                        start_y = j*inter_y
                        end_x = min(start_x + inter_x, n_bins - 1)
                        end_y = min(start_y + inter_y, n_bins - 1)
                        total_windows.append([start_x, start_y, end_x, end_y])
            
            start_x, start_y, end_x, end_y = random.choice(total_windows)
        else:
            inter = int(n_bins / 3)
            start_x = random.randint(0, inter * 2)
            start_y = random.randint(0, inter * 2)
            rect_w, rect_h = random.randint(inter, n_bins - 1), random.randint(inter, n_bins - 1)
            end_x, end_y = min(start_x + rect_w, n_bins - 1), min(start_y + rect_h, n_bins - 1)

        spatial_window_prompt = [start_x, start_y, end_x, end_y]
        valid_x = (pt_label[:, 0] > spatial_window_prompt[0]) & (pt_label[:, 0] <= spatial_window_prompt[2])
        valid_y = (pt_label[:, 1] > spatial_window_prompt[1]) & (pt_label[:, 1] <= spatial_window_prompt[3])
        valid_index = valid_x & valid_y
        return spatial_window_prompt, valid_index

    def get_char_window_prompt(self, num_chars, rec_label, global_step=0.4, n_bins=1000):
        min_char = 0
        max_char = num_chars
        min_num = 3
        chars, indices = rec_label[:,0].sort()
        len_chars = len(chars)

        if len_chars > 0:
            if random.uniform(0, 1) < global_step:
                start_char = min_char
                end_char = max_char
            else:
                min_num = min(min_num, len_chars)
                min_num = random.randint(min_num, len_chars)

                start_index = random.randint(0, len_chars - min_num)
                end_index = start_index + min_num -1

                start_char = chars[start_index].item()
                end_char = chars[end_index].item()
        else:
            start_char = random.randint(min_char, max_char)
            end_char = random.randint(start_char, max_char)

        # select rec
        valid_index = (rec_label[:,0] >= start_char) & (rec_label[:,0] <= end_char)

        start_char += n_bins
        end_char += n_bins
        char_window_prompt = torch.tensor([start_char, end_char],  dtype=torch.long)
        return char_window_prompt, valid_index

    def random_sample_seq(self, batch_center_pts, input_poly_seqs_, output_poly_seqs_, input_rec_seqs_, output_rec_seqs_):
        input_poly_seqs = []
        output_poly_seqs = []
        input_rec_seqs = []
        output_rec_seqs = []

        batch_size = len(batch_center_pts)
        for j in range(batch_size):
            length = input_poly_seqs_[j].shape[0]
            random_index = random.randint(0,length - 1)
            random_pt = batch_center_pts[j][random_index]
            tmp_input_poly_seq = torch.cat([random_pt, input_poly_seqs_[j][random_index]])
            input_poly_seqs.append(tmp_input_poly_seq)
            output_poly_seqs.append(output_poly_seqs_[j][random_index])
        for j in range(batch_size):
            length = input_rec_seqs_[j].shape[0]
            random_index = random.randint(0,length - 1)
            random_pt = batch_center_pts[j][random_index]
            tmp_input_rec_seq = torch.cat([random_pt, input_rec_seqs_[j][random_index]])
            input_rec_seqs.append(tmp_input_rec_seq)
            output_rec_seqs.append(output_rec_seqs_[j][random_index])
        return input_poly_seqs, output_poly_seqs, input_rec_seqs, output_rec_seqs

    def __call__(self, targets):
        input_pt_seqs_ = []
        output_pt_seqs_ = []

        input_poly_seqs_ = []
        output_poly_seqs_ = []

        input_rec_seqs_ = []
        output_rec_seqs_ = []

        batch_center_pts = []

        for target in targets:
            center_pts = target['center_pts']
            batch_center_pts.append(center_pts)

            # pt seq
            pt_label = center_pts
            rec_label = target['recog']

            spatial_window_prompt, valid_index = self.get_spatial_window_prompt(pt_label, self.global_prob, self.num_bins)

            valid_pt_label = pt_label[valid_index]
            valid_rec_label = rec_label[valid_index]

            # dont care index
            wo_dc_index = valid_rec_label[:,0] != (self.num_chars + 1)
            valid_pt_label = valid_pt_label[wo_dc_index]
            valid_rec_label = valid_rec_label[wo_dc_index]

            prompt_tensor = torch.tensor(spatial_window_prompt,  dtype=torch.long)

            if self.use_char_window_prompt:
                char_window_prompt, valid_index = self.get_char_window_prompt(self.num_chars, valid_rec_label, self.global_prob, self.num_bins)
                valid_pt_label = valid_pt_label[valid_index]
                prompt_tensor = torch.cat([prompt_tensor, char_window_prompt])

            input_pt_seq = torch.cat([prompt_tensor, self.pt_sos_tensor, valid_pt_label.flatten()])[:self.pt_seq_length]
            output_pt_seq = torch.cat([self.padding_pt_tensor, valid_pt_label.flatten(), self.pt_eos_tensor])[:self.pt_seq_length]          

            if self.train_vie:
                pt_label = target['sorted_instance_pts']
                if not self.use_char_window_prompt:
                    prompt_tensor = torch.tensor([0, 0, self.num_bins - 1, self.num_bins - 1],  dtype=torch.long)
                else:
                    prompt_tensor = torch.tensor([0, 0, self.num_bins - 1, self.num_bins - 1, self.num_bins, self.num_bins + self.num_chars],  dtype=torch.long)

                input_pt_seq = torch.cat([prompt_tensor, self.pt_sos_tensor, pt_label.flatten()])[:self.pt_seq_length]
                output_pt_seq = torch.cat([self.padding_pt_tensor, pt_label.flatten(), self.pt_eos_tensor])[:self.pt_seq_length]  

            input_pt_seqs_.append(input_pt_seq)
            output_pt_seqs_.append(output_pt_seq)

            # poly seq
            poly_label = target['polygons']
            num_texts = poly_label.shape[0]
            poly_label = (poly_label * self.num_bins).floor().type(torch.long)
            poly_label = torch.clamp(poly_label, min=0, max=self.num_bins - 1)

            input_poly_seq = torch.cat([self.poly_sos_tensor.unsqueeze(0).repeat(num_texts, 1), poly_label], dim=-1)
            output_poly_seq = torch.cat([self.padding_tensor.unsqueeze(0).repeat(num_texts, 1), poly_label, self.poly_eos_tensor.unsqueeze(0).repeat(num_texts, 1)], dim=-1)

            input_poly_seqs_.append(input_poly_seq)
            output_poly_seqs_.append(output_poly_seq)

            # rec seq
            rec_label = target['recog'] + self.num_bins
            rec_label[rec_label == (self.num_bins + self.num_chars + 1)] = self.padding_index

            for i in range(len(rec_label)):
                for j in range(len(rec_label[i])):
                    if j == 0 and rec_label[i,j].item() == self.padding_index:
                        break
                    if rec_label[i,j].item() == self.padding_index:
                        rec_label[i,j] = self.rec_eos_index
                        break

            input_rec_seq = torch.cat([self.rec_sos_tensor.unsqueeze(0).repeat(num_texts, 1), rec_label], dim=-1)
            output_rec_seq = torch.cat([self.padding_tensor.unsqueeze(0).repeat(num_texts, 1), rec_label], dim=-1)

            input_rec_seqs_.append(input_rec_seq)
            output_rec_seqs_.append(output_rec_seq)

        input_poly_seqs, output_poly_seqs, input_rec_seqs, output_rec_seqs = self.random_sample_seq(batch_center_pts, input_poly_seqs_, output_poly_seqs_, input_rec_seqs_, output_rec_seqs_)
        input_pt_seqs, output_pt_seqs = self.process_seq(input_pt_seqs_, output_pt_seqs_)
        input_poly_seqs, output_poly_seqs = self.process_seq(input_poly_seqs, output_poly_seqs)
        input_rec_seqs, output_rec_seqs = self.process_seq(input_rec_seqs, output_rec_seqs)

        input_seqs = [input_pt_seqs, input_poly_seqs, input_rec_seqs]
        output_seqs = [output_pt_seqs, output_poly_seqs, output_rec_seqs]
        
        return input_seqs, output_seqs


class CollateFN(object):
    def __init__(self, image_set, args):
        self.seq_constructor = SeqConstructor(args)
        self.train = (image_set == 'train')
        self.args = args

    def __call__(self, batch):
        batch = list(zip(*batch))

        tensors = batch[0]
        targets = batch[1]
    
        images = nested_tensor_from_tensor_list(tensors)

        if self.train:
            input_seqs, output_seqs = self.seq_constructor(targets)
            return images, input_seqs, output_seqs 
        else:
            return images, targets