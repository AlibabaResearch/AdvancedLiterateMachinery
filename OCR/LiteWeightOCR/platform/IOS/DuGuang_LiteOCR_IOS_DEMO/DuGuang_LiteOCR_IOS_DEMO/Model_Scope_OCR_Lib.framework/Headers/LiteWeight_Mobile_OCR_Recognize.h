//
//  Header.h
//  Model_Scope_OCR_Lib
//
//  Created by yixian on 2023/5/30.
//

#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>

#include <iostream>
#include <string>
#include <map>
#include <vector>
using namespace std;

bool init_ocr(const char * path, int thread_n);

vector<string> ocr_recognize(unsigned char * data, int width, int height, int channel, int use_dct_refine);

