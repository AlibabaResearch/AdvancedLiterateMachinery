//
//  ImgHelper.mm
//  TestXNN_PHOTO
//
//  Created by yixian on 2018/6/1.
//  Copyright © 2018年 yixian. All rights reserved.  


#import "ImgHelper.h"

@implementation ImgHelper

+ (CGContextRef)getBCFromImg:(CGImageRef)image withbcf:(uint32_t)bcf
{
    size_t img_width = CGImageGetWidth(image);
    size_t img_height = CGImageGetHeight(image);
    
    CGColorSpaceRef clrSpa = CGColorSpaceCreateDeviceRGB();
    
    if (!clrSpa)
    {
        return NULL;
    }
    
    unsigned char * imgdata = (unsigned char*) calloc(32 * img_height * img_width, sizeof(unsigned char));
    
    if (!imgdata) {
        CGColorSpaceRelease(clrSpa);
        return NULL;
    }
    
    CGContextRef imgctx = CGBitmapContextCreate(imgdata,
                                                img_width,
                                                img_height,
                                                8,
                                                img_width * 4,
                                                clrSpa,
                                                bcf);
    if (!imgctx)
    {
        free(imgdata);
    }
    
    CGColorSpaceRelease(clrSpa);
    
    return imgctx;
}

+ (unsigned char *)convertImageToData:(UIImage *)img {
    CGContextRef bc = [self getBCFromImg:img.CGImage withbcf:kCGImageByteOrder32Little|kCGImageAlphaNoneSkipFirst];
    
    if (!bc) {
        return NULL;
    }
    
    size_t img_width = CGImageGetWidth(img.CGImage);
    size_t img_height = CGImageGetHeight(img.CGImage);
    CGRect img_rect = CGRectMake(0, 0, img_width, img_height);
    CGContextDrawImage(bc, img_rect, img.CGImage);
    
    unsigned char *pdata =(unsigned char *)CGBitmapContextGetData(bc);
    
    if (bc)
    {
        CGContextRelease(bc);
    }
    
    return pdata;
}


@end
