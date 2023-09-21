//
//  ImgHelper.mm
//  TestXNN_PHOTO
//
//  Created by yixian on 2018/6/1.
//  Copyright © 2018年 yixian. All rights reserved.  


#import "ImgHelper.h"

@implementation ImgHelper


+ (unsigned char *)convertImageToData:(UIImage *)image
{
    size_t img_width = CGImageGetWidth(image.CGImage);
    size_t img_height = CGImageGetHeight(image.CGImage);
    if(img_width == 0 || img_height == 0)
        return NULL;
    
    unsigned char* imageData = new unsigned char[img_width * img_height * 4];
    if (imageData==NULL) {
        return NULL;
    }
    int bytesPerRow=(int)img_width * 4;
    CGImageAlphaInfo alphaInfo=kCGImageAlphaPremultipliedLast;
    
    
    CGColorSpaceRef cref = CGColorSpaceCreateDeviceGray();
    CGContextRef gcref = CGBitmapContextCreate(imageData, img_width, img_height, 8, bytesPerRow, cref, alphaInfo);
    CGColorSpaceRelease(cref);
    UIGraphicsPushContext(gcref);
    
   
    CGContextSetRGBFillColor(gcref, 1.0, 1.0, 1.0, 1.0);
    CGContextFillRect(gcref, CGRectMake(0.0, 0.0, (CGFloat)img_width, (CGFloat)img_height));
    
    
    CGRect rect = {{0, 0}, {(CGFloat)img_width, (CGFloat)img_height}};
    CGContextDrawImage(gcref, rect, image.CGImage);
    UIGraphicsPopContext();
    CGContextRelease(gcref);
    
    return imageData;
}


@end
