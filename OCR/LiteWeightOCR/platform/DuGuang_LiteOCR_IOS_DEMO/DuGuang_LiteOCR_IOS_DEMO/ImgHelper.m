//
//  ImgHelper.m
//  TestXNN_PHOTO
//
//  Created by yixian on 2018/6/1.
//  Copyright © 2018年 yixian. All rights reserved.
//

#import "ImgHelper.h"

@implementation ImgHelper


+ (CGContextRef)newBitmapContextFromImage:(CGImageRef)image WithBitmapInfo:(uint32_t)bitmapInfo {
    CGContextRef context = NULL;
    CGColorSpaceRef colorSpace;
    unsigned char *bitmapData;
    
    size_t bitsPerPixel = 32;
    size_t bitsPerComponent = 8;
    size_t bytesPerPixel = bitsPerPixel / bitsPerComponent;
    
    size_t width = CGImageGetWidth(image);
    size_t height = CGImageGetHeight(image);
    
    size_t bytesPerRow = width * bytesPerPixel;
    size_t bufferLength = bitsPerPixel * height * width;
    
    colorSpace = CGColorSpaceCreateDeviceRGB();
    
    if (!colorSpace) {
        NSLog(@"Error allocating color space RGB\n");
        return NULL;
    }
    
    // Allocate memory for image data
    bitmapData = (unsigned char*) calloc(bufferLength, sizeof(unsigned char));
    
    if (!bitmapData) {
        NSLog(@"Error allocating memory for bitmap\n");
        CGColorSpaceRelease(colorSpace);
        return NULL;
    }
    
    //Create bitmap context
    context = CGBitmapContextCreate(bitmapData,
                                    width,
                                    height,
                                    bitsPerComponent,
                                    bytesPerRow,
                                    colorSpace,
                                    bitmapInfo);
    if (!context) {
        free(bitmapData);
        NSLog(@"Bitmap context not created");
    }
    
    CGColorSpaceRelease(colorSpace);
    
    return context;
}

+ (unsigned char *)convertImageToData:(UIImage *)image {
    
    // 1. Convert to ARGB数据,小端存放，则变成BGRA数据
    uint32_t info = kCGImageByteOrder32Little|kCGImageAlphaNoneSkipFirst;
    CGContextRef bitmapContext = [self newBitmapContextFromImage:image.CGImage WithBitmapInfo:info];
    // Draw image into the context to get the raw image data
    if (!bitmapContext) {
        return NULL;
    }
    
    size_t width = CGImageGetWidth(image.CGImage);
    size_t height = CGImageGetHeight(image.CGImage);
    
    CGRect rect = CGRectMake(0, 0, width, height);
    // Draw image into the context to get the raw image data
    CGContextDrawImage(bitmapContext, rect, image.CGImage);
    
    unsigned char *pBitmap =(unsigned char *)CGBitmapContextGetData(bitmapContext);
    size_t length = CGBitmapContextGetBytesPerRow(bitmapContext) * CGBitmapContextGetHeight(bitmapContext);
    
    if (bitmapContext) {
        CGContextRelease(bitmapContext);
    }
    
    return pBitmap;
}


+(NSInteger)getMaxConfRectIndex:(NSArray*)array
{
    if (array==nil || [array count] == 0) {
        return -1;
    }
    float maxConf = [[array[0] objectForKey:@"conf"] floatValue];
    int maxIndex = 0;
    for (int m=1; m<[array count]; m++)
    {
        float conf = [[array[m] objectForKey:@"conf"] floatValue];
        if (conf > maxConf) {
            maxConf = conf;
            maxIndex = m;
        }
    }
    //阈值判定
    if (maxConf < 0.10) {
        return -1;
    }
    //NSLog(@"maxConf%f",maxConf);
    return maxIndex;
}



+ (UIImage *) convertBitmapRGBA8ToUIImage:(unsigned char *) buffer
                                withWidth:(int) width
                               withHeight:(int) height
{
    size_t bufferLength = width * height * 4;
    CGDataProviderRef provider = CGDataProviderCreateWithData(NULL, buffer, bufferLength, NULL);
    size_t bitsPerComponent = 8;
    size_t bitsPerPixel = 32;
    size_t bytesPerRow = 4 * width;
    
    CGColorSpaceRef colorSpaceRef = CGColorSpaceCreateDeviceRGB();
    if(colorSpaceRef == NULL) {
        NSLog(@"Error allocating color space");
        CGDataProviderRelease(provider);
        return nil;
    }
    
    CGBitmapInfo bitmapInfo = kCGBitmapByteOrderDefault |kCGImageAlphaPremultipliedLast;
    CGColorRenderingIntent renderingIntent = kCGRenderingIntentDefault;
    
    CGImageRef iref = CGImageCreate(width,
                                    height,
                                    bitsPerComponent,
                                    bitsPerPixel,
                                    bytesPerRow,
                                    colorSpaceRef,
                                    bitmapInfo,
                                    provider,    // data provider
                                    NULL,        // decode
                                    YES,            // should interpolate
                                    renderingIntent);
    
    uint32_t* pixels = (uint32_t*)malloc(bufferLength);
    
    if(pixels == NULL) {
        NSLog(@"Error: Memory not allocated for bitmap");
        CGDataProviderRelease(provider);
        CGColorSpaceRelease(colorSpaceRef);
        CGImageRelease(iref);
        return nil;
    }
    
    CGContextRef context = CGBitmapContextCreate(pixels,
                                                 width,
                                                 height,
                                                 bitsPerComponent,
                                                 bytesPerRow,
                                                 colorSpaceRef,
                                                 bitmapInfo);
    
    if(context == NULL) {
        NSLog(@"Error context not created");
        free(pixels);
    }
    
    UIImage *image = nil;
    if(context)
    {
        CGContextDrawImage(context, CGRectMake(0.0f, 0.0f, width, height), iref);
        
        CGImageRef imageRef = CGBitmapContextCreateImage(context);
        
        // Support both iPad 3.2 and iPhone 4 Retina displays with the correct scale
        if([UIImage respondsToSelector:@selector(imageWithCGImage:scale:orientation:)]) {
            float scale = [[UIScreen mainScreen] scale];
            image = [UIImage imageWithCGImage:imageRef scale:scale orientation:UIImageOrientationUp];
        } else {
            image = [UIImage imageWithCGImage:imageRef];
        }
        
        CGImageRelease(imageRef);
        CGContextRelease(context);
    }
    
    CGColorSpaceRelease(colorSpaceRef);
    CGImageRelease(iref);
    CGDataProviderRelease(provider);
    
    if(pixels) {
        free(pixels);
    }
    return image;
}


@end
