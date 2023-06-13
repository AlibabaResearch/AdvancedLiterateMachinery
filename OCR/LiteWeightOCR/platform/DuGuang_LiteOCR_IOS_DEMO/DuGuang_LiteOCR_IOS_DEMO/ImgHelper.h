//
//  ImgHelper.h
//  TestXNN_PHOTO
//
//  Created by yixian on 2018/6/1.
//  Copyright © 2018年 yixian. All rights reserved.
//

#import <Foundation/Foundation.h>

#import <UIKit/UIKit.h>

@interface ImgHelper : NSObject

+ (unsigned char *)convertImageToData:(UIImage *)image;
+(NSInteger)getMaxConfRectIndex:(NSArray*)array;
+ (UIImage *) convertBitmapRGBA8ToUIImage:(unsigned char *) buffer
                                withWidth:(int) width
                               withHeight:(int) height;
@end
