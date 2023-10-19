//
//  ViewController.m
//  DuGuang_LiteOCR_IOS_DEMO
//
//  Created by yixian on 2023/4/27.
//

#import "ViewController.h"

#import "Model_Scope_OCR_Lib.framework/Headers/Model_Scope_OCR_Lib.h"

#import "ImgHelper.h"

#include <sys/time.h>

@interface ViewController ()


@property (retain,nonatomic)UIImageView *imageV;
@property (retain,nonatomic)UILabel *ocrtext;
@property NSString * fileName;

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    self.edgesForExtendedLayout = UIRectEdgeNone;
    self.title=@"";
    [self.view setBackgroundColor:[UIColor whiteColor]];

    CGRect rc = [[UIScreen mainScreen] bounds];
    
    self.imageV=[[UIImageView alloc] initWithFrame:CGRectMake(0, (int)(rc.size.height * 0.05), rc.size.width, (int)(rc.size.height * 0.8))];
    self.imageV.hidden = NO;
    self.imageV.backgroundColor = [UIColor grayColor];
    self.imageV.contentMode = UIViewContentModeScaleAspectFit;
    [self.view addSubview:self.imageV];
    
    UIButton *captureButton=[[UIButton alloc] initWithFrame:CGRectMake(rc.size.width/4-30, (int)(rc.size.height * 0.88), 60, 50)];
    [captureButton addTarget:self action:@selector(captureButtonPressed:) forControlEvents:UIControlEventTouchUpInside];
    [captureButton setBackgroundColor:[UIColor blueColor]];
    [captureButton setTitle:@"相机" forState:UIControlStateNormal];
    [captureButton.layer setCornerRadius:5.0];
    [self.view addSubview:captureButton];
    
    UIButton *picButton=[[UIButton alloc] initWithFrame:CGRectMake(rc.size.width/2-30, (int)(rc.size.height * 0.88), 60, 50)];
    [picButton addTarget:self action:@selector(picButtonPressed:) forControlEvents:UIControlEventTouchUpInside];
    [picButton setBackgroundColor:[UIColor blueColor]];
    [picButton setTitle:@"相册" forState:UIControlStateNormal];
    [picButton.layer setCornerRadius:5.0];
    [self.view addSubview:picButton];
    
    //rec button
    UIButton *recButton=[[UIButton alloc] initWithFrame:CGRectMake(3*rc.size.width/4-30, (int)(rc.size.height * 0.88), 60, 50)];
    [recButton addTarget:self action:@selector(capture_ocr_rec:) forControlEvents:UIControlEventTouchUpInside];
    [recButton setBackgroundColor:[UIColor blueColor]];
    [recButton setTitle:@"识别" forState:UIControlStateNormal];
    [recButton.layer setCornerRadius:5.0];
    [self.view addSubview:recButton];
    
    //show ocr result
    self.ocrtext = [[UILabel alloc] init];
    self.ocrtext.text = @"";
    self.ocrtext.font = [UIFont systemFontOfSize:10];
    self.ocrtext.frame = CGRectMake(0, (int)(rc.size.height * 0.05), rc.size.width, (int)(rc.size.height * 0.8));
    self.ocrtext.backgroundColor = [UIColor whiteColor];
    self.ocrtext.hidden=YES;
    self.ocrtext.textColor = [UIColor blueColor];
    self.ocrtext.numberOfLines = 0;
    self.ocrtext.userInteractionEnabled=YES;
    UITapGestureRecognizer *labelTapGestureRecognizer = [[UITapGestureRecognizer alloc]initWithTarget:self action:@selector(labelTouchUpInside:)];
    [self.ocrtext addGestureRecognizer:labelTapGestureRecognizer];
    [self.view addSubview:self.ocrtext];

    //ocr model init
    NSString * model_path=[NSString stringWithFormat:@"%@",[[NSBundle mainBundle] bundlePath]];
    init_ocr([model_path UTF8String], 4);
}

-(void) labelTouchUpInside:(UITapGestureRecognizer *)recognizer{
    self.ocrtext.hidden=YES;
}

-(void)capture_ocr_rec:(id)sender
{
    UIImage * img = self.imageV.image;

    int height = CGImageGetHeight(img.CGImage);
    int width = CGImageGetWidth(img.CGImage);
    //bgra
    unsigned char * data = [ImgHelper convertImageToData:img];
    int channel = 4;

    //rgb
    unsigned char * recdata = new unsigned char[width*height*3];
    for (int i=0; i<width*height; i++)
    {
        recdata[i*3]=data[i*4+0];
        recdata[i*3+1]=data[i*4+1];
        recdata[i*3+2]=data[i*4+2];
    }

    struct timeval start;
    struct timeval end;
    gettimeofday(&start,0);
    vector<string> rst = ocr_recognize(recdata, width, height, 3, 1);
    gettimeofday( &end, NULL );
    int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec -start.tv_usec;
    printf("ocr rec time:%f\n",timeuse/1000.0/1000);

    string combine_rst="";
    for (int i=0; i<(int)(rst.size()); i++)
    {
        printf("%d:%s\n", i, rst[i].c_str());

        char kkk[2048]={0};
        sprintf(kkk, "%s\n",rst[i].c_str());

        combine_rst = combine_rst + (string)kkk;
    }

    NSString *nsMessage= [[NSString alloc] initWithCString:combine_rst.c_str() encoding:NSUTF8StringEncoding];

    delete []recdata;

    self.ocrtext.text = nsMessage;
    self.ocrtext.hidden = NO;
    [self.view bringSubviewToFront:self.ocrtext];
    
    
//    //for image OCR
//    {
//
//        //read img
//        NSString * imgpath=[NSString stringWithFormat:@"%@/IMG_7197.JPG",[[NSBundle mainBundle] bundlePath]];
//        UIImage * img = [UIImage imageWithContentsOfFile:imgpath];
//
//        int height = (int)CGImageGetHeight(img.CGImage);
//        int width = (int)CGImageGetWidth(img.CGImage);
//
//        //bgra
//        unsigned char * data = [ImgHelper convertImageToData:img];
//
//        //bgr
//        unsigned char * recdata = new unsigned char[width*height*3];
//        for (int i=0; i<width*height; i++)
//        {
//            recdata[i*3]=data[i*4+0];
//            recdata[i*3+1]=data[i*4+1];
//            recdata[i*3+2]=data[i*4+2];
//        }
//
//        struct timeval start;
//        struct timeval end;
//        gettimeofday(&start,0);
//        vector<string> rst = ocr_recognize(recdata, width, height, 3, 1);
//        gettimeofday( &end, NULL );
//        int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec -start.tv_usec;
//        printf("ocr rec time:%f\n",timeuse/1000.0/1000);
//
//        string combine_rst="";
//        for (int i=0; i<(int)(rst.size()); i++)
//        {
//            char kkk[2048]={0};
//            sprintf(kkk, "%s\n",rst[i].c_str());
//
//            combine_rst = combine_rst + (string)kkk;
//        }
//
//        printf("%s\n", combine_rst.c_str());
//    }
}


-(void)picButtonPressed:(id)sender
{
    self.ocrtext.hidden=YES;
    UIImagePickerController *controller = [[UIImagePickerController alloc] init];
    controller.delegate = self;
    controller.sourceType = UIImagePickerControllerSourceTypeSavedPhotosAlbum;
    controller.modalTransitionStyle=UIModalTransitionStyleFlipHorizontal;
    controller.allowsEditing=YES;
    [self presentViewController:controller animated:YES completion:NULL];
}

-(void)captureButtonPressed:(id)sender
{
    self.ocrtext.hidden=YES;
    UIImagePickerController *picker = [[UIImagePickerController alloc] init];
    
    if([UIImagePickerController isSourceTypeAvailable:UIImagePickerControllerSourceTypeCamera]) {
        picker.sourceType = UIImagePickerControllerSourceTypeCamera;
        NSArray *temp_MediaTypes = [UIImagePickerController availableMediaTypesForSourceType:picker.sourceType];
        picker.mediaTypes = temp_MediaTypes;
        picker.delegate = (id)self;
        picker.allowsImageEditing = YES;
    }
    [self presentModalViewController:picker animated:YES];
}
 
- (void)saveImage:(UIImage *)image {
    self.imageV.image = image;
}
 
- (void)imagePickerController:(UIImagePickerController *)picker didFinishPickingMediaWithInfo:(NSDictionary *)info
{
    
    NSString *mediaType = [info objectForKey:UIImagePickerControllerMediaType];
    
    BOOL success;
    NSFileManager *fileManager = [NSFileManager defaultManager];
    NSError *error;
    
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *documentsDirectory = [paths objectAtIndex:0];
    
    if ([mediaType isEqualToString:@"public.image"]){
        
        UIImage *image = [info objectForKey:@"UIImagePickerControllerEditedImage"];
        self.imageV.image=image;
       }
       else if([mediaType isEqualToString:@"public.movie"]){
           
    }
    [picker dismissModalViewControllerAnimated:YES];
}
 
- (void)imagePickerControllerDidCancel:(UIImagePickerController *)picker {
    
    [picker dismissModalViewControllerAnimated:YES];
}


@end
