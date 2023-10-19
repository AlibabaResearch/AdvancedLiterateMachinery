import os
import glob

def get_all_text(filename):
    all_rst=''
    fl = open(filename, 'r')
    
    for line in fl:
        line = line.strip()
        all_rst = all_rst + line
        
    fl.close()
    
    return all_rst
        

def get_our_text(filename):
    all_rst=''
    fl = open(filename, 'r')
    
    for line in fl:
        line = line.strip()
        
        line = line.replace('kfzkfzkfz ','')
        
#        if len(line) <= 3:
#            continue
#
#        sline=''
#        for m in range(3, len(line)):
#            sline = sline + line[m]
#
        all_rst = all_rst + line
        
    fl.close()
    
    return all_rst
    


def get_gt_text(filename):
    all_rst=''
    fl = open(filename, 'r')
    
    for line in fl:
        line = line.strip()
        line = line.split(',')

        if len(line)<=8:
            continue

        sline = ''

        for kk in range(8, len(line)):
            if kk == 8:
                sline = line[kk]
            else:
                sline = sline + ',' + line[kk]
                
        all_rst = all_rst + sline
        
    fl.close()
    
    return all_rst
    


def deal_rst(rst):
    rst=rst.replace('。','.')
    rst=rst.replace('，',',')
    rst=rst.replace(' ','')
    rst=rst.replace('；',';')
    rst=rst.replace('？','?')
    rst=rst.replace('：',':')
    rst=rst.replace('@','@')
    rst=rst.replace('#','#')
    rst=rst.replace('—','-')
    rst=rst.replace('（','(')
    rst=rst.replace('）',')')
    rst=rst.replace('！','!')
    rst=rst.replace('“','"')
    rst=rst.replace('”','"')
    return rst


def get_acc(lb, pre, is_show=False):

    length = len(lb)
    
    lb_pp = []
    
    for i in range(length):
        lb_pp.append(1)
    
    
    rightn=0
    wrongn=0
    mmm=0
    for m in pre:
        
        is_find=False
        for i in range(length):
            if lb_pp[i]==1:
                if m == lb[i]:
                    is_find=True
                    lb_pp[i] =0
                    break
        if is_find==True:
            rightn=rightn+1
        else:
        
            if is_show==True:
            
                if mmm + 2 < len(pre) and mmm -2 >=0:
            
                    print('wrong', m, ' | wrong word',  pre[mmm-2]+ pre[mmm-1] +pre[mmm]+pre[mmm+1]+pre[mmm+2])
                else:
                    print('wrong', m)
        
            wrongn=wrongn+1
        mmm=mmm+1
    
    return 1.0*rightn/length, 1.0*wrongn/length
                


img_list = glob.glob('./image/*.jpeg') + glob.glob('./image/*.jpg') + glob.glob('./image/*.png')

print('img_list', len(img_list))


all_rst_label=''
our_all_rst=''
apple_all_rst=''


gt_list=[]
apple_list=[]
our_list=[]

name_list =[]

for imgpath in img_list:
    
    imgname = os.path.basename(imgpath)
    
    name_list.append(imgname)
    
    imgname = imgname.split('.')
    
    num = len(imgname)
    
    txtname = ''
    
    for m in range(num-1):
        if m ==0:
            txtname = imgname[m]
        else:
            txtname = txtname + '.' + imgname[m]
    
    txtname = txtname + '.txt'
    
    gt_txtname = './gt/' + txtname
    apple_txtname = './apple_result/' + txtname
    our_txtname = './our_result/' + txtname
    
    
    gt_list.append(get_gt_text(gt_txtname))
    apple_list.append(get_all_text(apple_txtname))
    our_list.append(get_our_text(our_txtname))

    all_rst_label = all_rst_label + get_gt_text(gt_txtname)
    apple_all_rst = apple_all_rst + get_all_text(apple_txtname)
    our_all_rst = our_all_rst + get_our_text(our_txtname)
    

print('all_rst_label', all_rst_label)
print('our_all_rst', our_all_rst)
print('apple_all_rst', apple_all_rst)


all_rst_label=deal_rst(all_rst_label)
our_all_rst=deal_rst(our_all_rst)
apple_all_rst=deal_rst(apple_all_rst)


#a,b = get_acc(all_rst_label, our_all_rst)
#c,d = get_acc(all_rst_label, apple_all_rst)
#

#print('our a',a, 'b',b)
#print('apple c',c, 'c',d)


avg_apple_right=0
avg_apple_wrong=0
avg_our_right=0
avg_our_wrong=0
for m in range(len(gt_list)):

    print('\n\n')
    print('name', name_list[m])
#    print('our', our_list[m])

    gt_list[m] = deal_rst(gt_list[m])
    our_list[m] = deal_rst(our_list[m])
    apple_list[m] = deal_rst(apple_list[m])

    a,b = get_acc(gt_list[m], our_list[m], True)
    c,d = get_acc(gt_list[m], apple_list[m], False)
    
    print('length', len(gt_list[m]))

    print('out length', len(our_list[m]))
    print('our a',a, 'b',b)

    print('apple length', len(apple_list[m]))
    print('apple c',c, 'c',d)
    
    avg_our_right = avg_our_right + a
    avg_our_wrong = avg_our_wrong + b
    
    avg_apple_right = avg_apple_right + c
    avg_apple_wrong = avg_apple_wrong + d

print('avg_our_right', avg_our_right/len(gt_list))
print('avg_our_wrong', avg_our_wrong/len(gt_list))
print('avg_apple_right', avg_apple_right/len(gt_list))
print('avg_apple_wrong', avg_apple_wrong/len(gt_list))

print('all length', len(all_rst_label))
print('our length', len(our_all_rst))
print('apple length', len(apple_all_rst))
