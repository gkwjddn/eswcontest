import cv2
import numpy as np
import math

#codec
fourcc = cv2.VideoWriter_fourcc(*'XVID')


class line:
    def __init__(self):
        # 찾은 차선의 x값
        self.x = []
        # 찾은 차선의 y값
        self.y = []
        # 찾은 차선의 곡률반
        self.curvature = 3000

#곡률반경 추정
def curve(line):
    x = line.x
    y = line.y
    
    # 찾은 점들을 이어 선으로 만들어 이차방정식 구하기
    expression = np.polyfit(y , x , 2)
    
    if(x[0]< x[-1]):
        curvature = ((1 + (2 * expression[0] * np.max(y) + expression[1]) ** 2) ** 1.5) / np.absolute(
    2 * expression[0])
    else:
        curvature = -((1 + (2 * expression[0] * np.max(y)  + expression[1]) ** 2) ** 1.5) / np.absolute(
    2 * expression[0])
       
    line.curvature = curvature    
     



# 현재 운전중인 화면에 차선과 핸들 그리기
def drawing_lane_handle(left_line, right_line, img, curvature, offset_, curvature_, pre_curvature):
    
    white=np.zeros((480,640,3), dtype=np.uint8)
    
    left_x, right_x = left_line.x, right_line.x
    
    left_y, right_y = left_line.y, right_line.y
    
    roadColor = (255,255,255)
    laneColor = (0,255,0)
    red = (0,0,255) 
    
    width = img.shape[1]

    center_of_car = width / 2
    
    center_lane_point = (left_x[0] + right_x[0])/2

    # detect한 차선의 점들로 다각형을 만든다.
    # x, y 좌표를 cv2.fillConvexPoly()에 사용할 수 있도록 정렬한다.
    left_line_l = np.transpose(np.vstack([left_x-10, left_y]))
    left_line_r = np.array(np.transpose(np.vstack([left_x+10, left_y])), np.int32)
    left_line = np.array(np.vstack([left_line_l, left_line_r[::-1]]), np.int32)
    
    right_line_l = np.array(np.transpose(np.vstack([right_x-10, right_y])), np.int32)
    right_line_r = np.array(np.transpose(np.vstack([right_x+10, right_y])), np.int32)
    right_line = np.array(np.vstack([right_line_l, right_line_r[::-1]]), np.int32)

    # 화면 중심을 기준으로 detect한 첫번째 차선의 중심과의 거리를 offset으로 한다
    # offset이 일정 거리 이상으로 커진다면 차가 치우쳐진 차선의 색을 빨간색으로 바꾼다.
    offset = center_lane_point - center_of_car
    
    if offset>50:
        white = cv2.fillConvexPoly(white, left_line, red)
        white = cv2.fillConvexPoly(white, right_line, laneColor)    
    
    elif offset<-50:        
        white = cv2.fillConvexPoly(white, left_line, laneColor)
        white = cv2.fillConvexPoly(white, right_line, red)
    
    else:
        white = cv2.fillConvexPoly(white, left_line, laneColor)
        white = cv2.fillConvexPoly(white, right_line, laneColor)
    
    # 핸들을 현재 운전중인 화면에 그린다.
    handle_img = cv2.imread('handle.jpeg')
    handle_img = cv2.resize(handle_img, dsize=(100,100))
    handle_rows, handle_cols = handle_img.shape[:2]
    
    handle_RGB = cv2.cvtColor(handle_img, cv2.COLOR_BGR2RGB)
    handle_center = int(handle_rows/2), int(handle_cols/2)

    # 곡률반경이 일정 이상으로 커진다면 잘못된 값으로 판단하게 한다.
    if (curvature >2500):
        curvature = 2500
    if (curvature < -2500):
        curvature = -2500

    if (curvature < 70 and curvature > -70):
        curvature = pre_curvature

    if(offset > 50 or offset < -50):
        degree =  offset//offset_ + int((1/curvature)*curvature_)
    else:
        degree=int((1/curvature)*curvature_)

    # detect한 차선의 곡률반경에 따라 그에 반대로 핸들을 돌리게 한다.
    # 운전자가 취해야되는 핸들의 동작을 화면에 보여준다.
    rotation = cv2.getRotationMatrix2D(handle_center, -degree,1)
    handle_img = cv2.warpAffine(handle_RGB, rotation, (handle_rows, handle_cols), flags=cv2.INTER_LINEAR,borderValue=(255,255,255))
    
    handle_roi = img[20:20+handle_rows, 270:270+handle_cols]
    
    gray_handle = cv2.cvtColor(handle_img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray_handle, 170, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)
    
    img_bg = cv2.bitwise_and(handle_roi, handle_roi, mask=mask_inv)
    img_fg = cv2.bitwise_and(handle_img, handle_img, mask=mask)
    
    dst = cv2.add(img_bg, img_fg)
    img[20:20+handle_rows, 270:270+handle_cols] = dst
    result = cv2.addWeighted(img, 1 ,white, 0.16, 0)
    
 
    
    
    return result, pre_curvature


# 카메라로 인식한 차선을 토대로 가상의 bird_view를 그린다.
def bird_view_map(left_line, right_line):
    minimap = cv2.imread('black.jpg')
    
    left_x, right_x = left_line.x, right_line.x
    both_y = left_line.y
    
    laneColor = (0,255,0)

    center_line = (left_x[0] + right_x[0])/2
    street_width = right_x[0]-left_x[0] 
    
    offset = 50
    
    # birdview의 x, y 좌표를 cv2.fillConvexPoly()에 사용할 수 있도록 정렬한다.
    left_line_l = np.array(np.transpose(np.vstack([left_x-10+offset, both_y])), np.int32)
    left_line_r = np.array(np.transpose(np.vstack([left_x+10+offset, both_y])), np.int32)
    left_line = np.array(np.vstack([left_line_l, left_line_r[::-1]]), np.int32)
    
    right_line_l = np.array(np.transpose(np.vstack([right_x-10+offset, both_y])), np.int32)
    right_line_r = np.array(np.transpose(np.vstack([right_x+10+offset, both_y])), np.int32)
    right_line = np.array(np.vstack([right_line_l, right_line_r[::-1]]), np.int32)
    
    center_lane = np.array([np.vstack([left_line_r, right_line_l[::-1]])], np.int32)

    # 검은색 화면에 가상의 도로를 그려 운전자가 직관적으로 차의 위치를 파악할 수 있도록 한다.
    minimap = cv2.fillConvexPoly(minimap, left_line, laneColor)
    minimap = cv2.fillConvexPoly(minimap, right_line, laneColor)
    minimap = cv2.fillConvexPoly(minimap, center_lane, (255,255,255))
    
    
    
    car_img = cv2.imread('car.jpeg')
    center_line = int(center_line)

    car_x = 60
    car_y = 60
    car_img = cv2.resize(car_img, dsize=(car_x, car_y))
    
    # 차선을 그린 이미지에 자동차의 이미지를 넣는다
    car_roi = minimap[300-car_y: 300 , 120  : 120 +car_x ]
   
    car_gray = cv2.cvtColor(car_img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(car_gray, 180, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)
        
    img1_bg = cv2.bitwise_and(car_roi, car_roi, mask=mask_inv)
    img2_fg = cv2.bitwise_and(car_img, car_img, mask=mask)
    
    dst = cv2.add(img1_bg, img2_fg)
    
    minimap[300-car_y: 300 , 120  : 120 +car_x ] = dst

    return minimap
    


# birdview와 실제 카메라 화면을 합쳐서 운전자가 모든 정보를 한 눈에 볼수 있도록 한다.
def img_result(minimap, realmap):
    minimap = cv2.resize(minimap, dsize=(120,120))
    realmap[30: 30+120, 490: 490+120] = minimap
    
    return realmap


# 앞 차량을 인식해 앞 차량과의 거리가 가까워지면 운전자에게 감속 알림을 준다.
def vehicleDetection():

    # 주행 차선의 전방 차량만을 인식하기 위한 ROI 설정
    mask_ROI = np.zeros((src.shape[0], src.shape[1]), dtype=np.uint8)
    mask_ROI.fill(0)


    ROI_corners = np.array([[(310, 225), (113, 412), (541, 412), (337, 225)]], dtype=np.int32)
    cv2.fillPoly(mask_ROI, ROI_corners, (255,255,255))

    # 영상의 색 영역을 HSV로 변환해 모든 차량이 가지고 있는 후미등의 빨간색을 검출하도록 한다.
    hsv_roi = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([144, 67, 0])
    upper_red = np.array([199, 255, 255])
    mask_red = cv2.inRange(hsv_roi, lower_red, upper_red)


    ROI_final = cv2.bitwise_and(mask_red, mask_ROI)
        
    kernel = np.ones((4,4), np.uint8)
    # Open CV의 dilate 함수를 이용해 멀리 있는 차량의 후미등이라도 검출할 수 있도록 한다.
    dilation = cv2.dilate(ROI_final, kernel, iterations=3)
    dilation_ROI = cv2.bitwise_and(ROI_final,mask_ROI)
    # 검출한 빨간색에 컨투어를 그리게 한다.
    vehiclecontours, hierachy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    vehiclecontours = sorted(vehiclecontours, key=lambda x: cv2.contourArea(x), reverse=True)

    max_area = 0
    ci = 0
    mask_warning = np.zeros((src.shape[0],src.shape[1]), dtype=np.uint8)
    mask_warning.fill(255)
    warning_corners = np.array([[(10, 10), (10, 470), (630, 470), (630, 10)]], dtype=np.int32)
    # 그린 컨투어 중 가장 큰(=가장 근접한) 컨투어만 사각형으로 표시한다.
    for i in range(len(vehiclecontours)):
        cnt = vehiclecontours[i]
        area = cv2.contourArea(cnt)

        if (area> max_area):
            max_area = area
            ci = i
        cnt = vehiclecontours[ci]
        x,y,w,h = cv2.boundingRect(cnt) # x, y is left upper pixel value
        rect_area = w*h
        aspect_ratio = float(w)/h

        a = (round((2*x+w)/2),round((2*y+h)/2))
        ay = round((2*y+h)/2)

    # 시속 100km기준 전방 55m 이내에 차량이 있다면 주의 알림(화면 바깥쪽 10 pixel 만큼 빨간 불이 들어오게 한다.)
        if (ay>=225 and ay<233):#55m
            src[0:480, 0:10] = (0, 0, 255)
            src[0:480, 630:640] = (0, 0, 255)
            src[0:10, 10:630] = (0, 0, 255)
            src[470:480, 10:630] = (0, 0, 255)
    # 시속 100km기준 전방 35m 이내에 차량이 있다면 경계 알림(화면 바깥쪽 20 pixel 만큼 빨간 불이 들어오게 한다.)
        elif (ay>=233 and ay<276):#35m
            src[0:480, 0:20] = (0, 0, 255)
            src[0:480, 620:640] = (0, 0, 255)
            src[0:20, 20:620] = (0, 0, 255)
            src[460:480, 20:620] = (0, 0, 255)
    # 시속 100km기준 전방 15m 이내에 차량이 있다면 위험 알림(화면 바깥쪽 30 pixel 만큼 빨간 불이 들어오게 한다.)
        elif (ay>=276 and ay<=412):#15m
            src[0:480, 0:30] = (0, 0, 255)
            src[0:480, 610:640] = (0, 0, 255)
            src[0:30, 30:610] = (0, 0, 255)
            src[450:480, 30:610] = (0, 0, 255)

 

def getbirdviewmask(src,upx,upy,dwx,dwy):

    height=src.shape[0]
    # 원본 영상(src)에서 버드뷰 시점으로 변환하는 매트릭스를 생성한다
    # pts1 : src에서 버드뷰로 변환할 부분의 영역
    # pts2 : 변환 될 버드뷰의 크기
    pts1=np.float32([[320-upx,height-upy],[320+upx,height-upy],[320-dwx,height-dwy],[320+dwx,height-dwy]])
    pts2=np.float32([[0,0],[190,0],[0,300],[190,300]])

    # 버드뷰 변환 매트릭스
    matrix=cv2.getPerspectiveTransform(pts1,pts2)
    # 역변환 매트릭스
    matrix_inv=cv2.getPerspectiveTransform(pts2,pts1)
    # 버드뷰 변환 매트릭스를 이용해 버드뷰(output)를 생성한다
    output=cv2.warpPerspective(src,matrix,(190,300))
    output1=output.copy()

    # Filter Y&W range from birdview
    # 버드뷰 hsv 변환
    hsv = cv2.cvtColor(output, cv2.COLOR_RGB2HSV)

    ##### 버드뷰 이미지의 평균 밝기 값(V값)에 따라 다른 색 영역을 적용하여 도로의 밝기 정도(그림자)에 영향을 받지 않도록 한다.
    # 버드뷰 좌측 4분할 각각의 평균 V값
    vmean_l1=int(np.mean(np.mean(hsv[:hsv.shape[0]//4,:hsv.shape[1]//2,2])))
    vmean_l2=int(np.mean(np.mean(hsv[hsv.shape[0]//4:(hsv.shape[0]//4)*2,:hsv.shape[1]//2,2])))
    vmean_l3=int(np.mean(np.mean(hsv[(hsv.shape[0]//4)*2:(hsv.shape[0]//4)*3,:hsv.shape[1]//2,2])))
    vmean_l4=int(np.mean(np.mean(hsv[(hsv.shape[0]//4)*3:(hsv.shape[0]//4)*4,:hsv.shape[1]//2,2])))

    # 버드뷰 우측 4분할 각각의 평균 V값
    vmean_r1=int(np.mean(np.mean(hsv[:hsv.shape[0]//4,hsv.shape[1]//2:,2])))
    vmean_r2=int(np.mean(np.mean(hsv[hsv.shape[0]//4:(hsv.shape[0]//4)*2,hsv.shape[1]//2:,2])))
    vmean_r3=int(np.mean(np.mean(hsv[(hsv.shape[0]//4)*2:(hsv.shape[0]//4)*3,hsv.shape[1]//2:,2])))
    vmean_r4=int(np.mean(np.mean(hsv[(hsv.shape[0]//4)*3:(hsv.shape[0]//4)*4,hsv.shape[1]//2:,2])))
    
    Y_lower_hsv = np.array([YlowH, YlowS, YlowV])
    Y_higher_hsv = np.array([YhighH, YhighS, YhighV])

    W_higher_hsv_l = np.array([WhighH, WhighS, WhighV])


    # 각 구간에서의 vmean값에 따른 색 영역 설정 //명암(그림자) 적응
    if vmean_l1<=100:
        W_lower_hsv_l1 = np.array([WlowH, 80, vmean_l1+5])
        W_higher_hsv_l = np.array([WhighH, 110, WhighV])
    if 100<vmean_l1<=170:
        W_lower_hsv_l1 = np.array([WlowH, WlowS, vmean_l1+20])
    if vmean_l1>170:
        W_lower_hsv_l1 = np.array([WlowH, WlowS, vmean_l1+30])


    if vmean_l2<=100:
        W_lower_hsv_l2 = np.array([WlowH, 80, vmean_l2+5])
        W_higher_hsv_l = np.array([WhighH, 110, WhighV])
    if 100<vmean_l2<=170:
        W_lower_hsv_l2 = np.array([WlowH, WlowS, vmean_l2+20])
    if vmean_l2>170:
        W_lower_hsv_l2 = np.array([WlowH, WlowS, vmean_l2+30])


    if vmean_l3<=100:
        W_lower_hsv_l3 = np.array([WlowH, 80, vmean_l3+5])
        W_higher_hsv_l = np.array([WhighH, 110, WhighV])
    if 100<vmean_l3<=170:
        W_lower_hsv_l3 = np.array([WlowH, WlowS, vmean_l3+20])
    if vmean_l3>170:
        W_lower_hsv_l3 = np.array([WlowH, WlowS, vmean_l3+30])


    if vmean_l4<=100:
        W_lower_hsv_l4 = np.array([WlowH, 80, vmean_l4+15])
    if 100<vmean_l4<=170:
        W_lower_hsv_l4 = np.array([WlowH, WlowS, vmean_l4+20])
    if vmean_l4>170:
        W_lower_hsv_l4 = np.array([WlowH, WlowS, vmean_l4+30])




    W_higher_hsv_r = np.array([WhighH, WhighS, WhighV])



    if vmean_r1<=100:
        W_lower_hsv_r1 = np.array([WlowH, 80, vmean_r1+5])
        W_higher_hsv_r = np.array([WhighH, 110, WhighV])
    if 100<vmean_r1<=170:
        W_lower_hsv_r1 = np.array([WlowH, WlowS, vmean_r1+20])
    if vmean_r1>170:
        W_lower_hsv_r1 = np.array([WlowH, WlowS, vmean_r1+30])


    if vmean_r2<=100:
        W_lower_hsv_r2 = np.array([WlowH, 80, vmean_r2+5])
        W_higher_hsv_r = np.array([WhighH, 110, WhighV])
    if 100<vmean_r2<=170:
        W_lower_hsv_r2 = np.array([WlowH, WlowS, vmean_r2+20])
    if vmean_r2>170:
        W_lower_hsv_r2 = np.array([WlowH, WlowS, vmean_r2+30])


    if vmean_r3<=100:
        W_lower_hsv_r3 = np.array([WlowH, 80, vmean_r3+5])
        W_higher_hsv_r = np.array([WhighH, 110, WhighV])
    if 100<vmean_r3<=170:
        W_lower_hsv_r3 = np.array([WlowH, WlowS, vmean_r3+20])
    if vmean_r3>170:
        W_lower_hsv_r3 = np.array([WlowH, WlowS, vmean_r3+30])


    if vmean_r4<=100:
        W_lower_hsv_r4 = np.array([WlowH, 80, vmean_r4+15])
        W_higher_hsv_r = np.array([WhighH, 110, WhighV])
    if 100<vmean_r4<=170:
        W_lower_hsv_r4 = np.array([WlowH, WlowS, vmean_r4+20])
    if vmean_r4>170:
        W_lower_hsv_r4 = np.array([WlowH, WlowS, vmean_r4+30])

    yellowmask= cv2.inRange(hsv, Y_lower_hsv, Y_higher_hsv)

    
    # 각 구간에서 설정 된 색 영역으로 차선 필터링
    whitemaskl1= cv2.inRange(hsv[:hsv.shape[0]//4,:hsv.shape[1]//2], W_lower_hsv_l1, W_higher_hsv_l)
    whitemaskl2= cv2.inRange(hsv[hsv.shape[0]//4:(hsv.shape[0]//4)*2,:hsv.shape[1]//2], W_lower_hsv_l2, W_higher_hsv_l)
    whitemaskl3= cv2.inRange(hsv[(hsv.shape[0]//4)*2:(hsv.shape[0]//4)*3,:hsv.shape[1]//2], W_lower_hsv_l3, W_higher_hsv_l)
    whitemaskl4= cv2.inRange(hsv[(hsv.shape[0]//4)*3:(hsv.shape[0]//4)*4,:hsv.shape[1]//2], W_lower_hsv_l4, W_higher_hsv_l)

    whitemaskr1= cv2.inRange(hsv[:hsv.shape[0]//4,hsv.shape[1]//2:], W_lower_hsv_r1, W_higher_hsv_r)
    whitemaskr2= cv2.inRange(hsv[hsv.shape[0]//4:(hsv.shape[0]//4)*2,hsv.shape[1]//2:], W_lower_hsv_r2, W_higher_hsv_r)
    whitemaskr3= cv2.inRange(hsv[(hsv.shape[0]//4)*2:(hsv.shape[0]//4)*3,hsv.shape[1]//2:], W_lower_hsv_r3, W_higher_hsv_r)
    whitemaskr4= cv2.inRange(hsv[(hsv.shape[0]//4)*3:(hsv.shape[0]//4)*4,hsv.shape[1]//2:], W_lower_hsv_r4, W_higher_hsv_r)


    # 각 구간별 다른 색 영역으로 필터링 된 차선을 하나의 마스크에 통합
    mask[:hsv.shape[0]//4,:hsv.shape[1]//2]=whitemaskl1
    mask[hsv.shape[0]//4:(hsv.shape[0]//4)*2,:hsv.shape[1]//2]=whitemaskl2
    mask[(hsv.shape[0]//4)*2:(hsv.shape[0]//4)*3,:hsv.shape[1]//2]=whitemaskl3
    mask[(hsv.shape[0]//4)*3:(hsv.shape[0]//4)*4,:hsv.shape[1]//2]=whitemaskl4

    mask[:hsv.shape[0]//4,hsv.shape[1]//2:]=whitemaskr1
    mask[hsv.shape[0]//4:(hsv.shape[0]//4)*2,hsv.shape[1]//2:]=whitemaskr2
    mask[(hsv.shape[0]//4)*2:(hsv.shape[0]//4)*3,hsv.shape[1]//2:]=whitemaskr3
    mask[(hsv.shape[0]//4)*3:(hsv.shape[0]//4)*4,hsv.shape[1]//2:]=whitemaskr4

    return mask, matrix_inv, output1


def lanedetect(mask,output1,meanwid):
# 마스크의 하단 절반 영역에서 차선이 가장 많이 검출된 x좌표를 시작점으로 하여 Sliding Windows를 쌓아 올라가며 차선을 검출한다
    
    hist=np.sum(mask[mask.shape[0]//2:,:],axis=0) # 마스크의 하단 절반 영역에서 각 x좌표에서의 True 픽셀 수 합
    midpoint=np.int(hist.shape[0]/2) # 마스크의 가로 중심선
    
    # 마스크의 좌/우측 구간과 각 구간에서 True 픽셀이 가장 많은 x좌표(최하단 sliding windows)
    leftx_base=np.argmax(hist[:midpoint])
    leftx_max=np.max(hist[:midpoint])
    rightx_base=np.argmax(hist[midpoint:])+midpoint
    rightx_max=np.max(hist[midpoint:])
    

    nwindows=10 # Sliding Window의 개수 설정
    window_height=np.int(mask.shape[0]/nwindows) # 윈도우 하나의 높이

    # True 픽셀들의 x/y 좌표
    nonzero=mask.nonzero()
    nonzeroy=np.array(nonzero[0])
    nonzerox=np.array(nonzero[1])
    
    # 현재 window의 x좌표를 최하단 windows의 x좌표로 초기화한다.
    leftx_current=leftx_base
    rightx_current=rightx_base

    margin=35 # sliding windows 폭
    # sliding window의 좌표를 갱신하기 위해 필요한 window 내부의 최소/최대 픽셀 수
    minpix=30
    maxpix=110
    # windows 내부의 nonzero pixels 좌표
    left_lane_inds=[]
    right_lane_inds=[]
    # 좌/우측 검출 된 차선의 좌표
    left_wins=[]
    right_wins=[]
    # 좌/우측 windows의 개수
    lefcounts=0
    ritcounts=0

    wid=0


    # 좌/우측 차선의 유무 변수
    leftexists=0
    rightexists=0
    countexpand=0

    curvature = 0
    


    for window in range(nwindows):  # sliding windows를 그리는 과정
        # 좌/우측 구간의 유효 windows 개수 증가 여부
        lefrising=0
        ritrising=0
        # 각 windows 의 좌우상하 모서리 좌표
        win_y_low = mask.shape[0] - (window+1)*window_height
        win_y_high = mask.shape[0] - window*window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin

        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin


        
        if (leftx_max >= 2550): #좌측 구간에 10픽셀 이상 존재 할 경우 (차선이 존재 할 때) 
            leftexists=1

            if (np.sum(((nonzeroy <= win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero())>=2550): # 현재 window의 상단에 10픽셀 이상 존재
                        cv2.rectangle(output1,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
                        left_wins.append([leftx_current,int((win_y_low+win_y_high)/2),1]) # 현재 window의 좌표를 차선 배열에 추가
                        lefrising=1
                        lefcounts+=1

                        if lefcounts==1: # 1번 window 저장 시, 같은 x좌표 프레임 최 하단에  0번 window도 함께 저장
                            left_wins.insert(0,[leftx_current,int(win_y_high+window_height/2),1])





        if (rightx_max >= 2550): #좌측 구간에 10픽셀 이상 존재 할 경우 (차선이 존재 할 때)
            rightexists=1
            if (np.sum(((nonzeroy <= win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero())>=2550): # 현재 window의 상단에 10픽셀 이상 존재
                        cv2.rectangle(output1,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,255), 2)
                        right_wins.append([rightx_current,int((win_y_low+win_y_high)/2),1])# 현재 window의 좌표를 차선 배열에 추가
                        ritrising=1
                        ritcounts+=1
                        if ritcounts==1: # 1번 window 저장 시, 같은 x좌표 프레임 최 하단에  0번 window도 함께 저장
                            right_wins.insert(0,[rightx_current,int(win_y_high+window_height/2),1])

 


        # 좌/우측 windows 내부의 nonzero pixels x좌표
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # window 내부에서 minpix, maxpix 영역 사이의 픽셀을 찾으면 그 평균 좌표로 다음 window의 x좌표를 이동한다.
        if maxpix > len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if maxpix > len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
                
        # 좌/우측의 windows 개수가 다를 경우, 더 많은 window 개수에 맞춰 모자란쪽 window를 생성한다.
                
        # 좌/우측에 공통된 window 가진 영역 까지는 windows 사이의 평균 폭을 구한다
        if (lefcounts==ritcounts)&(lefcounts!=0)&(ritcounts!=0):
            wid+=(rightx_current-leftx_current)
            lowcounts=lefcounts
            countexpand=lefcounts # countexpand : 반대편에 비해 얼마나 더 많은 window가 있는지. (좌 : 1개, 우 : 3개 -> countexpand = 2)
        # 좌/우측 중 한 쪽의 window 만 있는 경우, 이전에서 구한 평균 폭만 큼 이동하여 반대편 window를 만든다
        if (lefcounts!=ritcounts)&(lefcounts!=0)&(ritcounts!=0)&(leftexists==1)&(rightexists==1):
            meanwid=int(wid/lowcounts)
            countexpand+=1
            
            if (lefcounts>ritcounts)&(countexpand<=lefcounts):
                cv2.rectangle(output1,(win_xleft_low+meanwid,win_y_low),(win_xleft_high+meanwid,win_y_high),(0,0,255), 2)
                right_wins.append([leftx_current+meanwid,int((win_y_low+win_y_high)/2),1])
                ritcounts+=1
            if (ritcounts>lefcounts)&(countexpand<=ritcounts):
                cv2.rectangle(output1,(win_xright_low-meanwid,win_y_low),(win_xright_high-meanwid,win_y_high),(0,0,255), 2)
                left_wins.append([rightx_current-meanwid,int((win_y_low+win_y_high)/2),1])
                lefcounts+=1
        # 한 쪽 차선만 검출 될 경우, 이전 프레임의 평균 도로폭을 이용하여 반대 쪽 차선을 생성한다
        if (leftexists==1)&(rightexists==0)&(lefrising==1):# 좌측 차선만 있을 경우
            cv2.rectangle(output1,(win_xleft_low+meanwid,win_y_low),(win_xleft_high+meanwid,win_y_high),(255,0,255), 2)
            right_wins.append([leftx_current+meanwid,int((win_y_low+win_y_high)/2),1])
            ritcounts+=1
            if ritcounts==1:
                right_wins.insert(0,[leftx_current+meanwid,int(win_y_high+window_height/2),1])
        if (leftexists==0)&(rightexists==1)&(ritrising==1):# 우측 차선만 있을 경우
            cv2.rectangle(output1,(win_xright_low-meanwid,win_y_low),(win_xright_high-meanwid,win_y_high),(255,0,255), 2)
            left_wins.append([rightx_current-meanwid,int((win_y_low+win_y_high)/2),1])
            lefcounts+=1
            if lefcounts==1:
                left_wins.insert(0,[rightx_current-meanwid,int(win_y_high+window_height/2),1])

    return left_wins, right_wins, lefcounts, ritcounts, meanwid

def change2src(left_wins,right_wins,lefcounts, ritcounts, matrix_inv):
    # 버드뷰에서 검출한 차선의 좌표를 원본 영상의 좌표계로 전환한다
    birdright=[]
    birdleft=[]

    l_line = line()
    r_line = line()
    bird_l_line = line()
    bird_r_line = line()
    

    # 버드뷰에서 검출한 차선 좌표 행렬을 src 좌표계로 전환하기 위하여 전치 행렬을 생성한다
    left_wins_array=np.array(left_wins)
    left_wins_array=left_wins_array.transpose()
    right_wins_array=np.array(right_wins)
    right_wins_array=right_wins_array.transpose()
 
    if len(right_wins_array)==3: # 우측 차선이 존재 할 때, 차선 좌표 배열과 역변환 매트릭스(matrix_inv)를 사용해 src 좌표계로 전환
                                 # src 좌표계의 차선 좌표 배열 (realright)를 생성한다.
        birdright=right_wins_array[[0,1],:].transpose()
    
        for i in birdright:
            bird_r_line.x.append(i[0])
            bird_r_line.y.append(i[1])
    
        for ritcount in range(ritcounts+1):
            rx=right_wins_array[0][ritcount]
            ry=right_wins_array[1][ritcount]
            rz=right_wins_array[2][ritcount]
            realright=[int((matrix_inv[0][0]*rx+matrix_inv[0][1]*ry+matrix_inv[0][2])/(matrix_inv[2][0]*rx+matrix_inv[2][1]*ry+matrix_inv[2][2])),
                    int((matrix_inv[1][0]*rx+matrix_inv[1][1]*ry+matrix_inv[1][2])/(matrix_inv[2][0]*rx+matrix_inv[2][1]*ry+matrix_inv[2][2]))]
            
            # 원본 좌표계에서의 우측 촤선 좌표 배열
            r_line.x.append(realright[0])
            r_line.y.append(realright[1])
               
    if len(left_wins_array)==3: # 우측 차선이 존재 할 때, 차선 좌표 배열과 역변환 매트릭스(matrix_inv)를 사용해 src 좌표계로 전환
                                 # src 좌표계의 차선 좌표 배열 (realright)를 생성한다.
        birdleft=left_wins_array[[0,1],:].transpose()
        for i in birdleft:
            bird_l_line.x.append(i[0])
            bird_l_line.y.append(i[1])
        
        for lefcount in range(lefcounts+1):
            lx=left_wins_array[0][lefcount]
            ly=left_wins_array[1][lefcount]
            lz=left_wins_array[2][lefcount]
            realleft=[int((matrix_inv[0][0]*lx+matrix_inv[0][1]*ly+matrix_inv[0][2])/(matrix_inv[2][0]*lx+matrix_inv[2][1]*ly+matrix_inv[2][2])),
                    int((matrix_inv[1][0]*lx+matrix_inv[1][1]*ly+matrix_inv[1][2])/(matrix_inv[2][0]*lx+matrix_inv[2][1]*ly+matrix_inv[2][2]))]
            
            # 원본 좌표계에서의 좌측 촤선 좌표 배열
            l_line.x.append(realleft[0])
            l_line.y.append(realleft[1])


    return bird_l_line, bird_r_line, l_line, r_line
        
def list_to_array(l_line, r_line):
    l_line.x = np.array(l_line.x)
    l_line.y = np.array(l_line.y)
    r_line.x = np.array(r_line.x)
    r_line.y = np.array(r_line.y)    

global offset_, curvature_, pre_curvature
curvature_ = 3500
offset_ = 20
pre_curvature = 3000  


videofile='testvideo.avi'
img=cv2.VideoCapture(videofile)
fps=img.get(cv2.CAP_PROP_FPS)
delay=int(1000/fps)

#실시간 영상처리를 위한 real time video source
#img= cv2.VideoCapture(0)
#img= cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

#프레임의 가로/세로 픽셀 수
#width=int(img.get(cv2.CAP_PROP_FRAME_WIDTH))
#height=int(img.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter('result.avi', fourcc, 30.0, (640, 480))

#이전 프레임의 도로폭 저장을 위한 전역 변수
global meanwid
meanwid=90


# color range of White & Yellow
YlowH = 0
YhighH = 0
YlowS = 0
YhighS = 0
YlowV = 0
YhighV = 0

WlowH = 0
WhighH = 179
WlowS = 0
WhighS = 110
WlowV = 133
WhighV = 255
#차선인지 ROI
upx=64
upy=225
dwx=320
dwy=36


mask = np.zeros((300,190), dtype=np.uint8)

firstcheck=0
save_l_line=line()
save_r_line=line()
save_curvature=float()

while(img.isOpened()):
    bird_l_line = line()
    bird_r_line = line()
    l_line = line()
    r_line = line()
   
    #현 프레임 가져오기
    ret,src=img.read()
    
    vehicleDetection()

    mask, matrix_inv, output1 = getbirdviewmask(src,upx,upy,dwx,dwy)

    left_wins,right_wins,lefcounts, ritcounts, meanwid =lanedetect(mask,output1,meanwid)

    bird_l_line, bird_r_line, l_line, r_line = change2src(left_wins,right_wins,lefcounts, ritcounts, matrix_inv)
    

    list_to_array(r_line, l_line)
    list_to_array(bird_r_line, bird_l_line)
    
    if (bird_l_line.x.size > 3 and bird_r_line.x.size > 3):
        firstcheck=1
        curve(bird_l_line)
        curve(bird_r_line)
        curvature = (bird_l_line.curvature + bird_r_line.curvature)/2
        minimap = bird_view_map(bird_l_line, bird_r_line)
        
        
    elif (bird_r_line.x.size > 3):
        curve(bird_r_line)
        curvature = bird_r_line.curvature
    elif (bird_l_line.x.size > 3):
        curve(bird_l_line)
        curvature = bird_l_line.curvature

    

    if (l_line.x.size > 3 and r_line.x.size > 3):
        src, pre_curvature = drawing_lane_handle(l_line, r_line, src, curvature, offset_, curvature_, pre_curvature)
        src = img_result(minimap, src)
        save_l_line=l_line
        save_r_line=r_line
        save_curvature=curvature

    else:
        if firstcheck==1:
            src, pre_curvature = drawing_lane_handle(save_l_line, save_r_line, src, save_curvature, offset_, curvature_, pre_curvature)
            src = img_result(minimap, src) 
    
    
    cv2.imshow('vid',src)
    
    writer.write(src)
    k=cv2.waitKey(delay)
    if k==27:
        break


img.release()
writer.release()
cv2.destroyAllWindows()

