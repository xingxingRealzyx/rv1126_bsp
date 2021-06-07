/*
 * @Descripttion: yolov5.h
 * @version: 0.0.1
 * @Author: zyx
 * @Date: 2021-06-04 20:16:30
 * @LastEditors: zyx
 * @LastEditTime: 2021-06-05 15:33:41
 */
#ifndef __YOLOV5_H__
#define __YOLOV5_H__
#include "rknn_api.h"

class yolov5 
{
public:

    yolov5();
    ~yolov5();
    int load_model(char *model_name);
    int init();
    int deinit();
    int inference();

private:
    
    unsigned char *model_data;
    int model_data_size;
    rknn_context ctx;
};


#endif /* __YOLOV5_H__ */