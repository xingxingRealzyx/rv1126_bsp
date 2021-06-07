/*
 * @Descripttion: yolov5.cpp
 * @version: 0.0.1
 * @Author: zyx
 * @Date: 2021-06-04 20:16:39
 * @LastEditors: zyx
 * @LastEditTime: 2021-06-05 16:00:49
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "yolov5.h"



/**
 * @name: yolov5
 * @test: test font
 * @msg: 构造yolov5
 * @param {*}
 * @return {*}
 */ 
yolov5::yolov5()
{

}

/**
 * @name: 
 * @test: test font
 * @msg: 
 * @param {*}
 * @return {*}
 */
yolov5::~yolov5()
{

}


/**
 * @name: load_data
 * @test: test font
 * @msg: 读取数据
 * @param {FILE} *fp
 * @param {size_t} ofst
 * @param {size_t} sz
 * @return {*}
 */
static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}


/**
 * @name: load_model
 * @test: test font
 * @msg: 载入模型
 * @param {char} *model_name
 * @param {int} model_size
 * @return {*}
 */
int yolov5::load_model(char *model_name)
{
    FILE *fp;


    fp = fopen(model_name, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", model_name);
        return -1;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    model_data = load_data(fp, 0, size);

    if (model_data == NULL)
    {
        fclose(fp);
        return -1;
    }
    

    fclose(fp);

    model_data_size = size;

    return 0;
}


/**
 * @name: 
 * @test: test font
 * @msg: 
 * @param {rknn_tensor_attr} *attr
 * @return {*}
 */
static void printRKNNTensor(rknn_tensor_attr *attr)
{
    printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d "
           "fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2],
           attr->dims[1], attr->dims[0], attr->n_elems, attr->size, 0, attr->type,
           attr->qnt_type, attr->fl, attr->zp, attr->scale);
}


/**
 * @name: init
 * @test: test font
 * @msg: 初始化
 * @param {*}
 * @return {*}
 */
int yolov5::init()
{
    int ret;
    ret = rknn_init(&ctx, model_data, model_data_size, 0);
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version,
                     sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);
    
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }
        printRKNNTensor(&(input_attrs[i]));
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }
        printRKNNTensor(&(output_attrs[i]));
    }

    int channel = 3;
    int width = 0;
    int height = 0;
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        width = input_attrs[0].dims[0];
        height = input_attrs[0].dims[1];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        width = input_attrs[0].dims[1];
        height = input_attrs[0].dims[2];
    }

    printf("model input height=%d, width=%d, channel=%d\n", height, width, channel);

    

}

/**
 * @name: deinit
 * @test: test font
 * @msg: 退出
 * @param {*}
 * @return {*}
 */
int yolov5::deinit()
{

}

/**
 * @name: inference
 * @test: test font
 * @msg: 推理
 * @param {*}
 * @return {*}
 */
int yolov5::inference()
{

}
