// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include "global_config.h"
#include "global_build_info_time.h"
#include "global_build_info_version.h"


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <dlfcn.h>

#define _BASETSD_H

#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "drm_func.h"
#include "rga_func.h"
#include "rknn_api.h"
#include "postprocess.h"

#include "rkmedia_api.h"

#define PERF_WITH_POST 1


#define SCREEN_WIDTH 720
#define SCREEN_HEIGHT 1280

/*-------------------------------------------
                  Functions
-------------------------------------------*/

static void printRKNNTensor(rknn_tensor_attr *attr)
{
    printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d "
           "fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2],
           attr->dims[1], attr->dims[0], attr->n_elems, attr->size, 0, attr->type,
           attr->qnt_type, attr->fl, attr->zp, attr->scale);
}
double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

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

static unsigned char *load_model(const char *filename, int *model_size)
{

    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

static int saveFloat(const char *file_name, float *output, int element_size)
{
    FILE *fp;
    fp = fopen(file_name, "w");
    for (int i = 0; i < element_size; i++)
    {
        fprintf(fp, "%.6f\n", output[i]);
    }
    fclose(fp);
    return 0;
}



static void yolov5_inference(char *model_name)
{

    /* Create the neural network */
    printf("Loading mode...\n");
    int model_data_size = 0;
    unsigned char *model_data = load_model(model_name, &model_data_size);
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
    printf("sdk version: %s driver version: %s\n", version.api_version,
           version.drv_version);

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input,
           io_num.n_output);

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]),
                         sizeof(rknn_tensor_attr));
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
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]),
                         sizeof(rknn_tensor_attr));
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

    printf("model input height=%d, width=%d, channel=%d\n", height, width,
           channel);

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = width * height * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    
    // DRM alloc buffer
    drm_fd = drm_init(&drm_ctx);
    drm_buf = drm_buf_alloc(&drm_ctx, drm_fd, img_width, img_height, channel * 8,
                            &buf_fd, &handle, &actual_size);
    memcpy(drm_buf, orig_img.data, img_width * img_height * channel);
    void *resize_buf = malloc(height * width * channel);

    // init rga context
    RGA_init(&rga_ctx);
    img_resize_slow(&rga_ctx, drm_buf, img_width, img_height, resize_buf, width, height);
    inputs[0].buf = resize_buf;
    gettimeofday(&start_time, NULL);
    rknn_inputs_set(ctx, io_num.n_input, inputs);

    rknn_output outputs[io_num.n_output];


    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        outputs[i].want_float = 0;
    }

    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
    gettimeofday(&stop_time, NULL);
    printf("once run use %f ms\n",
           (__get_us(stop_time) - __get_us(start_time)) / 1000);

    //post process
    float scale_w = (float)width / img_width;
    float scale_h = (float)height / img_height;

    detect_result_group_t detect_result_group;
    std::vector<float> out_scales;
    std::vector<uint8_t> out_zps;
    for (int i = 0; i < io_num.n_output; ++i)
    {
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }
    post_process((uint8_t *)outputs[0].buf, (uint8_t *)outputs[1].buf, (uint8_t *)outputs[2].buf, height, width,
                 conf_threshold, nms_threshold, vis_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);


}


/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main(int argc, char **argv)
{
    int status = 0;
    char *model_name = NULL;
    rknn_context ctx;
    void *drm_buf = NULL;
    int drm_fd = -1;
    int buf_fd = -1; // converted from buffer handle
    unsigned int handle;
    size_t actual_size = 0;
    int img_width = 0;
    int img_height = 0;
    int img_channel = 0;
    rga_context rga_ctx;
    drm_context drm_ctx;
    const float vis_threshold = 0.1;
    const float nms_threshold = 0.5;
    const float conf_threshold = 0.3;
    struct timeval start_time, stop_time;
    int ret;
    memset(&rga_ctx, 0, sizeof(rga_context));
    memset(&drm_ctx, 0, sizeof(drm_context));


    /*  initial vo */
    RK_CHAR *pDeviceName = "/dev/dri/card0";
    RK_CHAR *u32PlaneType = "Primary";
    RK_U8 u8EnableRandom = 0;
    RK_S32 s32Xpos = 0;
    RK_S32 s32Ypos = 0;
    RK_U32 u32Fps = 0;
    RK_U32 u32Zpos = 0;
    RK_U32 u32DispWidth = 720;
    RK_U32 u32DispHeight = 1280;
    RK_FLOAT fltImgRatio = 0.0;
    RK_U32 u32FrameSize = 0;

    printf("#Device: %s\n", pDeviceName);
    printf("#PlaneType: %s\n", u32PlaneType);
    printf("#Display RandomPos: %s\n", u8EnableRandom ? "Enable" : "Disable");
    printf("#Display Xpos: %d\n", s32Xpos);
    printf("#Display Ypos: %d\n", s32Ypos);
    printf("#Display Width: %u\n", u32DispWidth);
    printf("#Display Height: %u\n", u32DispHeight);
    printf("#Display Zpos: %u\n", u32Zpos);
    printf("#Display Fps: %u\n", u32Fps);

    RK_MPI_SYS_Init();

    VO_CHN_ATTR_S stVoAttr = {0};
    stVoAttr.pcDevNode = pDeviceName;
    stVoAttr.u32Width = SCREEN_WIDTH;
    stVoAttr.u32Height = SCREEN_HEIGHT;
    stVoAttr.u16Fps = u32Fps;
    stVoAttr.u16Zpos = u32Zpos;
    stVoAttr.stImgRect.s32X = 0;
    stVoAttr.stImgRect.s32Y = 0;
    stVoAttr.stImgRect.u32Width = u32DispWidth;
    stVoAttr.stImgRect.u32Height = u32DispHeight;
    stVoAttr.stDispRect.s32X = s32Xpos;
    stVoAttr.stDispRect.s32Y = s32Ypos;
    stVoAttr.stDispRect.u32Width = u32DispWidth;
    stVoAttr.stDispRect.u32Height = u32DispHeight;
    if (!strcmp(u32PlaneType, "Primary")) {
    // VO[0] for primary plane
    stVoAttr.emPlaneType = VO_PLANE_PRIMARY;
    stVoAttr.enImgType = IMAGE_TYPE_RGB888;
    fltImgRatio = 3.0; // for RGB888
    } else if (!strcmp(u32PlaneType, "Overlay")) {
    // VO[0] for overlay plane
    stVoAttr.emPlaneType = VO_PLANE_OVERLAY;
    stVoAttr.enImgType = IMAGE_TYPE_NV12;
    fltImgRatio = 1.5; // for NV12
    } else {
    printf("ERROR: Unsupport plane type:%s\n", u32PlaneType);
    return -1;
    }

    ret = RK_MPI_VO_CreateChn(0, &stVoAttr);
    if (ret) {
        printf("Create vo[0] failed! ret=%d\n", ret);
        return -1;
    }

    // Create buffer pool. Note that VO only support dma buffer.
    MB_POOL_PARAM_S stBufferPoolParam;
    stBufferPoolParam.u32Cnt = 3;
    stBufferPoolParam.u32Size =
        0; // Automatic calculation using imgInfo internally
    stBufferPoolParam.enMediaType = MB_TYPE_VIDEO;
    stBufferPoolParam.bHardWare = RK_TRUE;
    stBufferPoolParam.u16Flag = MB_FLAG_NOCACHED;
    stBufferPoolParam.stImageInfo.enImgType = stVoAttr.enImgType;
    stBufferPoolParam.stImageInfo.u32Width = SCREEN_WIDTH;
    stBufferPoolParam.stImageInfo.u32Height = SCREEN_HEIGHT;
    stBufferPoolParam.stImageInfo.u32HorStride = SCREEN_WIDTH;
    stBufferPoolParam.stImageInfo.u32VerStride = SCREEN_HEIGHT;

    MEDIA_BUFFER_POOL mbp = RK_MPI_MB_POOL_Create(&stBufferPoolParam);
    if (!mbp) {
        printf("Create buffer pool for vo failed!\n");
        return -1;
    }

    printf("%s initial  vo finish\n", __func__);

    RK_U32 u32FrameId = 0;
    RK_U64 u64TimePeriod = 33333; // us
    MB_IMAGE_INFO_S stImageInfo = {0};
    u32FrameSize = (RK_U32)(u32DispWidth * u32DispHeight * fltImgRatio);

    MEDIA_BUFFER mb = RK_MPI_MB_POOL_GetBuffer(mbp, RK_TRUE);
    if (!mb) {
      printf("WARN: BufferPool get null buffer...\n");
      usleep(10000);
    }


    // resize image wxh.
    stImageInfo.u32Width = u32DispWidth;
    stImageInfo.u32Height = u32DispHeight;
    stImageInfo.u32HorStride = u32DispWidth;
    stImageInfo.u32VerStride = u32DispHeight;
    stImageInfo.enImgType = stVoAttr.enImgType;

    mb = RK_MPI_MB_ConvertToImgBuffer(mb, &stImageInfo);
    if (!mb) {
        printf("ERROR: convert to img buffer failed!\n");

    }

    memset(RK_MPI_MB_GetPtr(mb), 0x00, u32FrameSize);
    RK_MPI_MB_SetSize(mb, u32FrameSize);




    model_name = "model/rv1109_rv1126/yolov5s_relu_rv1109_rv1126_out_opt.rknn";
    char *image_name = "model/test2.jpg";

    printf("Read %s ...\n", image_name);
    cv::Mat orig_img = cv::imread(image_name, 1);
    if (!orig_img.data)
    {
        printf("cv::imread %s fail!\n", image_name);
        return -1;
    }
    img_width = orig_img.cols;
    img_height = orig_img.rows;
    printf("img width = %d, img height = %d\n", img_width, img_height);

    









    // Draw Objects
    for (int i = 0; i < detect_result_group.count; i++)
    {
        detect_result_t *det_result = &(detect_result_group.results[i]);
        printf("%s @ (%d %d %d %d) %f\n",
               det_result->name,
               det_result->box.left, det_result->box.top, det_result->box.right, det_result->box.bottom,
               det_result->prop);
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;
        rectangle(orig_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0, 255), 3);
        putText(orig_img, det_result->name, cv::Point(x1, y1 + 12), 1, 2, cv::Scalar(0, 255, 0, 255));
    }

    imwrite("./out.jpg", orig_img);
    ret = rknn_outputs_release(ctx, io_num.n_output, outputs);

    // loop test
    int test_count = 10;
    gettimeofday(&start_time, NULL);
    for (int i = 0; i < test_count; ++i)
    {
        img_resize_slow(&rga_ctx, drm_buf, img_width, img_height, resize_buf, width, height);
        rknn_inputs_set(ctx, io_num.n_input, inputs);
        ret = rknn_run(ctx, NULL);
        ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
#if PERF_WITH_POST
        post_process((uint8_t *)outputs[0].buf, (uint8_t *)outputs[1].buf, (uint8_t *)outputs[2].buf, height, width,
                    conf_threshold, nms_threshold, vis_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);
#endif
        ret = rknn_outputs_release(ctx, io_num.n_output, outputs);
    }
    gettimeofday(&stop_time, NULL);
    printf("loop count = %d , average run  %f ms\n", test_count,
           (__get_us(stop_time) - __get_us(start_time)) / 1000.0 / test_count);

    // release
    ret = rknn_destroy(ctx);
    drm_buf_destroy(&drm_ctx, drm_fd, buf_fd, handle, drm_buf, actual_size);

    drm_deinit(&drm_ctx, drm_fd);
    RGA_deinit(&rga_ctx);
    if (model_data)
    {
        free(model_data);
    }

    if (resize_buf)
    {
        free(resize_buf);
    }


    stVoAttr.stImgRect.s32X = 0;
    stVoAttr.stImgRect.s32Y = 0;
    stVoAttr.stImgRect.u32Width = u32DispWidth;
    stVoAttr.stImgRect.u32Height = u32DispHeight;
    stVoAttr.stDispRect.s32X = s32Xpos;
    stVoAttr.stDispRect.s32Y = s32Ypos;
    stVoAttr.stDispRect.u32Width = u32DispWidth;
    stVoAttr.stDispRect.u32Height = u32DispHeight;
    ret = RK_MPI_VO_SetChnAttr(0, &stVoAttr);
    if (ret) {
        printf("Set VO Attr:%d, %d, %d, %d failed!\n", s32Xpos, s32Ypos,
            u32DispWidth, u32DispHeight);

    }


    ret = RK_MPI_SYS_SendMediaBuffer(RK_ID_VO, 0, mb);
    if (ret) {
      printf("ERROR: RK_MPI_SYS_SendMediaBuffer to VO[0] failed! ret=%d\n",
             ret);
      RK_MPI_MB_ReleaseBuffer(mb);
    }

    usleep(100000000);

    RK_MPI_MB_ReleaseBuffer(mb);


    RK_MPI_VO_DestroyChn(0);
    // buffer pool destroy should after destroy vo.
    RK_MPI_MB_POOL_Destroy(mbp);


    return 0;
}
