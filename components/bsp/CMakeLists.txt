################# Add include #################
list(APPEND ADD_INCLUDE "include/rga"
                        "include/rkaiq"
                        "include/rkaiq/algos"
                        "include/rkmedia"
                        "include/drm"
                        "include/drm/libdrm"
    )
# list(APPEND ADD_PRIVATE_INCLUDE "include_private")
###############################################

############## Add source files ###############
list(APPEND ADD_SRCS  "src/bsp.c"
    )
# aux_source_directory(src ADD_SRCS)  # collect all source file in src dir, will set var ADD_SRCS
# append_srcs_dir(ADD_SRCS "src")     # append source file in src dir to var ADD_SRCS
# list(REMOVE_ITEM COMPONENT_SRCS "src/test.c")
###############################################

###### Add required/dependent components ######
# list(APPEND ADD_REQUIREMENTS component1)
###############################################

############ Add static libs ##################
# if(CONFIG_COMPONENT1_INCLUDE_STATIC_LIB)
#     list(APPEND ADD_STATIC_LIB "lib/libtest.a")
# endif()
###############################################

############ Add dynamic libs ##################
list(APPEND ADD_DYNAMIC_LIB 
                            "lib/libasound.so"
                            "lib/libavcodec.so"
                            "lib/libavformat.so"
                            "lib/libavutil.so"
                            "lib/libdrm.so"
                            "lib/libeasymedia.so"
                            "lib/libmd_share.so"
                            "lib/libod_share.so"
                            "lib/librga.so"
                            "lib/librkaiq.so"
                            "lib/libRKAP_3A.so"
                            "lib/libRKAP_ANR.so"
                            "lib/libRKAP_Common.so"
                            "lib/librockchip_mpp.so"
                            "lib/libswresample.so"
                            "lib/libv4l2.so"
                            "lib/libv4lconvert.so"
)
###############################################

#### Add compile option for this component ####
#### Just for this component, won't affect other 
#### modules, including component that depend 
#### on this component
# list(APPEND ADD_DEFINITIONS_PRIVATE -DAAAAA=1)

#### Add compile option for this component
#### and components denpend on this component
# list(APPEND ADD_DEFINITIONS -DAAAAA222=1
#                             -DAAAAA333=1)
###############################################


register_component()

