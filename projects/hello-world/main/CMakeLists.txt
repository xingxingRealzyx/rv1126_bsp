list(APPEND ADD_DEFINITIONS -g -O0 -ggdb -gdwarf -funwind-tables -rdynamic)
############### Add include ###################
list(APPEND ADD_INCLUDE "include"
    )
list(APPEND ADD_PRIVATE_INCLUDE "")
###############################################

############ Add source files #################
# list(APPEND ADD_SRCS  "src/main.c"
#                       "src/test.c"
#     )
# aux_source_directory(src ADD_SRCS)  # collect all source file in src dir, will set var ADD_SRCS
append_srcs_dir(ADD_SRCS "src")       # append source file in src dir to var ADD_SRCS
# list(REMOVE_ITEM COMPONENT_SRCS "src/test2.c")
###############################################


set(OpenCV_DIR ./../../../components/opencv/share/OpenCV)

find_package(OpenCV REQUIRED)


###### Add required/dependent components ######

list(APPEND ADD_REQUIREMENTS bsp)
list(APPEND ADD_REQUIREMENTS npu)
list(APPEND ADD_REQUIREMENTS dl)
list(APPEND ADD_REQUIREMENTS ${OpenCV_LIBS})
###############################################

############ Add static libs ##################
# list(APPEND ADD_STATIC_LIB "lib/libtest.a")
###############################################

############ Add dynamic libs ##################
# list(APPEND ADD_DYNAMIC_LIB "lib/librknn_api.so"
                            
# )
###############################################

#### Add compile option for this component ####
#### Just for this component, won't affect other 
#### modules, including component that depend 
#### on this component
# list(APPEND ADD_DEFINITIONS_PRIVATE -DAAAAA=1)

#### Add compile option for this component
#### and components denpend on this component
# list(APPEND ADD_DEFINITIONS  )
###############################################



register_component()

