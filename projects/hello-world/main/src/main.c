/*-------------------------------------------
                Includes
-------------------------------------------*/

#include "global_config.h"
#include "global_build_info_time.h"
#include "global_build_info_version.h"

#include <stdio.h>
#include "hello.h"


#include "rknn_api.h"


/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main(int argc, char **argv)
{
    hello();
    return 0;
}


void hello()
{
    printf("Hello rv1126");
}