#include <stdio.h>
#include <stdlib.h>
#include <gotcha/gotcha.h>

gotcha_wrappee_handle_t orig_main_handle;

static int appDaemon_initialized = 0;

int likwid_appDaemon_main(int argc, char** argv)
{
    int return_code = 0;
    typeof(&likwid_appDaemon_main) orig_main = (int (*)(int, char**))gotcha_get_wrappee(orig_main_handle);
    char* nvEventStr = getenv("NVMON_EVENTS");
    char* nvGpuStr = getenv("NVMON_GPUS");

    if (appDaemon_initialized)
    {
        return_code = orig_main(argc, argv);
    }
    else
    {

        appDaemon_initialized = 1;


        return_code = orig_main(argc, argv);
    }






    appDaemon_initialized = 0;
    return return_code;
}


struct gotcha_binding_t likwid_appDaemon_overwrites[] = {
  {"main", likwid_appDaemon_main, (void*)&orig_main_handle},
};


void __attribute__((constructor)) likwid_appDaemon_constructor()
{
    gotcha_wrap(likwid_appDaemon_overwrites, 1 ,"likwid_appDaemon");
}
