/*
 * =======================================================================================
 *
 *      Filename:  appDaemon.c
 *
 *      Description:  Implementation a interface library to hook into applications
 *                    using the GOTCHA library
 *
 *      Version:   5.1.1
 *      Released:  31.03.2021
 *
 *      Author:   Thoams Roehl (tr), thomas.roehl@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2021 RRZE, University Erlangen-Nuremberg
 *
 *      This program is free software: you can redistribute it and/or modify it under
 *      the terms of the GNU General Public License as published by the Free Software
 *      Foundation, either version 3 of the License, or (at your option) any later
 *      version.
 *
 *      This program is distributed in the hope that it will be useful, but WITHOUT ANY
 *      WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 *      PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 *      You should have received a copy of the GNU General Public License along with
 *      this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * =======================================================================================
 */


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
