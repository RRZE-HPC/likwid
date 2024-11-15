/*
 * =======================================================================================
 *
 *      Filename:  sysfsDaemon.c
 *
 *      Description:  Implementation of access daemon for procfs and sysfs files.
 *
 *      Version:   5.4.0
 *      Released:  15.11.2024
 *
 *      Authors:  Michael Meier, michael.meier@rrze.fau.de
 *                Jan Treibig (jt), jan.treibig@gmail.com,
 *                Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2024 RRZE, University Erlangen-Nuremberg
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

/* #####   HEADER FILE INCLUDES   ######################################### */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <fcntl.h>
#include <syslog.h>
#include <signal.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <sys/fsuid.h>
#include <getopt.h>
#include <dirent.h>
#include <sys/mman.h>
#include <glob.h>
#include <libgen.h>

#include <lock.h>
#include <likwid.h>
#include <access_sysfs_client.h>
//#include <error.h>
//#include <sysfs_client.h>

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */

#define SA struct sockaddr
#define str(x) #x

#define CHECK_FILE_ERROR(func, msg)  \
    if ((func) == 0) { syslog(LOG_ERR, "ERROR - [%s:%d] " str(msg) " - %s \n", __FILE__, __LINE__, strerror(errno)); }

#define LOG_AND_EXIT_IF_ERROR(func, msg)  \
    if ((func) < 0) {  \
        syslog(LOG_ERR, "ERROR - [%s:%d] " str(msg) " - %s \n", __FILE__, __LINE__, strerror(errno)); \
        exit(EXIT_FAILURE); \
    }

#define CHECK_ERROR(func, msg)  \
    if ((func) < 0) { \
        fprintf(stderr, "ERROR - [%s:%d] " str(msg) " - %s \n", __FILE__, __LINE__, strerror(errno));  \
    }

#define PCI_ROOT_PATH    "/proc/bus/pci/"
#define CPUFREQ_DIR_FORMAT "/sys/devices/system/cpu/cpu%d/cpufreq"
#define PROCFS_SYS_KERNEL_DIR "/proc/sys/kernel"

/* Lock file controlled from outside which prevents likwid to start.
 * Can be used to synchronize access to the hardware counters
 * with an external monitoring system. */

/* #####   TYPE DEFINITIONS   ########### */

typedef enum {
    SYSFSD_ACCESS_TYPE_ABS,
    SYSFSD_ACCESS_TYPE_GLOB,
} SysfsDAccessType;

typedef enum {
    SYSFSD_ACCESS_PERM_RDONLY,
    SYSFSD_ACCESS_PERM_RW,
    SYSFSD_ACCESS_PERM_WRONLY,
} SysfsDAccessPermissions;

typedef struct {
    char* file;
    SysfsDAccessType type;
    SysfsDAccessPermissions perm;
    LikwidDeviceType dev_type;
} SysfsDAccessFileDefinition;

typedef struct {
    char* dir;
    LikwidDeviceType dev_type;
    SysfsDAccessType type;
    SysfsDAccessFileDefinition* files;
    int num_files;
    char* fnmatch;
} SysfsDAccessDirDefinition;


static SysfsDAccessFileDefinition valid_procfs_files[] = {
    {"numa_balancing", SYSFSD_ACCESS_TYPE_ABS, SYSFSD_ACCESS_PERM_RW},
    {NULL},
};

static SysfsDAccessFileDefinition valid_cpufreq_files[] ={
    {"scaling_cur_freq", SYSFSD_ACCESS_TYPE_ABS, SYSFSD_ACCESS_PERM_RW},
    {"scaling_max_freq", SYSFSD_ACCESS_TYPE_ABS, SYSFSD_ACCESS_PERM_RW},
    {"scaling_min_freq", SYSFSD_ACCESS_TYPE_ABS, SYSFSD_ACCESS_PERM_RW},
    {"scaling_available_frequencies", SYSFSD_ACCESS_TYPE_ABS, SYSFSD_ACCESS_PERM_RDONLY},
    {"scaling_available_governors", SYSFSD_ACCESS_TYPE_ABS, SYSFSD_ACCESS_PERM_RDONLY},
    {"scaling_driver", SYSFSD_ACCESS_TYPE_ABS, SYSFSD_ACCESS_PERM_RDONLY},
    {"scaling_setspeed", SYSFSD_ACCESS_TYPE_ABS, SYSFSD_ACCESS_PERM_RW},
    {"scaling_governor", SYSFSD_ACCESS_TYPE_ABS, SYSFSD_ACCESS_PERM_RW},
    {"energy_performance_preference", SYSFSD_ACCESS_TYPE_ABS, SYSFSD_ACCESS_PERM_RW},
    {"energy_performance_available_preferences", SYSFSD_ACCESS_TYPE_ABS, SYSFSD_ACCESS_PERM_RDONLY},
    {NULL},
};

static SysfsDAccessDirDefinition default_access_dirs[] = {
    {"/proc/sys/kernel", DEVICE_TYPE_NODE, SYSFSD_ACCESS_TYPE_ABS, valid_procfs_files, 1},
    {"/sys/devices/system/cpu/cpu*/cpufreq", DEVICE_TYPE_HWTHREAD, SYSFSD_ACCESS_TYPE_GLOB, valid_cpufreq_files, 10},
    {NULL},
};





/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

static int sockfd = -1;
static int connfd = -1; /* temporary in to make it compile */
static char* filepath;
static const char* ident = "sysfsD";

/*static OpenFiles *open_files = NULL;*/
/*static int num_open_files = 0;*/
static SysfsDAccessDirDefinition** access_dirs = NULL;
static int num_access_dirs = 0;


/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

static int resolve_dir_glob(SysfsDAccessDirDefinition* format, int* num_out, char*** out)
{
    int err = 0;
    int listlen = 0;
    char** list = NULL;

    if ((!format) || (format->dev_type == DEVICE_TYPE_INVALID) || (!num_out) || (!out))
    {
        return -EINVAL;
    }

    glob_t globbuf;
    globbuf.gl_pathc = 0;
    globbuf.gl_offs = 0;

    err = glob(format->dir, GLOB_ONLYDIR, NULL, &globbuf);
    switch (err)
    {
        case GLOB_NOSPACE:
            return -ENOMEM;
            break;
        case GLOB_ABORTED:
            return -EBADFD;
            break;
        case GLOB_NOMATCH:
            *num_out = 0;
            *out = NULL;
            return 0;
            break;
        default:
            break;
    }
    
    list = malloc(globbuf.gl_pathc * sizeof(char*));
    if (!list)
    {
        globfree(&globbuf);
        return -ENOMEM;
    }
    listlen = 0;

    for (int i = 0; i < globbuf.gl_pathc; i++)
    {
        list[i] = malloc((strlen(globbuf.gl_pathv[i])+2)* sizeof(char));
        if (!list[i])
        {
            for (int j = 0; j < i; j++)
            {
                free(list[j]);
            }
            free(list);
            globfree(&globbuf);
            return -ENOMEM;
        }
        err = snprintf(list[i], strlen(globbuf.gl_pathv[i])+1, "%s", globbuf.gl_pathv[i]);
        if (err > 0)
        {
            list[i][err] = '\0';
        }
        listlen++;
    }
    globfree(&globbuf);
    *num_out = listlen;
    *out = list;
    return 0;

}

static int init_access_dirs_from_file(char* configFile)
{
    return -ENOSYS;
}


static int _init_access_dirs_from_defaults_new(char* dir, SysfsDAccessDirDefinition* def, SysfsDAccessDirDefinition* out)
{

    if ((!dir) || (!def) || (!out))
    {
        return -EINVAL;
    }

    out->dir = malloc((strlen(dir)+2) * sizeof(char));
    if (!out->dir)
    {
        return -ENOMEM;
    }
    int ret = snprintf(out->dir, strlen(dir)+1, "%s", dir);
    if (ret < 0)
    {
        return ret;
    }
    out->dir[ret] = '\0';
    out->type = SYSFSD_ACCESS_TYPE_ABS;
    out->dev_type = def->dev_type;

    return 0;
}

static int _init_access_dirs_from_defaults_register(SysfsDAccessDirDefinition* def, int *index)
{
    if ((!def) || (!index) || (!def->dir))
    {
        return -EINVAL;
    }
    
    SysfsDAccessDirDefinition** tmp = realloc(access_dirs, (num_access_dirs+1)* sizeof(SysfsDAccessDirDefinition*));
    if (!tmp)
    {
        return -ENOMEM;
    }
    access_dirs = tmp;
    int idx = num_access_dirs;
    access_dirs[num_access_dirs++] = def;
    *index = idx;
    return 0;
}

static int _init_access_dirs_from_defaults_files(SysfsDAccessDirDefinition* def, SysfsDAccessDirDefinition* out)
{
    int i = 0;
    int num_files = 0;
    int outidx = 0;

    while (def->files[i].file != NULL)
    {
        num_files++;
        i++;
    }

    out->files = malloc((num_files + 1) * sizeof(SysfsDAccessFileDefinition));
    if (!out->files)
    {
        return -ENOMEM;
    }

    for (i = 0; i < num_files; i++)
    {
        char fname[1025];
        int ret = snprintf(fname, 1024, "%s/%s", out->dir, def->files[i].file);
        if (ret < 0)
        {
            for (int j = 0; j < i-1; j++)
            {
                free(out->files[j].file);
            }
            free(out->files);
            return -ENOMEM;
        }
        if (!access(fname, F_OK))
        {
            out->files[i].file = malloc((strlen(def->files[i].file) + 2) * sizeof(char));
            if (!out->files[i].file)
            {
                for (int j = 0; j < i-1; j++)
                {
                    free(out->files[j].file);
                }
                free(out->files);
                return -ENOMEM;
            }
            ret = snprintf(out->files[i].file, strlen(def->files[i].file) + 1, "%s", def->files[i].file);
            if (ret < 0)
            {
                free(out->files[i].file);
                for (int j = 0; j < i-1; j++)
                {
                    free(out->files[j].file);
                }
                free(out->files);
                return -ENOMEM;
            }
            out->files[i].file[ret] = '\0';
            out->files[i+1].file = NULL;
        }
    }
    out->num_files = num_files;
    return 0;
}

static int init_access_dirs_from_defaults()
{
    int i = 0, j = 0, idx = 0;
    int err = 0;
    int num_dirs = 0;
    char** dirs = NULL;
    i = 0;
    while (default_access_dirs[i].dir != NULL)
    {
        SysfsDAccessDirDefinition* def;
        switch (default_access_dirs[i].type)
        {
            case SYSFSD_ACCESS_TYPE_ABS:
                def = malloc(sizeof(SysfsDAccessDirDefinition));
                if (!def)
                {
                    return -ENOMEM;
                }
                err = _init_access_dirs_from_defaults_new(default_access_dirs[i].dir, &default_access_dirs[i], def);
                if (err < 0)
                {
                    free(def);
                    return err;
                }
                err = _init_access_dirs_from_defaults_files(&default_access_dirs[i], def);
                if (err < 0)
                {
                    free(def);
                    return err;
                }
                err = _init_access_dirs_from_defaults_register(def, &idx);
                if (err < 0)
                {
                    j = 0;
                    while (def->files[j].file != NULL)
                    {
                        free(def->files[j].file);
                        j++;
                    }
                    free(def);
                    return err;
                }
                break;
            case SYSFSD_ACCESS_TYPE_GLOB:
                err = resolve_dir_glob(&default_access_dirs[i], &num_dirs, &dirs);
                if (err < 0)
                {
                    return err;
                }
                for (j = 0; j < num_dirs; j++)
                {
                    def = malloc(sizeof(SysfsDAccessDirDefinition));
                    if (!def)
                    {
                        return -ENOMEM;
                    }
                    err = _init_access_dirs_from_defaults_new(dirs[j], &default_access_dirs[i], def);
                    if (err < 0)
                    {
                        free(def);
                        return err;
                    }
                    err = _init_access_dirs_from_defaults_files(&default_access_dirs[i], def);
                    if (err < 0)
                    {
                        free(def);
                        return err;
                    }
                    err = _init_access_dirs_from_defaults_register(def, &idx);
                    if (err < 0)
                    {
                        j = 0;
                        while (def->files[j].file != NULL)
                        {
                            free(def->files[j].file);
                            j++;
                        }
                        free(def);
                        return err;
                    }
                }
                if (dirs)
                {
                    for (j = 0; j < num_dirs; j++)
                    {
                        free(dirs[j]);
                    }
                    free(dirs);
                    dirs = NULL;
                    num_dirs = 0;
                }
        }
        i++;
    }
    return 0;
}

static int init_access_dirs(char* configFile)
{
    return (configFile == NULL ? init_access_dirs_from_defaults() : init_access_dirs_from_file(configFile));
}

static void cleanup_access_dirs()
{
    if (access_dirs)
    {
        for (int i = 0; i < num_access_dirs; i++)
        {
            SysfsDAccessDirDefinition* dir = access_dirs[i];
            for (int j = 0; j < dir->num_files; j++)
            {
                free(dir->files[j].file);
            }
            free(dir->files);
            free(dir->dir);
            free(dir);
        }
        free(access_dirs);
        access_dirs = NULL;
        num_access_dirs = 0;
    }
}


static int check_file_access(char* filename)
{
    for (int i = 0; i < num_access_dirs; i++)
    {
        SysfsDAccessDirDefinition* dir = access_dirs[i];
        if (strncmp(filename, dir->dir, strlen(dir->dir)) == 0)
        {
            for (int j = 0; j < dir->num_files; j++)
            {
                if (strncmp(&filename[strlen(dir->dir)+1], dir->files[j].file, strlen(dir->files[j].file)) == 0)
                {
                    return 1;
                }
            }
        }
    }
    return 0;
}

static int read_sysfs_file(char* filename, int max_len, char* data)
{
    int err = check_file_access(filename);
    if (!err)
    {
        return -ENOENT;
    }
    FILE* fp = fopen(filename, "r");
    if (fp)
    {
        int ret = fread(data, sizeof(char), max_len, fp);
        if (ret >= 0)
        {
            data[ret] = '\0';
        }
        for (int k = ret-1; k >= 0; k--)
        {
            if (data[k] == '\n')
            {
                data[k] = '\0';
                break;
            }
        }
        fclose(fp);
    }
    return -0;
}

static int write_sysfs_file(char* filename, int max_len, char* data)
{
    int err = check_file_access(filename);
    if (!err)
    {
        return -ENOENT;
    }
    FILE* fp = fopen(filename, "w");
    if (fp)
    {
        fwrite(data, sizeof(char), max_len, fp);
        fclose(fp);
    }
    return -0;
}



/*int main(int argc, char* argv[])*/
/*{*/
/*    char data[1025];*/
/*    char fname[] = "/sys/devices/system/cpu/cpu1/cpufreq/scaling_min_freq";*/
/*    int err = init_access_dirs(NULL);*/
/*    if (err < 0)*/
/*    {*/
/*        return err;*/
/*    }*/
/*    */
/*    err = read_sysfs_file(fname, 1024, data);*/
/*    if (err < 0)*/
/*    {*/
/*        return err;*/
/*    }*/
/*    printf("Data '%s'\n", data);*/
/*    cleanup_access_dirs();*/
/*    return 0;*/
/*}*/

static void
kill_client(void)
{
    if (connfd != -1)
    {
        CHECK_ERROR(close(connfd), socket close failed);
    }

    connfd = -1;
}

static void
stop_daemon(void)
{
    kill_client();

    cleanup_access_dirs();

    closelog();
    exit(EXIT_SUCCESS);
}



static void
Signal_Handler(int sig)
{
    if (sig == SIGPIPE)
    {
        syslog(LOG_NOTICE, "SIGPIPE? client crashed?!");
        stop_daemon();
    }

    /* For SIGALRM we just return - we're just here to create a EINTR */
    if (sig == SIGTERM)
    {
        stop_daemon();
    }
}

static void
daemonize(int* parentPid)
{
    pid_t pid, sid;

    *parentPid = getpid();

    /* already a daemon */
    if ( getppid() == 1 ) return;

    /* Fork off the parent process */
    pid = fork();

    if (pid < 0)
    {
        syslog(LOG_ERR, "fork failed: %s", strerror(errno));
        exit(EXIT_FAILURE);
    }

    /* If we got a good PID, then we can exit the parent process. */
    if (pid > 0)
    {
        exit(EXIT_SUCCESS);
    }

    /* At this point we are executing as the child process */

    /* Create a new SID for the child process */
    sid = setsid();

    if (sid < 0)
    {
        syslog(LOG_ERR, "setsid failed: %s", strerror(errno));
        exit(EXIT_FAILURE);
    }

    /* Change the current working directory.  This prevents the current
       directory from being locked; hence not being able to remove it. */
    if ((chdir("/")) < 0)
    {
        syslog(LOG_ERR, "chdir failed:  %s", strerror(errno));
        exit(EXIT_FAILURE);
    }

    /* Redirect standard files to /dev/null */
    {
        CHECK_FILE_ERROR(freopen( "/dev/null", "r", stdin), freopen stdin failed);
        CHECK_FILE_ERROR(freopen( "/dev/null", "w", stdout), freopen stdout failed);
        CHECK_FILE_ERROR(freopen( "/dev/null", "w", stderr), freopen stderr failed);
    }
}

int main(void)
{
    int ret;
    pid_t pid;
    SysfsDataRecord dRecord;
    struct sockaddr_un  addr1;
    socklen_t socklen;
    mode_t oldumask;

    openlog(ident, 0, LOG_USER);

    if (!lock_check())
    {
        syslog(LOG_ERR,"Access to performance counters is locked.\n");
        stop_daemon();
    }

    int err = init_access_dirs(NULL);
    if (err < 0)
    {
        stop_daemon();
    }

    daemonize(&pid);
#ifdef DEBUG_LIKWID
    syslog(LOG_INFO, "SysfsDaemon runs with UID %d, eUID %d\n", getuid(), geteuid());
#endif


    /* setup filename for socket */
    filepath = (char*) calloc(sizeof(addr1.sun_path), 1);
    snprintf(filepath, sizeof(addr1.sun_path), TOSTRING(LIKWIDSOCKETBASE) "-sysfs-%d", pid);

    /* get a socket */
    LOG_AND_EXIT_IF_ERROR(sockfd = socket(AF_LOCAL, SOCK_STREAM, 0), socket failed);

    /* initialize socket data structure */
    bzero(&addr1, sizeof(addr1));
    addr1.sun_family = AF_LOCAL;
    strncpy(addr1.sun_path, filepath, (sizeof(addr1.sun_path) - 1)); /* null terminated by the bzero() above! */

    /* Change the file mode mask so only the calling user has access
     * and switch the user/gid with which the following socket creation runs. */
    oldumask = umask(077);
    CHECK_ERROR(setfsuid(getuid()), setfsuid failed);

    /* bind and listen on socket */
    LOG_AND_EXIT_IF_ERROR(bind(sockfd, (SA*) &addr1, sizeof(addr1)), bind failed);
    LOG_AND_EXIT_IF_ERROR(listen(sockfd, 1), listen failed);
    LOG_AND_EXIT_IF_ERROR(chmod(filepath, S_IRUSR|S_IWUSR), chmod failed);

    socklen = sizeof(addr1);

    { /* Init signal handler */
        struct sigaction sia;
        sia.sa_handler = Signal_Handler;
        sigemptyset(&sia.sa_mask);
        sia.sa_flags = 0;
        sigaction(SIGALRM, &sia, NULL);
        sigaction(SIGPIPE, &sia, NULL);
        sigaction(SIGTERM, &sia, NULL);
    }

    /* setup an alarm to stop the daemon if there is no connect.*/
    alarm(15U);

    if ((connfd = accept(sockfd, (SA*) &addr1, &socklen)) < 0)
    {
        if (errno == EINTR)
        {
            syslog(LOG_ERR, "exiting due to timeout - no client connected after 15 seconds.");
        }
        else
        {
            syslog(LOG_ERR, "accept() failed:  %s", strerror(errno));
        }
        CHECK_ERROR(unlink(filepath), unlink of socket failed);
        exit(EXIT_FAILURE);
    }

    alarm(0);
    CHECK_ERROR(unlink(filepath), unlink of socket failed);

    /* Restore the old umask and fs ids. */
    (void) umask(oldumask);
    CHECK_ERROR(setfsuid(geteuid()), setfsuid failed);


LOOP:
    //syslog(LOG_ERR, "Starting loop %d\n", avail_cpus);
    while (1)
    {
        ret = read(connfd, (void*) &dRecord, sizeof(SysfsDataRecord));

        if (ret < 0)
        {
            syslog(LOG_ERR, "ERROR - Read returns %d", ret);
            stop_daemon();
        }
        else if ((ret == 0) && (dRecord.type != SYSFS_EXIT))
        {
            syslog(LOG_ERR, "ERROR - [%s:%d] zero read, remote socket closed before reading", __FILE__, __LINE__);
            stop_daemon();
        }

        if (dRecord.type == SYSFS_READ)
        {
            read_sysfs_file(dRecord.filename, dRecord.datalen, (char*)&dRecord.data);
        }
        else if (dRecord.type == SYSFS_WRITE)
        {
            write_sysfs_file(dRecord.filename, dRecord.datalen, dRecord.data);
        }
        else if (dRecord.type == SYSFS_EXIT)
        {
            stop_daemon();
        }
        else
        {
            syslog(LOG_ERR, "unknown daemon command type %d", dRecord.type);
            dRecord.errorcode = SYSFS_ERR_UNKNOWN;
        }

        LOG_AND_EXIT_IF_ERROR(write(connfd, (void*) &dRecord, sizeof(SysfsDataRecord)), write failed);
    }

    /* never reached */
    return EXIT_SUCCESS;
}
