#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>



#include <configuration.h>

Configuration config = {NULL,NULL,-1,0,0,0};
int init_config = 0;

int init_configuration(void)
{
    FILE* fp;
    char line[512];
    char name[128];
    char value[256];
    char filename[512] = "";
    
    if (access("/usr/local/etc/likwid.cfg", R_OK) != 0)
    {
        if (access("/etc/likwid.cfg", R_OK) != 0)
        {
            if (!access("./likwid.cfg",R_OK))
            {
                sprintf(filename,"./likwid.cfg");
            }
        }
        else
        {
            sprintf(filename,"/etc/likwid.cfg");
        }
    } 
    else
    {
        sprintf(filename,"/usr/local/etc/likwid.cfg");
    }

    if (strlen(filename) == 0)
    {
        return -EFAULT;
    }
    fp = fopen(filename, "r");

    while (fgets(line, 512, fp) != NULL) {
        if (sscanf(line,"%s = %s", name, value) != 2)
        {
            continue;
        }
        if (strcmp(name, "topology_file") == 0)
        {
            config.topologyCfgFileName = (char*)malloc((strlen(value)+1) * sizeof(char));
            strcpy(config.topologyCfgFileName, value);
            config.topologyCfgFileName[strlen(value)] = '\0';
        }
        else if (strcmp(name, "daemon_path") == 0)
        {
            config.daemonPath = (char*)malloc((strlen(value)+1) * sizeof(char));
            strcpy(config.daemonPath, value);
            config.daemonPath[strlen(value)] = '\0';
        }
        else if (strcmp(name, "daemon_mode") == 0)
        {
            if (strcmp(value, "daemon") == 0) 
            {
                config.daemonMode = DAEMON_AM_ACCESS_D;
            }
            else if (strcmp(value, "direct") == 0)
            {
                config.daemonMode = DAEMON_AM_DIRECT;
            }
        }
        else if (strcmp(name, "max_threads") == 0)
        {
            config.maxNumThreads = atoi(value);
        }
        else if (strcmp(name, "max_nodes") == 0)
        {
            config.maxNumNodes = atoi(value);
        }
        else if (strcmp(name, "max_hashtable_size") == 0)
        {
            config.maxHashTableSize = atoi(value);
        }
    }
    
    init_config = 1;
    
    fclose(fp);
    return 0;
}

Configuration_t get_configuration(void)
{
    return &config;
}

int destroy_configuration(void)
{
    if (init_config == 0) 
    {
        return -EFAULT;
    }
    free(config.topologyCfgFileName);
    free(config.daemonPath);
    init_config = 0;
    return 0;
}
