/*
 * =======================================================================================
 *
 *      Filename:  perfmon.c
 *
 *      Description:  Implementation of perfmon Module.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2014 Jan Treibig
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
#include <string.h>
#include <math.h>
#include <float.h>
#include <unistd.h>
#include <sys/types.h>
#include <assert.h>

#include <types.h>
#include <bitUtil.h>
#include <bstrlib.h>
#include <strUtil.h>
#include <bitUtil.h>
#include <error.h>
#include <timer.h>
#include <accessClient.h>
#include <msr.h>
#include <pci.h>
#include <lock.h>
#include <cpuid.h>
#include <affinity.h>
#include <tree.h>
#include <power.h>
#include <thermal.h>
#include <perfmon.h>
#include <asciiTable.h>
#include <registers.h>


/* #####   EXPORTED VARIABLES   ########################################### */

int perfmon_verbose = 0;
int perfmon_csvoutput = 0;

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

static PerfmonGroup groupSet = _NOGROUP;
static PerfmonEvent* eventHash;
static PerfmonCounterMap* counter_map;
static PerfmonGroupMap* group_map;
static PerfmonGroupHelp* group_help;
static EventSetup * eventSetup;

static TimerData timeData;
static double rdtscTime;
static PerfmonEventSet perfmon_set;
static int perfmon_numGroups;
static int perfmon_numCounters;
static int perfmon_numArchEvents;
static int perfmon_numThreads;
static int perfmon_numRegions;
static FILE* OUTSTREAM;
static double** perfmon_threadState;
static PerfmonThread* perfmon_threadData;

static int socket_fd = -1;
static int socket_lock[MAX_NUM_NODES];

/* #####   PROTOTYPES  -  LOCAL TO THIS SOURCE FILE   ##################### */

static void initResultTable(PerfmonResultTable* tableData,
        bstrList* firstColumn,
        int numRows,
        int numColumns);

static void initStatisticTable(PerfmonResultTable* tableData,
        bstrList* firstColumn,
        int numRows);

static void printResultTable(PerfmonResultTable* tableData);
static void freeResultTable(PerfmonResultTable* tableData);
static void initThread(int , int );

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */

#define CHECKERROR \
        if (ret == EOF) \
        { \
            fprintf (stderr, "sscanf: Failed to read marker file!\n" ); \
            exit (EXIT_FAILURE);}

#define bstrListAdd(bl,id,name) \
    label = bfromcstr(#name);  \
    (bl)->entry[id] = bstrcpy(label);  \
    (bl)->qty++; \
    bdestroy(label);

#define INIT_EVENTS   \
    fc = bstrListCreate(); \
    bstrListAlloc(fc, numRows+1); \
    bstrListAdd(fc,0,Event); \
    for (i=0; i<numRows; i++) \
    { \
        fc->entry[1+i] = \
           bfromcstr(perfmon_set.events[i].event.name); } 

#define INIT_BASIC  \
    fc = bstrListCreate(); \
    bstrListAlloc(fc, numRows+1); \
    bstrListAdd(fc,0,Metric);

#include <perfmon_pm.h>
#include <perfmon_atom.h>
#include <perfmon_core2.h>
#include <perfmon_nehalem.h>
#include <perfmon_westmere.h>
#include <perfmon_westmereEX.h>
#include <perfmon_nehalemEX.h>
#include <perfmon_sandybridge.h>
#include <perfmon_ivybridge.h>
#include <perfmon_haswell.h>
#include <perfmon_phi.h>
#include <perfmon_k8.h>
#include <perfmon_k10.h>
#include <perfmon_interlagos.h>
#include <perfmon_kabini.h>
#include <perfmon_silvermont.h>

/* #####  EXPORTED  FUNCTION POINTERS   ################################### */
void (*perfmon_startCountersThread) (int thread_id);
void (*perfmon_stopCountersThread) (int thread_id);
void (*perfmon_readCountersThread) (int thread_id);
void (*perfmon_setupCounterThread) (int thread_id,
        PerfmonEvent* event, PerfmonCounterIndex index);
void (*printDerivedMetrics) (PerfmonGroup group);
void (*logDerivedMetrics) (PerfmonGroup group, double time, double timeStamp);
void (*perfmon_getDerivedCounterValuesArch)(PerfmonGroup group, float * values, float * out_max, float * out_min);


/* #####   FUNCTION POINTERS  -  LOCAL TO THIS SOURCE FILE ################ */

static void (*initThreadArch) (PerfmonThread *thread);

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

static int getIndex (bstring reg, PerfmonCounterIndex* index)
{
    int ret = FALSE;
    int err = 0;
    uint64_t tmp;
    for (int i=0; i< perfmon_numCounters; i++)
    {
        if (biseqcstr(reg, counter_map[i].key))
        {
            *index = counter_map[i].index;
            ret = TRUE;
        }
    }
    if ((ret) && (counter_map[*index].type != THERMAL) && (counter_map[*index].type != POWER))
    {
        if (counter_map[*index].device == 0)
        {
            tmp = msr_read(0, counter_map[*index].configRegister);
            msr_write(0, counter_map[*index].configRegister,0x0ULL);
        }
        else
        {
            tmp = pci_read(0, counter_map[*index].device, counter_map[*index].configRegister);
            pci_write(0, counter_map[*index].device, counter_map[*index].configRegister, 0x0U);
        }
    }
    else if ((ret) && (counter_map[*index].type == POWER))
    {
        tmp = msr_read(0, counter_map[*index].counterRegister);
    }

    return ret;
}


static int
getEvent(bstring event_str, PerfmonEvent* event)
{
    for (int i=0; i< perfmon_numArchEvents; i++)
    {
        if (biseqcstr(event_str, eventHash[i].name))
        {
            *event = eventHash[i];

            if (perfmon_verbose)
            {
                fprintf(OUTSTREAM,"Found event %s : \
                    Event_id 0x%02X Umask 0x%02X CfgBits 0x%02X Cmask 0x%02X \n",
                        bdata( event_str),
                        event->eventId,
                        event->umask,
                        event->cfgBits,
                        event->cmask);
            }
            return TRUE;
        }
    }

    return FALSE;
}

static void
initThread(int thread_id, int cpu_id)
{
    for (int i=0; i<NUM_PMC; i++)
    {
        perfmon_threadData[thread_id].counters[i].init = FALSE;
    }

    perfmon_threadData[thread_id].processorId = cpu_id;
    initThreadArch(&perfmon_threadData[thread_id]);
}

struct cbsScan{
	/* Parse state */
	bstring src;
	int line;
    LikwidResults* results;
};

static int lineCb (void* parm, int ofs, int len)
{
    int ret;
    struct cbsScan* st = (struct cbsScan*) parm;
    struct bstrList* strList;
    bstring line;

    if (!len) return 1;
    strList = bstrListCreate();

    line = blk2bstr (st->src->data + ofs, len);

    if (st->line < perfmon_numRegions)
    {
        int id;
        strList = bsplit(line,':');

        if( strList->qty < 2 )
        {
            ERROR_PLAIN_PRINT(Failed to read marker file);
        }
        ret = sscanf (bdata(strList->entry[0]), "%d", &id); CHECKERROR;
        st->results[id].tag = bstrcpy(line);
	 bdelete(st->results[id].tag, 0, blength(strList->entry[0])+1);
    }
    else
    {
        int tagId;
        int threadId;

        strList = bsplit(line,32);

        if( strList->qty < (3+NUM_PMC))
        {
            ERROR_PLAIN_PRINT(Failed to read marker file);
        }

        ret = sscanf(bdata(strList->entry[0]), "%d", &tagId); CHECKERROR;
        ret = sscanf(bdata(strList->entry[1]), "%d", &threadId); CHECKERROR;
        ret = sscanf(bdata(strList->entry[2]), "%u", &st->results[tagId].count[threadId]); CHECKERROR;
        ret = sscanf(bdata(strList->entry[3]), "%lf", &st->results[tagId].time[threadId]); CHECKERROR;

        for (int i=0;i<NUM_PMC; i++)
        {
            ret = sscanf(bdata(strList->entry[4+i]), "%lf", &st->results[tagId].counters[threadId][i]); CHECKERROR;
        }
    }

    bstrListDestroy(strList);
    st->line++;
    bdestroy(line);
    return 1;
}

static void
readMarkerFile(bstring filename, LikwidResults** resultsRef)
{
    int numberOfThreads=0;
    int ret;
    int i,j,k;
    struct cbsScan sl;
    FILE * fp;
    LikwidResults* results = *resultsRef;

    if (NULL != (fp = fopen (bdata(filename), "r")))
    {
        bstring src = bread ((bNread) fread, fp);

        /* read header info */
        ret = sscanf (bdata(src), "%d %d", &numberOfThreads, &perfmon_numRegions); CHECKERROR;
        results = (LikwidResults*) malloc(perfmon_numRegions * sizeof(LikwidResults));

        if (perfmon_numRegions == 0)
        {
            fprintf(OUTSTREAM,"ERROR: No region results are listed in marker file\n");
            ERROR_PLAIN_PRINT(No region results in marker file);
        }
        else if (numberOfThreads != perfmon_numThreads)
        {
            fprintf(OUTSTREAM,"ERROR: Is the number of threads for likwid-perfctr equal to the number in the measured application?\n");
            fprintf(OUTSTREAM,"likwid_markerInit and likwid_markerClose must be called in serial region.\n");

            ERROR_PRINT(Number of threads %d in marker file unequal to number of threads in likwid-perfCtr %d,numberOfThreads,perfmon_numThreads);
        }

        /* allocate  LikwidResults struct */
        for (i=0;i<perfmon_numRegions; i++)
        {
            results[i].time = (double*) malloc(numberOfThreads * sizeof(double));
            results[i].count = (uint32_t*) malloc(numberOfThreads * sizeof(uint32_t));
            results[i].counters = (double**) malloc(numberOfThreads * sizeof(double*));

            for (j=0;j<numberOfThreads; j++)
            {
                results[i].time[j] = 0.0;
                results[i].counters[j] = (double*) malloc(NUM_PMC * sizeof(double));

                for (k=0;k<NUM_PMC; k++)
                {
                        results[i].counters[j][k] = 0.0;
                }
            }
        }

        sl.src = src;
        sl.line = 0;
        sl.results = results;
        bsplitcb (src, (char) '\n', bstrchr(src,10)+1, lineCb, &sl);

        fclose (fp);
        bdestroy (src);
    }
    else
    {
        fprintf(OUTSTREAM,"ERROR: The marker result file could not be found!\n");
        fprintf(OUTSTREAM,"Did you call likwid_markerClose() at the end of your measurement?\n");
        ERROR;
    }

    *resultsRef = results;
    bstring exeString = bformat("rm  -f %s",bdata(filename));
    ret = system(bdata(exeString));

    if (ret == EOF)
    {
        ERROR;
    }

    bdestroy(exeString);
}

static void
printResultTable(PerfmonResultTable * tableData)
{
    if (perfmon_csvoutput) 
    {
        int r, c;
        for (c = 0; c < tableData->header->qty; c++) 
        {
            fprintf(OUTSTREAM, "%s%s", ((c == 0) ? "\n" : ","), tableData->header->entry[c]->data);
        }
        fprintf(OUTSTREAM, "%s", "\n");

        for (r = 0; r < tableData->numRows; r++) 
        {
            fprintf(OUTSTREAM, "%s", tableData->rows[r].label->data);

            for (c = 0; c < tableData->numColumns; c++) 
            {
                if (!isnan(tableData->rows[r].value[c])) 
                {
                    fprintf(OUTSTREAM, ",%lf", tableData->rows[r].value[c]);
                }
                else
                {
                    fprintf(OUTSTREAM, ",%s", "nan");
                }
            }
            fprintf(OUTSTREAM, "%s", "\n");
        }
        fprintf(OUTSTREAM, "%s", "\n");
    }
    else
    {
        int i,j;
        TableContainer* table;
        bstrList* labelStrings = NULL;
        bstring label = bfromcstr("NO");

        table = asciiTable_allocate(tableData->numRows,
                tableData->numColumns+1,
                tableData->header);
        asciiTable_setOutput(OUTSTREAM);

        labelStrings = bstrListCreate();
        bstrListAlloc(labelStrings, tableData->numColumns+1);

        for (i=0; i<tableData->numRows; i++)
        {
            labelStrings->qty = 0;
            labelStrings->entry[0] = bstrcpy(tableData->rows[i].label);
            labelStrings->qty++;

            for (j=0; j<(tableData->numColumns);j++)
            {
                label = bformat("%g", tableData->rows[i].value[j]);
                labelStrings->entry[1+j] = bstrcpy(label);
                labelStrings->qty++;
            }
            asciiTable_appendRow(table,labelStrings);
        }

        asciiTable_print(table);
        bdestroy(label);
        bstrListDestroy(labelStrings);
        asciiTable_free(table);
    }
}

static int
getGroupId(bstring groupStr,PerfmonGroup* group)
{
    *group = _NOGROUP;

    for (int i=0; i<perfmon_numGroups; i++)
    {
        if (biseqcstr(groupStr,group_map[i].key)) 
        {
            *group = group_map[i].index;
            return i;
        }
    }

    return -1;
}

static int
checkCounter(bstring counterName, const char* limit)
{
    int i;
    struct bstrList* tokens;
    int value = FALSE;
    bstring limitString = bfromcstr(limit);

    tokens = bstrListCreate();
    tokens = bsplit(limitString,'|');

    for(i=0; i<tokens->qty; i++)
    {
        if(bstrncmp(counterName, tokens->entry[i], blength(tokens->entry[i])))
        {
            value = FALSE;
        }
        else
        {
            value = TRUE;
            break;
        }
    }

    bdestroy(limitString);
    bstrListDestroy(tokens);
    return value;
}

static void
freeResultTable(PerfmonResultTable* tableData)
{
    int i;

    bstrListDestroy(tableData->header);

    for (i=0; i<tableData->numRows; i++)
    {
        free(tableData->rows[i].value);
    }

    free(tableData->rows);
}

static void 
initResultTable(PerfmonResultTable* tableData,
        bstrList* firstColumn,
        int numRows,
        int numColumns)
{
    int i;
    bstrList* header;
    bstring label;

    header = bstrListCreate();
    bstrListAlloc(header, numColumns+1);
    header->entry[0] = bstrcpy(firstColumn->entry[0]); header->qty++;

    for (i=0; i<perfmon_numThreads;i++)
    {
        label = bformat("core %d",perfmon_threadData[i].processorId);
        header->entry[1+i] = bstrcpy(label); header->qty++;
    }

    tableData->numRows = numRows;
    tableData->numColumns = numColumns;
    tableData->header = header;
    tableData->rows = (PerfmonResult*) malloc(numRows*sizeof(PerfmonResult));

    for (i=0; i<numRows; i++)
    {
//        tableData->rows[i].label =
//           bfromcstr(perfmon_set.events[i].event.name);

        tableData->rows[i].label = firstColumn->entry[1+i];

        tableData->rows[i].value =
            (double*) malloc((numColumns)*sizeof(double));
    }
}

static void 
initStatisticTable(PerfmonResultTable* tableData,
        bstrList* firstColumn,
        int numRows)
{
    int i;
    int numColumns = 4;
    bstrList* header;
    bstring label;

    header = bstrListCreate();
    bstrListAlloc(header, numColumns+1);
    header->entry[0] = bstrcpy(firstColumn->entry[0]); header->qty++;

    label = bformat("Sum");
    header->entry[1] = bstrcpy(label); header->qty++;
    label = bformat("Max");
    header->entry[2] = bstrcpy(label); header->qty++;
    label = bformat("Min");
    header->entry[3] = bstrcpy(label); header->qty++;
    label = bformat("Avg");
    header->entry[4] = bstrcpy(label); header->qty++;

    tableData->numRows = numRows;
    tableData->numColumns = numColumns;
    tableData->header = header;
    tableData->rows = (PerfmonResult*) malloc(numRows*sizeof(PerfmonResult));

    for (i=0; i<numRows; i++)
    {
//        tableData->rows[i].label =
//           bfromcstr(perfmon_set.events[i].event.name);

        tableData->rows[i].label = firstColumn->entry[1+i];
        bcatcstr(tableData->rows[i].label," STAT");

        tableData->rows[i].value =
            (double*) malloc((numColumns)*sizeof(double));
    }
}

static void printDerivedMetricsFixed(void)
{
    int threadId;
    double time = rdtscTime;
    double inverseClock = 1.0 /(double) timer_getCpuClock();
    PerfmonResultTable tableData;
    int numRows;
    int numColumns = perfmon_numThreads;
    bstring label;
    bstrList* fc;
    double tmpValue;

    numRows = 4;
    INIT_BASIC;

    bstrListAdd(fc,1,Runtime (RDTSC) [s]);
    bstrListAdd(fc,2,Runtime unhalted [s]);
    bstrListAdd(fc,3,Clock [MHz]);
    bstrListAdd(fc,4,CPI);

    initResultTable(&tableData, fc, numRows, numColumns);

    for(threadId=0; threadId < perfmon_numThreads; threadId++)
    {
        tmpValue = time;
        if (!isnan(tmpValue))
        {
            tableData.rows[0].value[threadId] = tmpValue;
        }
        else
        {
            tableData.rows[0].value[threadId] = 0.0;
        }

        tmpValue = perfmon_getResult(threadId,"FIXC1")*inverseClock;
        if (!isnan(tmpValue))
        {
            tableData.rows[1].value[threadId] = tmpValue;
        }
        else
        {
            tableData.rows[1].value[threadId] = 0.0;
        }

        tmpValue = 1.E-06*(perfmon_getResult(threadId,"FIXC1")/perfmon_getResult(threadId,"FIXC2"))/inverseClock;
        if (!isnan(tmpValue))
        {
            tableData.rows[2].value[threadId] = tmpValue;
        }
        else
        {
            tableData.rows[2].value[threadId] = 0.0;
        }

        tmpValue = perfmon_getResult(threadId,"FIXC1")/perfmon_getResult(threadId,"FIXC0");
        if (!isnan(tmpValue))
        {
            tableData.rows[3].value[threadId] = tmpValue;
        }
        else
        {
            tableData.rows[3].value[threadId] = 0.0;
        }

    }
    printResultTable(&tableData);
    freeResultTable(&tableData);
}

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

void
perfmon_setCSVMode(int v)
{
    perfmon_csvoutput = v;
}

void
perfmon_printCounters(void)
{
    fprintf(OUTSTREAM,"This architecture has %d counters.\n", perfmon_numCounters);
    fprintf(OUTSTREAM,"Counters names:  ");

    for (int i=0; i<perfmon_numCounters; i++)
    {
        fprintf(OUTSTREAM,"%s\t",counter_map[i].key);
    }
    fprintf(OUTSTREAM,".\n");
}

void
perfmon_printEvents(void)
{
    int i;

    fprintf(OUTSTREAM,"This architecture has %d events.\n", perfmon_numArchEvents);
    fprintf(OUTSTREAM,"Event tags (tag, id, umask, counters):\n");

    for (i=0; i<perfmon_numArchEvents; i++)
    {
        fprintf(OUTSTREAM,"%s, 0x%X, 0x%X, %s \n",
                eventHash[i].name,
                eventHash[i].eventId,
                eventHash[i].umask,
                eventHash[i].limit);
    }
}


double
perfmon_getResult(int threadId, char* counterString)
{
    bstring counter = bfromcstr(counterString);
    PerfmonCounterIndex  index;

   if (getIndex(counter,&index))
   {
           return perfmon_threadData[threadId].counters[index].counterData;
   }

   fprintf (stderr, "perfmon_getResult: Failed to get counter Index!\n" );
   return 0.0;
}


void
perfmon_initEventSet(StrUtilEventSet* eventSetConfig, PerfmonEventSet* set)
{
    set->numberOfEvents = eventSetConfig->numberOfEvents;
    set->events = (PerfmonEventSetEntry*)
        malloc(set->numberOfEvents * sizeof(PerfmonEventSetEntry));

    for (int i=0; i<set->numberOfEvents; i++)
    {
        /* get register index */
        if (!getIndex(eventSetConfig->events[i].counterName,
                    &set->events[i].index))
        {
            ERROR_PRINT(Counter register %s not supported,bdata(
                  eventSetConfig->events[i].counterName));
        }

        /* setup event */
        if (!getEvent(eventSetConfig->events[i].eventName,
                    &set->events[i].event))
        {
            ERROR_PRINT(Event %s not found for current architecture,
                bdata(eventSetConfig->events[i].eventName));
        }

        /* is counter allowed for event */
        if (!checkCounter(eventSetConfig->events[i].counterName,
                    set->events[i].event.limit))
        {
            ERROR_PRINT(Register not allowed  for event  %s,
                bdata(eventSetConfig->events[i].eventName));
        }
    }
}

void
perfmon_printMarkerResults(bstring filepath)
{
    int i;
    int j;
    int region;
    LikwidResults* results = NULL;
    PerfmonResultTable tableData;
    PerfmonResultTable regionData;
    int numRows = perfmon_set.numberOfEvents;
    int numColumns = perfmon_numThreads;
    bstrList* fc;
    bstrList* regionLabels;
    bstring label;
    INIT_EVENTS;

    readMarkerFile(filepath, &results);
    initResultTable(&tableData, fc, numRows, numColumns);
    regionLabels = bstrListCreate();
    bstrListAlloc(regionLabels, 3);
    bstrListAdd(regionLabels, 0, Region Info);
    bstrListAdd(regionLabels, 1, RDTSC Runtime [s]);
    bstrListAdd(regionLabels, 2, call count);

    for (region=0; region<perfmon_numRegions; region++)
    {
        initResultTable(&tableData, fc, numRows, numColumns);
        fprintf(OUTSTREAM,"\n=====================\n");
        fprintf(OUTSTREAM,"Region: %s \n", bdata(results[region].tag));
        fprintf(OUTSTREAM,"=====================\n");
        initResultTable(&regionData, regionLabels, 2, numColumns);

        for (j=0; j<numColumns; j++)
        {
            regionData.rows[0].value[j] = results[region].time[j];
            regionData.rows[1].value[j] = (double) results[region].count[j];
        }
        printResultTable(&regionData);

        for (i=0; i<numRows; i++)
        {
            for (j=0; j<numColumns; j++)
            {
                tableData.rows[i].value[j] =
                    results[region].counters[j][perfmon_set.events[i].index];
            }
        }

        printResultTable(&tableData);

        for (j=0; j<numColumns; j++)
        {
            for (i=0; i<numRows; i++)
            {
                perfmon_threadData[j].counters[perfmon_set.events[i].index].counterData =
                    results[region].counters[j][perfmon_set.events[i].index];
            }
        }
        rdtscTime = results[region].time[0];
        if (groupSet != _NOGROUP)
        {
            printDerivedMetrics(groupSet);
        }
        else if ( cpuid_info.family == P6_FAMILY )
        {
            printDerivedMetricsFixed();
        }
    }

    for (i=0;i<perfmon_numRegions; i++)
    {
        for (j=0;j<perfmon_numThreads; j++)
        {
            free(results[i].counters[j]);
        }

        free(results[i].counters);
        free(results[i].time);
    }

    freeResultTable(&tableData);
    freeResultTable(&regionData);
    bstrListDestroy(fc);
    bstrListDestroy(regionLabels);
}

void 
perfmon_logCounterResults(double time)
{
    int i;
    int j;
    double tmp;
    static double timeStamp = 0.0;

    timeStamp += time;

    for (i=0; i<perfmon_set.numberOfEvents; i++)
    {
        fprintf(OUTSTREAM, "%s %e ", perfmon_set.events[i].event.name, timeStamp);
        for (j=0; j<perfmon_numThreads; j++)
        {
            fprintf(OUTSTREAM, "%e ",
                    (double) (perfmon_threadData[j].counters[perfmon_set.events[i].index].counterData) - perfmon_threadState[j][i]);
            tmp =perfmon_threadData[j].counters[perfmon_set.events[i].index].counterData;
            perfmon_threadData[j].counters[perfmon_set.events[i].index].counterData -=
              perfmon_threadState[j][i];
            perfmon_threadState[j][i] = tmp;
        }
        fprintf(OUTSTREAM,"\n");
    }

    if (groupSet != _NOGROUP)
    {
        logDerivedMetrics(groupSet, time, timeStamp);
    }

    fflush(OUTSTREAM);
}

void 
perfmon_printCounterResults()
{
    int i;
    int j;
    PerfmonResultTable tableData;
    int numRows = perfmon_set.numberOfEvents;
    int numColumns = perfmon_numThreads;
    double stat[perfmon_set.numberOfEvents][4]; /* 0:sum, 1:max, 2:min, 3:avg */
    bstrList* fc;
    bstring label;
    INIT_EVENTS;

    for (i=0; i<numRows; i++)
    {
        stat[i][0] = 0;
        stat[i][1] = 0;
        stat[i][2] = DBL_MAX;
    }

    initResultTable(&tableData, fc, numRows, numColumns);

    /* print raw event data */
    for (i=0; i<numRows; i++)
    {
        for (j=0; j<numColumns; j++)
        {
            tableData.rows[i].value[j] =
                (double) perfmon_threadData[j].counters[perfmon_set.events[i].index].counterData;
            stat[i][0] +=
                (double) perfmon_threadData[j].counters[perfmon_set.events[i].index].counterData;
            stat[i][1] =  MAX(stat[i][1],
                    (double) perfmon_threadData[j].counters[perfmon_set.events[i].index].counterData);
            stat[i][2] =  MIN(stat[i][2],
                    (double) perfmon_threadData[j].counters[perfmon_set.events[i].index].counterData);
        }
    }
    printResultTable(&tableData);
    freeResultTable(&tableData);


    /* for threaded results print sum, max, min and avg */
    if (perfmon_numThreads > 1)
    {
        initStatisticTable(&tableData, fc, numRows);

        for (i=0; i<numRows; i++)
        {
            stat[i][3] =  stat[i][0]/perfmon_numThreads;

            for (j=0; j<4; j++)
            {
                tableData.rows[i].value[j] = stat[i][j];
            }
        }
        printResultTable(&tableData);
        freeResultTable(&tableData);
    }

    if (groupSet != _NOGROUP)
    {
        /* print derived metrics */
        printDerivedMetrics(groupSet);
    }
    else if ( cpuid_info.family == P6_FAMILY )
    {
        printDerivedMetricsFixed();
    }
}

double
perfmon_getEventResult(int thread, int index)
{
    return (double) perfmon_threadData[thread].counters[perfmon_set.events[index].index].counterData;
}

EventSetup perfmon_prepareEventSetup(char* eventGroupString){
     EventSetup setup;
     bstring eventString = bfromcstr(eventGroupString);

     setup.eventSetConfig = malloc(sizeof(setup.eventSetConfig));
     setup.perfmon_set = malloc(sizeof(setup.perfmon_set));

     int groupId = getGroupId(eventString, & setup.groupSet);
     setup.groupName = strdup(eventGroupString);
     setup.groupIndex = groupId;
     if (setup.groupSet == _NOGROUP)
     {
        /* eventString is a custom eventSet */
        bstr_to_eventset(setup.eventSetConfig, eventString);
     }
     else
     {
        /* eventString is a group */
        eventString = bfromcstr(group_map[groupId].config);
        bstr_to_eventset(setup.eventSetConfig, eventString);
     }

     perfmon_initEventSet(setup.eventSetConfig, setup.perfmon_set);
     bdestroy(eventString);

     setup.eventNames = (const char**) malloc(setup.perfmon_set->numberOfEvents * sizeof(const char*));

     setup.numberOfEvents = setup.perfmon_set->numberOfEvents;
     for (int i=0; i< setup.perfmon_set->numberOfEvents; i++)
     {
        setup.eventNames[i] = setup.perfmon_set->events[i].event.name;
     }

     setup.numberOfDerivedCounters = group_map[groupId].derivedCounters;
     setup.derivedNames = (const char**) malloc(setup.numberOfDerivedCounters * sizeof(const char*));

     for(int i=0; i < group_map[groupId].derivedCounters; i++){
        setup.derivedNames[i] = group_map[groupId].derivedCounterNames[i];
     }

     return setup;
}


void perfmon_setupCountersForEventSet(EventSetup * setup){    
    perfmon_set = *setup->perfmon_set;
    groupSet = setup->groupSet;
    eventSetup = setup;
    perfmon_setupCounters();
}

void perfmon_getEventCounterValues(uint64_t * values, uint64_t * out_max, uint64_t * out_min){
    
    for(int e = 0; e < perfmon_set.numberOfEvents; e++ ){
        uint64_t sum = 0;
        uint64_t min = (uint64_t) -1;
        uint64_t max = 0;

        for(int i = 0; i < perfmon_numThreads; i++){
            uint64_t cur = perfmon_threadData[i].counters[e].counterData;
            sum += cur;
            max = max > cur ? max : cur;
            min = min < cur ? min : cur;
        }
        values[e] = sum / perfmon_numThreads;
        out_min[e] = min;
        out_max[e] = max;
    }
}

void perfmon_getDerivedCounterValues(float * values, float * out_max, float * out_min){
    perfmon_getDerivedCounterValuesArch(eventSetup->groupSet, values, out_max, out_min);
}

int
perfmon_setupEventSetC(char* eventCString, const char*** eventnames)
{
     int i;
     bstring eventString = bfromcstr(eventCString);
     StrUtilEventSet eventSetConfig;
     int groupId;

     groupId = getGroupId(eventString, &groupSet);
     if (groupSet == _NOGROUP)
     {
        /* eventString is a custom eventSet */
        bstr_to_eventset(&eventSetConfig, eventString);
     }
     else
     {
        /* eventString is a group */
        eventString = bfromcstr(group_map[groupId].config);
        bstr_to_eventset(&eventSetConfig, eventString);
     }

     perfmon_initEventSet(&eventSetConfig, &perfmon_set);
     perfmon_setupCounters();
     bdestroy(eventString);

     (*eventnames) = (const char**) malloc(perfmon_set.numberOfEvents * sizeof(const char*));

     for (i=0; i<perfmon_set.numberOfEvents; i++)
     {
         (*eventnames)[i] = perfmon_set.events[i].event.name;
     }

     return perfmon_set.numberOfEvents;
}

void
perfmon_setupEventSet(bstring eventString, BitMask* counterMask)
{
    int groupId;
    int eventBool = FALSE;
    StrUtilEventSet eventSetConfig;
    PerfmonEvent eventSet;
    struct bstrList* subStr;
    

    groupId = getGroupId(eventString, &groupSet);
    
    if (groupSet == _NOGROUP)
    {
        subStr = bstrListCreate();
        subStr = bsplit(eventString,':');
        eventBool = getEvent(subStr->entry[0], &eventSet);
        bstrListDestroy(subStr);
    }
    
    if (groupSet == _NOGROUP && eventBool != FALSE)
    {
        /* eventString is a custom eventSet */
        /* append fixed counters for Intel processors */
        if ( cpuid_info.family == P6_FAMILY )
        {
            if (cpuid_info.perf_num_fixed_ctr > 0)
            {
                bcatcstr(eventString,",INSTR_RETIRED_ANY:FIXC0");
            }
            if (cpuid_info.perf_num_fixed_ctr > 1)
            {
                bcatcstr(eventString,",CPU_CLK_UNHALTED_CORE:FIXC1");
            }
            if (cpuid_info.perf_num_fixed_ctr > 2)
            {
                bcatcstr(eventString,",CPU_CLK_UNHALTED_REF:FIXC2");
            }
        }
        bstr_to_eventset(&eventSetConfig, eventString);
    }
    else if (groupId < 0 && eventBool == FALSE)
    {
        ERROR_PLAIN_PRINT(Unsupported group or event for this architecture!);
        exit(EXIT_FAILURE);
    }
    else
    {
        if ( group_map[groupId].isUncore )
        {
            if ( (cpuid_info.model != SANDYBRIDGE_EP) &&
                    (cpuid_info.model != IVYBRIDGE_EP))
            {
                ERROR_PLAIN_PRINT(Uncore not supported on Desktop processors!);
                exit(EXIT_FAILURE);
            }
        }

        fprintf(OUTSTREAM,"Measuring group %s\n", group_map[groupId].key);
        /* eventString is a group */
        eventString = bfromcstr(group_map[groupId].config);
        bstr_to_eventset(&eventSetConfig, eventString);
    }

    perfmon_initEventSet(&eventSetConfig, &perfmon_set);
    perfmon_setupCounters();

    if ( counterMask != NULL )
    {
        bitMask_init((*counterMask));
        /* Extract counter mask from first thread */
        for (int index=0; index<perfmon_numCounters; index++)
        {
            if ( perfmon_threadData[0].counters[index].init == TRUE )
            {
                bitMask_set((*counterMask),index);
            }
        }
    }
}


void
perfmon_setupCounters()
{
    for (int j=0; j<perfmon_set.numberOfEvents; j++)
    {
        for (int i=0; i<perfmon_numThreads; i++)
        {
            perfmon_setupCounterThread(i,
                    &perfmon_set.events[j].event,
                    perfmon_set.events[j].index);
        }
    }
}

void
perfmon_startCounters(void)
{
    for (int i=0;i<perfmon_numThreads;i++)
    {
        perfmon_startCountersThread(i);
    }

    timer_start(&timeData);
}

void
perfmon_stopCounters(void)
{
    int i;

    timer_stop(&timeData);

    for (i=0;i<perfmon_numThreads;i++)
    {
        perfmon_stopCountersThread(i);
    }

    rdtscTime = timer_print(&timeData);
}

void
perfmon_readCounters(void)
{
    int i;

    for (i=0;i<perfmon_numThreads;i++)
    {
        perfmon_readCountersThread(i);
    }
}


void
perfmon_printAvailableGroups()
{
    int i;

    fprintf(OUTSTREAM,"Available groups on %s:\n",cpuid_info.name);

    for(i=0; i<perfmon_numGroups; i++)
    {
        if ( group_map[i].isUncore )
        {
            if ( (cpuid_info.model == SANDYBRIDGE_EP) ||
                    (cpuid_info.model == IVYBRIDGE_EP))
            {
                fprintf(OUTSTREAM,"%s: %s\n",group_map[i].key,
                        group_map[i].info);
            }
        }
        else
        {
            fprintf(OUTSTREAM,"%s: %s\n",group_map[i].key,
                    group_map[i].info);
        }
    }
}

void
perfmon_printGroupHelp(bstring group)
{
    int i;
    PerfmonGroup groupDummy;

    if ((i = getGroupId(group,&groupDummy))<0)
    {
        ERROR_PLAIN_PRINT(Group not found);
    }
    else
    {
        fprintf(OUTSTREAM,"Group %s:\n",bdata(group));
        fprintf(OUTSTREAM,"%s",group_help[i].msg);
    }
}



void
perfmon_init(int numThreads_local, int threads[], FILE* outstream)
{
    if (!lock_check())
    {
        fprintf(stderr,"Access to performance counters is locked.\n");
        exit(EXIT_FAILURE);
    }

    perfmon_numThreads = numThreads_local;
    perfmon_threadData = (PerfmonThread*)
        malloc(perfmon_numThreads * sizeof(PerfmonThread));
    /* This is specific for daemon mode. */
    perfmon_threadState = (double**)
        malloc(perfmon_numThreads * sizeof(double*));

    for (int i=0; i<perfmon_numThreads; i++)
    {
        perfmon_threadState[i] = (double*)
            malloc(NUM_PMC * sizeof(double));
    }

    OUTSTREAM = outstream;

    for(int i=0; i<MAX_NUM_NODES; i++) socket_lock[i] = LOCK_INIT;

    if (accessClient_mode != DAEMON_AM_DIRECT)
    {
        accessClient_init(&socket_fd);
    }

    msr_init(socket_fd);

    switch ( cpuid_info.family )
    {
        case P6_FAMILY:

            switch ( cpuid_info.model )
            {
                case PENTIUM_M_BANIAS:

                case PENTIUM_M_DOTHAN:

                    eventHash = pm_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEvents_pm;

                    group_map = pm_group_map;
                 //   group_help = pm_group_help;
                    perfmon_numGroups = perfmon_numGroups_pm;

                    counter_map = pm_counter_map;
                    perfmon_numCounters = perfmon_numCounters_pm;

                    initThreadArch = perfmon_init_pm;
                    printDerivedMetrics = perfmon_printDerivedMetrics_pm;
                    assert(FALSE && "NOT SUPPORTED");
                    perfmon_startCountersThread = perfmon_startCountersThread_pm;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_pm;
                    perfmon_setupCounterThread = perfmon_setupCounterThread_pm;
                    break;

                case ATOM_45:

                case ATOM_32:

                case ATOM_22:

                case ATOM:

                    eventHash = atom_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsAtom;

                    group_map = atom_group_map;
                    group_help = atom_group_help;
                    perfmon_numGroups = perfmon_numGroupsAtom;

                    counter_map = core2_counter_map;
                    perfmon_numCounters = perfmon_numCountersCore2;

                    initThreadArch = perfmon_init_core2;
                    printDerivedMetrics = perfmon_printDerivedMetricsAtom;
                    perfmon_getDerivedCounterValuesArch = perfmon_getDerivedCounterValuesAtom;
                    perfmon_startCountersThread = perfmon_startCountersThread_core2;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_core2;
                    perfmon_setupCounterThread = perfmon_setupCounterThread_core2;
                    break;
                    
                case ATOM_SILVERMONT:
                    power_init(0);
                    thermal_init(0);
                    eventHash = silvermont_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsSilvermont;

                    group_map = silvermont_group_map;
                    group_help = silvermont_group_help;
                    perfmon_numGroups = perfmon_numGroupsSilvermont;

                    counter_map = silvermont_counter_map;
                    perfmon_numCounters = perfmon_numCountersSilvermont;

                    initThreadArch = perfmon_init_silvermont;
                    printDerivedMetrics = perfmon_printDerivedMetricsSilvermont;
                    perfmon_startCountersThread = perfmon_startCountersThread_silvermont;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_silvermont;
                    perfmon_setupCounterThread = perfmon_setupCounterThread_silvermont;
                    break;


                case CORE_DUO:
                    ERROR_PLAIN_PRINT(Unsupported Processor);
                    break;

                case XEON_MP:

                case CORE2_65:

                case CORE2_45:

                    eventHash = core2_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsCore2;

                    group_map = core2_group_map;
                    group_help = core2_group_help;
                    perfmon_numGroups = perfmon_numGroupsCore2;

                    counter_map = core2_counter_map;
                    perfmon_numCounters = perfmon_numCountersCore2;

                    initThreadArch = perfmon_init_core2;
                    printDerivedMetrics = perfmon_printDerivedMetricsCore2;
                    perfmon_getDerivedCounterValuesArch = perfmon_getDerivedCounterValuesCore2;

                    logDerivedMetrics = perfmon_logDerivedMetricsCore2;
                    perfmon_startCountersThread = perfmon_startCountersThread_core2;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_core2;
                    perfmon_readCountersThread = perfmon_readCountersThread_core2;
                    perfmon_setupCounterThread = perfmon_setupCounterThread_core2;
                    break;

                case NEHALEM_EX:

                    eventHash = nehalemEX_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsNehalemEX;

                    group_map = nehalemEX_group_map;
                    group_help = nehalemEX_group_help;
                    perfmon_numGroups = perfmon_numGroupsNehalemEX;

                    counter_map = westmereEX_counter_map;
                    perfmon_numCounters = perfmon_numCountersWestmereEX;

                    initThreadArch = perfmon_init_nehalemEX;
                    printDerivedMetrics = perfmon_printDerivedMetricsNehalemEX;
                    perfmon_getDerivedCounterValuesArch = perfmon_getDerivedCounterValuesNehalemEX;
                    logDerivedMetrics = perfmon_logDerivedMetricsNehalemEX;
                    perfmon_startCountersThread = perfmon_startCountersThread_nehalemEX;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_nehalemEX;
                    perfmon_readCountersThread = perfmon_readCountersThread_nehalemEX;
                    perfmon_setupCounterThread = perfmon_setupCounterThread_nehalemEX;
                    break;

                case WESTMERE_EX:

                    eventHash = westmereEX_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsWestmereEX;

                    group_map = westmereEX_group_map;
                    group_help = westmereEX_group_help;
                    perfmon_numGroups = perfmon_numGroupsWestmereEX;

                    counter_map = westmereEX_counter_map;
                    perfmon_numCounters = perfmon_numCountersWestmereEX;

                    initThreadArch = perfmon_init_westmereEX;                    
                    printDerivedMetrics = perfmon_printDerivedMetricsWestmereEX;
                    perfmon_getDerivedCounterValuesArch = perfmon_getDerivedCounterValuesWestmereEX;
                    logDerivedMetrics = perfmon_logDerivedMetricsWestmereEX;
                    perfmon_startCountersThread = perfmon_startCountersThread_westmereEX;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_westmereEX;
                    perfmon_readCountersThread = perfmon_readCountersThread_westmereEX;
                    perfmon_setupCounterThread = perfmon_setupCounterThread_westmereEX;
                    break;

                case NEHALEM_BLOOMFIELD:

                case NEHALEM_LYNNFIELD:

                    thermal_init(0);

                    eventHash = nehalem_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsNehalem;

                    group_map = nehalem_group_map;
                    group_help = nehalem_group_help;
                    perfmon_numGroups = perfmon_numGroupsNehalem;

                    counter_map = nehalem_counter_map;
                    perfmon_numCounters = perfmon_numCountersNehalem;

                    initThreadArch = perfmon_init_nehalem;
                    printDerivedMetrics = perfmon_printDerivedMetricsNehalem;
                    perfmon_getDerivedCounterValuesArch = perfmon_getDerivedCounterValuesNehalem;

                    logDerivedMetrics = perfmon_logDerivedMetricsNehalem;
                    perfmon_startCountersThread = perfmon_startCountersThread_nehalem;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_nehalem;
                    perfmon_readCountersThread = perfmon_readCountersThread_nehalem;
                    perfmon_setupCounterThread = perfmon_setupCounterThread_nehalem;
                    break;

                case NEHALEM_WESTMERE_M:

                case NEHALEM_WESTMERE:

                    thermal_init(0);

                    eventHash = westmere_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsWestmere;

                    group_map = westmere_group_map;
                    group_help = westmere_group_help;
                    perfmon_numGroups = perfmon_numGroupsWestmere;

                    counter_map = nehalem_counter_map;
                    perfmon_numCounters = perfmon_numCountersNehalem;

                    initThreadArch = perfmon_init_nehalem;
                    printDerivedMetrics = perfmon_printDerivedMetricsWestmere;
                    perfmon_getDerivedCounterValuesArch = perfmon_getDerivedCounterValuesWestmere;

                    logDerivedMetrics = perfmon_logDerivedMetricsWestmere;
                    perfmon_startCountersThread = perfmon_startCountersThread_nehalem;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_nehalem;
                    perfmon_readCountersThread = perfmon_readCountersThread_nehalem;
                    perfmon_setupCounterThread = perfmon_setupCounterThread_nehalem;
                    break;

                case IVYBRIDGE:

                case IVYBRIDGE_EP:

                    power_init(0); /* FIXME Static coreId is dangerous */
                    thermal_init(0);
                    pci_init(socket_fd); 

                    eventHash = ivybridge_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsIvybridge;

                    group_map = ivybridge_group_map;
                    group_help = ivybridge_group_help;
                    perfmon_numGroups = perfmon_numGroupsIvybridge;

                    counter_map = ivybridge_counter_map;
                    perfmon_numCounters = perfmon_numCountersIvybridge;

                    initThreadArch = perfmon_init_ivybridge;
                    printDerivedMetrics = perfmon_printDerivedMetricsIvybridge;
                    perfmon_getDerivedCounterValuesArch = perfmon_getDerivedCounterValuesIvybridge;

                    logDerivedMetrics = perfmon_logDerivedMetricsIvybridge;
                    perfmon_startCountersThread = perfmon_startCountersThread_ivybridge;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_ivybridge;
                    perfmon_readCountersThread = perfmon_readCountersThread_ivybridge;
                    perfmon_setupCounterThread = perfmon_setupCounterThread_ivybridge;
                    break;

                case HASWELL:

                case HASWELL_EX:

                case HASWELL_M1:

                case HASWELL_M2:

                    power_init(0); /* FIXME Static coreId is dangerous */
                    thermal_init(0);

                    eventHash = haswell_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsHaswell;

                    group_map = haswell_group_map;
                    group_help = haswell_group_help;
                    perfmon_numGroups = perfmon_numGroupsHaswell;

                    counter_map = haswell_counter_map;
                    perfmon_numCounters = perfmon_numCountersHaswell;

                    initThreadArch = perfmon_init_haswell;
                    printDerivedMetrics = perfmon_printDerivedMetricsHaswell;
                    perfmon_getDerivedCounterValuesArch = perfmon_getDerivedCounterValuesHaswell;
                    logDerivedMetrics = perfmon_logDerivedMetricsHaswell;
                    perfmon_startCountersThread = perfmon_startCountersThread_haswell;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_haswell;
                    perfmon_readCountersThread = perfmon_readCountersThread_haswell;
                    perfmon_setupCounterThread = perfmon_setupCounterThread_haswell;
                    break;

                case SANDYBRIDGE:

                case SANDYBRIDGE_EP:

                    power_init(0); /* FIXME Static coreId is dangerous */
                    thermal_init(0);
                    pci_init(socket_fd);

                    eventHash = sandybridge_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsSandybridge;

                    group_map = sandybridge_group_map;
                    group_help = sandybridge_group_help;
                    perfmon_numGroups = perfmon_numGroupsSandybridge;

                    counter_map = sandybridge_counter_map;
                    perfmon_numCounters = perfmon_numCountersSandybridge;

                    initThreadArch = perfmon_init_sandybridge;
                    printDerivedMetrics = perfmon_printDerivedMetricsSandybridge;
                    perfmon_getDerivedCounterValuesArch = perfmon_getDerivedCounterValuesSandybridge;
                    logDerivedMetrics = perfmon_logDerivedMetricsSandybridge;
                    perfmon_startCountersThread = perfmon_startCountersThread_sandybridge;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_sandybridge;
                    perfmon_readCountersThread = perfmon_readCountersThread_sandybridge;
                    perfmon_setupCounterThread = perfmon_setupCounterThread_sandybridge;
                    break;

                default:
                    ERROR_PLAIN_PRINT(Unsupported Processor);
                    break;
            }
            break;

        case MIC_FAMILY:

            switch ( cpuid_info.model )
            {
                case XEON_PHI:

                    eventHash = phi_arch_events;
                    perfmon_numArchEvents = perfmon_numArchEventsPhi;

                    group_map = phi_group_map;
                    group_help = phi_group_help;
                    perfmon_numGroups = perfmon_numGroupsPhi;

                    counter_map = phi_counter_map;
                    perfmon_numCounters = perfmon_numCountersPhi;

                    initThreadArch = perfmon_init_phi;
                    printDerivedMetrics = perfmon_printDerivedMetricsPhi;
                    perfmon_getDerivedCounterValuesArch = perfmon_getDerivedCounterValuesPhi;
                    logDerivedMetrics = perfmon_logDerivedMetricsPhi;
                    perfmon_startCountersThread = perfmon_startCountersThread_phi;
                    perfmon_stopCountersThread = perfmon_stopCountersThread_phi;
                    perfmon_readCountersThread = perfmon_readCountersThread_phi;
                    perfmon_setupCounterThread = perfmon_setupCounterThread_phi;
                    break;

                default:
                    ERROR_PLAIN_PRINT(Unsupported Processor);
                    break;
            }
            break;

        case K8_FAMILY:
            eventHash = k8_arch_events;
            perfmon_numArchEvents = perfmon_numArchEventsK8;

            group_map = k8_group_map;
            group_help = k8_group_help;
            perfmon_numGroups = perfmon_numGroupsK8;

            counter_map = k10_counter_map;
            perfmon_numCounters = perfmon_numCountersK10;

            initThreadArch = perfmon_init_k10;
            printDerivedMetrics = perfmon_printDerivedMetricsK8;
            perfmon_getDerivedCounterValuesArch = perfmon_getDerivedCounterValuesK8;
            logDerivedMetrics = perfmon_logDerivedMetricsK8;
            perfmon_startCountersThread = perfmon_startCountersThread_k10;
            perfmon_stopCountersThread = perfmon_stopCountersThread_k10;
            perfmon_readCountersThread = perfmon_readCountersThread_k10;
            perfmon_setupCounterThread = perfmon_setupCounterThread_k10;
            break;

        case K10_FAMILY:
            eventHash = k10_arch_events;
            perfmon_numArchEvents = perfmon_numArchEventsK10;

            group_map = k10_group_map;
            group_help = k10_group_help;
            perfmon_numGroups = perfmon_numGroupsK10;

            counter_map = k10_counter_map;
            perfmon_numCounters = perfmon_numCountersK10;

            initThreadArch = perfmon_init_k10;
            printDerivedMetrics = perfmon_printDerivedMetricsK10;
            perfmon_getDerivedCounterValuesArch = perfmon_getDerivedCounterValuesK10;
            logDerivedMetrics = perfmon_logDerivedMetricsK10;
            perfmon_startCountersThread = perfmon_startCountersThread_k10;
            perfmon_stopCountersThread = perfmon_stopCountersThread_k10;
            perfmon_readCountersThread = perfmon_readCountersThread_k10;
            perfmon_setupCounterThread = perfmon_setupCounterThread_k10;
            break;

        case K15_FAMILY:
            eventHash = interlagos_arch_events;
            perfmon_numArchEvents = perfmon_numArchEventsInterlagos;

            group_map = interlagos_group_map;
            group_help = interlagos_group_help;
            perfmon_numGroups = perfmon_numGroupsInterlagos;

            counter_map = interlagos_counter_map;
            perfmon_numCounters = perfmon_numCountersInterlagos;

            initThreadArch = perfmon_init_interlagos;
            printDerivedMetrics = perfmon_printDerivedMetricsInterlagos;
            perfmon_getDerivedCounterValuesArch = perfmon_getDerivedCounterValuesInterlagos;
            logDerivedMetrics = perfmon_logDerivedMetricsInterlagos;
            perfmon_startCountersThread = perfmon_startCountersThread_interlagos;
            perfmon_stopCountersThread = perfmon_stopCountersThread_interlagos;
            perfmon_readCountersThread = perfmon_readCountersThread_interlagos;
            perfmon_setupCounterThread = perfmon_setupCounterThread_interlagos;
            break;

        case K16_FAMILY:
            eventHash = kabini_arch_events;
            perfmon_numArchEvents = perfmon_numArchEventsKabini;

            group_map = kabini_group_map;
            group_help = kabini_group_help;
            perfmon_numGroups = perfmon_numGroupsKabini;

            counter_map = kabini_counter_map;
            perfmon_numCounters = perfmon_numCountersKabini;

            initThreadArch = perfmon_init_kabini;
            printDerivedMetrics = perfmon_printDerivedMetricsKabini;
            perfmon_getDerivedCounterValuesArch = perfmon_getDerivedCounterValuesKabini;
            logDerivedMetrics = perfmon_logDerivedMetricsKabini;
            perfmon_startCountersThread = perfmon_startCountersThread_kabini;
            perfmon_stopCountersThread = perfmon_stopCountersThread_kabini;
            perfmon_readCountersThread = perfmon_readCountersThread_kabini;
            perfmon_setupCounterThread = perfmon_setupCounterThread_kabini;
           break;

        default:
            ERROR_PLAIN_PRINT(Unsupported Processor);
            break;
    }


    for (int i=0; i<perfmon_numThreads; i++)
    {
        initThread(i,threads[i]);
    }
}

void
perfmon_finalize()
{
    int i;

    free(perfmon_threadData);

    for (i=0; i<perfmon_numThreads; i++)
    {
        free(perfmon_threadState[i]);
    }
    free(perfmon_threadState);
    msr_finalize();
    pci_finalize();
    accessClient_finalize(socket_fd);
}

