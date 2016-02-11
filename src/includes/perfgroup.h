#ifndef PERFGROUP_H
#define PERFGROUP_H


 /*! \brief The groupInfo data structure describes a performance group

Groups can be either be read in from file or be a group with custom event set. For
performance groups commonly all values are set. For groups with custom event set,
the fields groupname and shortinfo are set to 'Custom', longinfo is NULL and in
general the nmetrics value is 0.
*/
typedef struct {
    char* groupname; /*!< \brief Name of the group: performance group name or 'Custom' */
    char* shortinfo; /*!< \brief Short info string for the group or 'Custom' */
    int nevents; /*!< \brief Number of event/counter combinations */
    char** events; /*!< \brief List of events */
    char** counters; /*!< \brief List of counter registers */
    int nmetrics; /*!< \brief Number of metrics */
    char** metricnames; /*!< \brief Metric names */
    char** metricformulas; /*!< \brief Metric formulas */
    char* longinfo; /*!< \brief Descriptive text about the group or empty */
} GroupInfo;

typedef struct {
    int counters; /*!< \brief Number of entries in the list */
    char** cnames; /*!< \brief List of counter names */
    double* cvalues; /*!< \brief List of counter values */
} CounterList;

typedef enum {
    GROUP_NONE = 0,
    GROUP_SHORT,
    GROUP_EVENTSET,
    GROUP_METRICS,
    GROUP_LONG
} GroupFileSections;

static char* groupFileSectionNames[5] = {
    "NONE",
    "SHORT",
    "EVENTSET",
    "METRICS",
    "LONG"
};

extern int get_groups(char* grouppath, char* architecture, char*** groupnames, char*** groupshort, char*** grouplong);
extern int read_group(char* grouppath, char* architecture, char* groupname, GroupInfo* ginfo);
extern int custom_group(char* eventStr, GroupInfo* ginfo);
extern char* get_eventStr(GroupInfo* ginfo);
extern char* get_shortInfo(GroupInfo* ginfo);
extern char* get_longInfo(GroupInfo* ginfo);
extern void return_group(GroupInfo* ginfo);

extern void init_clist(CounterList* clist);
extern int add_to_clist(CounterList* clist, char* counter, double result);
extern void destroy_clist(CounterList* clist);

extern double calc_metric(char* formula, CounterList* clist);

#endif
