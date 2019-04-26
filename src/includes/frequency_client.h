#ifndef LIKWID_FREQUENCY_CLIENT_H
#define LIKWID_FREQUENCY_CLIENT_H

#define LIKWID_FREQUENCY_MAX_DATA_LENGTH   100

typedef enum {
    FREQ_READ = 0,
    FREQ_WRITE,
    FREQ_EXIT
} FreqDataRecordType;


typedef enum {
    FREQ_LOC_MIN = 0,
    FREQ_LOC_MAX,
    FREQ_LOC_CUR,
    FREQ_LOC_GOV,
    FREQ_LOC_AVAIL_GOV,
    FREQ_LOC_AVAIL_FREQ,
    FREQ_LOC_CONF_MIN,
    FREQ_LOC_CONF_MAX,
}FreqDataRecordLocation;

typedef enum {
    FREQ_ERR_NONE = 0,
    FREQ_ERR_NOFILE,
    FREQ_ERR_NOPERM,
    FREQ_ERR_UNKNOWN
} FreqDataRecordError;

typedef struct {
    uint32_t cpu;
    FreqDataRecordType type;
    FreqDataRecordLocation loc;
    FreqDataRecordError errorcode;
    int datalen;
    char data[LIKWID_FREQUENCY_MAX_DATA_LENGTH];
} FreqDataRecord;

#endif /* LIKWID_FREQUENCY_CLIENT_H */
