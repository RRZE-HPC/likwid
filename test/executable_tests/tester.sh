#!/bin/bash
if [ $# -ne 1 ]; then
    echo "You need to give application to test on commandline"
    exit 1
fi

EXECPATH=/usr/local/bin
EXEC=$1
TMPFILE=/tmp/testout
FREQ="2.3"

f_grep() {
    ARG="$1"
    if [ `grep "${ARG}" ${TMPFILE} | wc -l` == "0" ]; then
        return 1
    fi
    return 0
}

f_ngrep() {
    ARG="$1"
    if [ `grep "${ARG}" ${TMPFILE} | wc -l` != "0" ]; then
        return 1
    fi
    return 0
}

f_listlen() {
    LIST=$(cat ${TMPFILE})
    DELIM=$(echo ${1} | cut -d ' ' -f 1)
    COUNT=$(echo ${1} | cut -d ' ' -f 2)
    CHARS=${LIST//[^${DELIM}]}
    LENGTH=$(expr ${#CHARS} + 1)
    if [ ${LENGTH} != "${COUNT}" ]; then
        return 1
    fi
    return 0
}

if [ ! -e ${EXEC}.txt ]; then
    echo "Cannot find testfile ${EXEC}.txt"
    exit 1
fi
if [ "${EXEC}" == "likwid-setFrequencies" ]; then
    FREQ=$(${EXECPATH}/likwid-setFrequencies -l | grep -v frequencies | awk '{print $2}')
    CURFREQ=$(${EXECPATH}/likwid-setFrequencies -p | head -n2 | tail -n 1 | rev | awk '{print $2}' | rev | awk -F'/' '{print $2}')
fi
if [ "${EXEC}" == "likwid-mpirun" ]; then
    if [ -z "$(which mpiexec)" ] && [ -z "$(which mpiexec.hydra)" ] && [ -z "$(which mpirun)" ]; then
        echo "Cannot find MPI implementation, neither mpiexec, mpiexec.hydra nor mpirun can be found in any directory in PATH"
        exit 1
    fi
fi

while read -r LINE || [[ -n $LINE ]]; do
    if [ -z "${LINE}" ]; then continue; fi
    if [[ "${LINE}" =~ \#.* ]]; then continue; fi
    OPTIONS=$(echo "${LINE}" | cut -d '|' -f 1)
    OPTIONS=${OPTIONS//'FREQ'/"${FREQ}"}
    RESULTS=$(echo "${LINE}" | cut -d '|' -f 2-)
    NUM_RESULTS="${RESULTS//[^|]}"
    EXITCODE=$(${EXECPATH}/${EXEC} ${OPTIONS} 1>${TMPFILE} 2>&1 </dev/null; echo $?)
    STATE=0
    for ((i=1;i<=${#NUM_RESULTS}+1;i++)); do
        RESULT=$(echo ${RESULTS} | cut -d '|' -f ${i})
        RESULT_CMD=$(echo $RESULT | cut -d' ' -f1)
        RESULT_OPTS=$(echo $RESULT | cut -d ' ' -f 2-)
        if [ ${RESULT_CMD} == "EXIT" ]; then
            if [ "${RESULT_OPTS}" != "$EXITCODE" ]; then
                STATE=1
            fi
        elif [ ${RESULT_CMD} == "GREP" ]; then
            f_grep "${RESULT_OPTS}"
            STATE=$?
        elif [ ${RESULT_CMD} == "NGREP" ]; then
            f_ngrep "${RESULT_OPTS}"
            STATE=$?
        elif [ ${RESULT_CMD} == "LISTLEN" ]; then
            f_listlen "${RESULT_OPTS}"
            STATE=$?
        fi
    done
    if [ $STATE -eq 0 ]; then
        echo "SUCCESS : ${EXEC}" "${OPTIONS}"
    else
        echo "FAIL : ${EXEC}" "${OPTIONS}"
    fi
done < ${EXEC}.txt


if [ "${EXEC}" == "likwid-setFrequencies" ]; then
    ${EXECPATH}/${EXEC} -f "${CURFREQ}"
fi

rm -f /tmp/topo.txt /tmp/test /tmp/test.txt /tmp/out.txt /tmp/out

