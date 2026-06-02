/*
 * =======================================================================================
 *
 *      Filename:  pthread-overload.c
 *
 *      Description:  Overloaded library for pthread_create call. 
 *                    Implements pinning of threads together with likwid-pin.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Michael Panzlaff (mp), michale.panzlaff@fau.de
 *      Project:  likwid
 *
 *      Copyright (C) 2016 RRZE, University Erlangen-Nuremberg
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

#include <assert.h>
#include <errno.h>
#include <link.h>
#include <sched.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <types.h>

#ifdef COLOR
#include <textcolor.h>
#endif

typedef void *(*start_routine_ptr)(void *);

typedef int (*pthread_create_ptr)(
        pthread_t *thread,
        const pthread_attr_t *attr,
        start_routine_ptr start_routine,
        void *arg);

struct pthread_create_wrapper_arg {
    start_routine_ptr start_routine_orig;
    void *arg_orig;
};

static bool silent = false;
static bool pin_ids_printed = false;

static uint32_t *pin_ids;
static size_t pin_ids_count;
static uint64_t pin_skip_mask;

static uintptr_t openmp_text_start;
static uintptr_t openmp_text_end;
static bool openmp_found = false;

static pthread_create_ptr pthread_create_orig;

#ifdef COLOR
#define COLOR_PRINT_FORCE_PREFIX(prefix, msg, ...) \
    fprintf(stderr, "\e[1;%dm" prefix msg "\e[0m", COLOR + 30, ##__VA_ARGS__)
#else
#define COLOR_PRINT_FORCE_PREFIX(prefix, msg, ...) \
    fprintf(stderr, prefix msg, ##__VA_ARGS__)
#endif

#define COLOR_PRINT_FORCE(msg, ...) \
    COLOR_PRINT_FORCE_PREFIX("[pin-%d] ", msg, getpid(), ##__VA_ARGS__)

#define COLOR_PRINT(msg, ...) \
    do { \
        if (!silent) \
            COLOR_PRINT_FORCE(msg, ##__VA_ARGS__); \
    } while (0)

static void pdie(const char *msg)
{
    perror(msg);
    abort();
}

static void die(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);

    vfprintf(stderr, fmt, args);
    abort();

    va_end(args);
}

static char *get_cmdline(void)
{
    FILE *f = fopen("/proc/self/cmdline", "r");
    if (!f)
        return NULL;
    char cmdline[256];
    size_t bytes_read = fread(cmdline, 1, sizeof(cmdline), f);
    if (bytes_read == 0) {
        fclose(f);
        return NULL;
    }

    for (size_t i = 0; i < bytes_read; i++) {
        if (cmdline[i] == '\0' && i + 1 < bytes_read)
            cmdline[i] = ' ';
    }

    return strdup(cmdline);
}

static void init_pin_ids(const char *pin_str_orig)
{
    const long avail_cpus = sysconf(_SC_NPROCESSORS_CONF);

    char *pin_str = strdup(pin_str_orig);
    if (!pin_str)
        pdie("strdup");

    // parse LIKWID_PIN string
    char *saveptr = NULL;
    for (char *token = strtok_r(pin_str, ",", &saveptr); token; token = strtok_r(NULL, ",", &saveptr)) {
        errno = 0;
        const unsigned long pin_id = strtoul(token, NULL, 10);
        if (errno != 0)
            die("Cannot parse CPU ID '%s': %s\n", token, strerror(errno));

        if (pin_id >= (uint32_t)avail_cpus)
            die("CPU ID %lu exceeds maximum number of CPUs (%ld)\n", pin_id, avail_cpus);

        const size_t pin_ids_count_new = pin_ids_count + 1;
        uint32_t *pin_ids_new = realloc(pin_ids, pin_ids_count_new * sizeof(*pin_ids));
        if (!pin_ids_new)
            pdie("realloc");

        pin_ids = pin_ids_new;
        pin_ids[pin_ids_count] = (uint32_t)pin_id;
        pin_ids_count = pin_ids_count_new;
    }

    free(pin_str);

    char *cmdline = get_cmdline();
    if (pin_ids_printed) {
        COLOR_PRINT("Pinning enabled (%s)\n", cmdline ? cmdline : "(unknown)");
    } else {
        COLOR_PRINT_FORCE("Pinning enabled (%s), mapping:\n", cmdline ? cmdline : "(unknown)");
        const size_t COLUMN_COUNT = 8;
        size_t column = 0;
        for (size_t i = 0; i < pin_ids_count; i++) {
            if (column == 0)
                COLOR_PRINT_FORCE("  %4zu -> %4u", i, pin_ids[i]);
            else
                COLOR_PRINT_FORCE_PREFIX("", " | %4zu -> %4u", i, pin_ids[i]);

            if (column + 1 >= COLUMN_COUNT) {
                COLOR_PRINT_FORCE_PREFIX("", "\n");
                column = 0;
            } else {
                column += 1;
            }
        }
        if (column > 0)
            COLOR_PRINT_FORCE_PREFIX("", "\n");
    }
    free(cmdline);

    (void)setenv("LIKWID_PIN_PRINTED", "1", 1);
}

static int find_openmp_callback(struct dl_phdr_info *dl_info, size_t size, void *data)
{
    (void)size;
    (void)data;

    static const char *omp_patterns[] = {
        "libgomp.so.1.0.0",
        "libgomp.so.1",
        "libiomp.so",
        "libiomp5.so",
        "libomp.so",
        "libomp5.so",
        "libomp.so.5",
    };


    /* Check if the current library matches any of our patterns. */
    bool is_openmp = false;
    for (size_t i = 0; i < ARRAY_COUNT(omp_patterns); i++) {
        /* Check if a pattern matches */
        if (strstr(dl_info->dlpi_name, omp_patterns[i])) {
            is_openmp = true;
            break;
        }
    }

    if (!is_openmp)
        return 0;

    /* If we found a matching library previously, raise an error, since we
     * cannot reliably determine, which one is actually the right.
     * An application shouldn't be able to ship with multiple OpenMP runtimes.
     * What could still happen is that our patterns are unreliable
     * and falsely detects a library as OpenMP library. */
    if (openmp_found)
        die("Conflicting OpenMP library detected. Did LIKWID detect one incorrectly?: %s", dl_info->dlpi_name);

    openmp_found = true;

    COLOR_PRINT("Found OpenMP runtime: %s\n", dl_info->dlpi_name);

    /* Okay, we found the OpenMP library. Now let's find the .text section.
     * While we cannot read the section name, we can see if a section is executable.
     * We use this section address range in order to determine whether a function
     * is an OpenMP function or not.
     *
     * It may be possible that a library has more than one executable section.
     * So far we haven't seen this during development. Should it happen in the future,
     * we will have to memorize the address range of multiple sections instead of just one. */

    bool text_found = false;
    for (size_t i = 0; i < dl_info->dlpi_phnum; i++) {
        const ElfW(Phdr) *phdr = &dl_info->dlpi_phdr[i];

        /* If the section isn't loaded or executed, ignore it.
         * Those sections cannot be our code section. */
        if (phdr->p_type != PT_LOAD || !(phdr->p_flags & PF_X))
            continue;

        /* If we've found an executable section previously, raise an error.
         * This shouldn't happen. If this happens for a legit reason (because the library
         * actually has multiple executable sections), then we have to memorize all address ranges. */
        if (text_found)
            die("Multiple OpenMP executable sections found. "
                "Support for multiple executable sections is not yet implemented\n");

        openmp_text_start = dl_info->dlpi_addr + phdr->p_vaddr;
        openmp_text_end = openmp_text_start + phdr->p_memsz;
        text_found = true;
    }

    return 0;
}

static void find_openmp(void)
{
    // Search for all loaded shared libraries and see, if they look like an OpenMP library.
    dl_iterate_phdr(find_openmp_callback, NULL);
}

static void find_pthread(void)
{
    // Clear (possible) previous error
    dlerror();
    pthread_create_orig = dlsym(RTLD_NEXT, "pthread_create");
    const char *err = dlerror();
    if (err)
        fprintf(stderr, "Cannot find pthread_create (Does the program use pthreads?): %s\n", err);
}

void __attribute__((constructor)) init_pthread_overload(void)
{
    pin_ids = NULL;
    pin_ids_count = 0;

    if (getenv("LIKWID_SILENT"))
        silent = true;

    if (getenv("LIKWID_PIN_PRINTED"))
        pin_ids_printed = true;

    const char *pin_str_orig = getenv("LIKWID_PIN");
    if (pin_str_orig)
        init_pin_ids(pin_str_orig);
    else
        COLOR_PRINT("LIKWID_PIN environment variable not set. Disabling pinning.\n");

    const char *skip_str = getenv("LIKWID_SKIP");
    if (skip_str)
        pin_skip_mask = strtoull(skip_str, NULL, 16);

    if (pin_skip_mask != 0)
        COLOR_PRINT("PIN SKIP MASK: %#" PRIx64 "\n", pin_skip_mask);

    find_openmp();
    find_pthread();

    /* If OMP_PLACES is not set, raise a warning. This should only happen, if there is
     * a bug in likwid-perfctr or likwid-pin. */
    if (!getenv("OMP_PLACES"))
        COLOR_PRINT("OMP_PLACES is not set in the environment. OpenMP pinning will not work\n");
}

void __attribute__((destructor)) close_pthread_overload(void)
{
    free(pin_ids);
}

/*
 * libpthread.so.0 pinning via direct overrides
 *
 * We only override pthread_create in order to pin newly created threads.
 * Threads, which are created as part of an OpenMP runtime are explicitly
 * excluded here. The reason is that it's not trivial to determine, which
 * of the OpenMP threads is actually used for execution, and which for
 * internal management ("shepherd threads").
 *
 * Therefore OpenMP pinning is done via the OMP_PLACES environment variable.
 *
 * For everything else, pinning is performed by wrapping the start_routine,
 * doing the pinning there, and then executing the original start_routine.
 */

static void *start_routine_wrapper(void *arg)
{
    assert(arg);

    struct pthread_create_wrapper_arg pcw_arg = *(struct pthread_create_wrapper_arg *)arg;
    free(arg);

    static size_t threads_started = 0;
    static size_t threads_pinned = 0;

    const size_t thread_start_id = __atomic_fetch_add(&threads_started, 1, __ATOMIC_ACQ_REL);

    // If no pinning was provided via environment, immediately start original routine
    if (!pin_ids)
        return pcw_arg.start_routine_orig(pcw_arg.arg_orig);

    assert(pin_ids_count > 0);

    // If the current thread is in the skip mask, do not pin it.
    if (thread_start_id < sizeof(pin_skip_mask) * 8 && (pin_skip_mask & (1ull << thread_start_id)))
        return pcw_arg.start_routine_orig(pcw_arg.arg_orig);

    const size_t thread_pin_id = __atomic_fetch_add(&threads_pinned, 1, __ATOMIC_ACQ_REL);

    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    CPU_SET((int)pin_ids[thread_pin_id % pin_ids_count], &cpu_set);

    if (sched_setaffinity(0, sizeof(cpu_set), &cpu_set) < 0)
        COLOR_PRINT_FORCE("PIN ERROR: sched_setaffinity failed: %s\n", strerror(errno));

    return pcw_arg.start_routine_orig(pcw_arg.arg_orig);
}

int __attribute__ ((visibility ("default") ))
pthread_create(pthread_t *thread,
        const pthread_attr_t *attr,
        void *(*start_routine)(void *),
        void *arg)
{
    bool do_pin = true;

    if (openmp_found) {
        const uintptr_t uip_start_routine = (uintptr_t)start_routine;
        if (uip_start_routine >= openmp_text_start && uip_start_routine < openmp_text_end)
            do_pin = false;
    }

    if (!pthread_create_orig) {
        COLOR_PRINT_FORCE("Cannot call pthread_create. The pthread library either isn't "
                "loaded or pthread_create wasn't found for another reason.\n");
        return ELIBACC;
    }

    if (!do_pin)
        return pthread_create_orig(thread, attr, start_routine, arg);

    /* Call our pthread_create wrapper. We allocate and pass an argument struct to the
     * wrapper, so it can reconstruct the original start_routine call. If the thread
     * is successfuly started, the malloc'ed data is freed in the new thread. */
    struct pthread_create_wrapper_arg *wrapper_arg = malloc(sizeof(*wrapper_arg));
    if (!wrapper_arg)
        return errno;

    wrapper_arg->start_routine_orig = start_routine;
    wrapper_arg->arg_orig = arg;

    int retval = pthread_create_orig(thread, attr, start_routine_wrapper, wrapper_arg);
    if (retval == 0)
        return 0;

    free(wrapper_arg);
    return retval;
}
