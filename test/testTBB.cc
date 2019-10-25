/*
    File: testTBB.cc
    Author: timday (stackoverflow)
    Source: http://stackoverflow.com/questions/10607215/simplest-tbb-example

    Extended by Thomas Roehl to do LIKWID Marker API calls and print the CPU for
    the threads instead of 'n'
*/

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/task_scheduler_init.h"
#include <iostream>
#include <vector>

// Added by Thomas Roehl
#include <sched.h>
#include <likwid-cpumarker.h>


struct mytask {
  mytask(size_t n)
    :_n(n)
  {}
  void operator()() {
    
    for (int i=0;i<10000000;++i) {}  // Deliberately run slow
    std::cerr << "[" << sched_getcpu() << "]";
    
  }
  size_t _n;
};

struct executor
{
  executor(std::vector<mytask>& t)
    :_tasks(t)
  {}
  executor(executor& e,tbb::split)
    :_tasks(e._tasks)
  {}

  void operator()(const tbb::blocked_range<size_t>& r) const {
    LIKWID_MARKER_START("TBB");
    for (size_t i=r.begin();i!=r.end();++i)
      _tasks[i]();
    LIKWID_MARKER_STOP("TBB");
  }

  std::vector<mytask>& _tasks;
};

int main(int,char**) {

  tbb::task_scheduler_init init;  // Automatic number of threads

  LIKWID_MARKER_INIT;
  std::vector<mytask> tasks;
  for (int i=0;i<1000;++i)
    tasks.push_back(mytask(i));

  executor exec(tasks);
  tbb::parallel_for(tbb::blocked_range<size_t>(0,tasks.size()),exec);
  std::cerr << std::endl;
  LIKWID_MARKER_CLOSE;
  return 0;
}
