! =======================================================================================
!
!      Filename:  F-markerAPI.F90
!
!      Description:  Example how to use the Fortran90 Marker API
!
!      Version:   <VERSION>
!      Released:  <DATE>
!
!      Author:  Thomas Roehl (tr), thomas.roehl@googlemail.com
!      Project:  likwid
!
!      Copyright (C) 2015 RRZE, University Erlangen-Nuremberg
!
!      This program is free software: you can redistribute it and/or modify it under
!      the terms of the GNU General Public License as published by the Free Software
!      Foundation, either version 3 of the License, or (at your option) any later
!      version.
!
!      This program is distributed in the hope that it will be useful, but WITHOUT ANY
!      WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
!      PARTICULAR PURPOSE.  See the GNU General Public License for more details.
!
!      You should have received a copy of the GNU General Public License along with
!      this program.  If not, see <http://www.gnu.org/licenses/>.
!
! =======================================================================================

#define SLEEPTIME 2

program FmarkerAPI
    use likwid
    include "omp_lib.h"
    INTEGER :: nr_events
    DOUBLE PRECISION, DIMENSION(10) :: events
    DOUBLE PRECISION :: time
    INTEGER :: c
    nr_events = 10
    ! Init Marker API in serial region once in the beginning.
    call likwid_markerInit()

!$OMP PARALLEL
    ! Each thread must add itself to the Marker API, therefore must be
    ! in parallel region.
    call likwid_markerthreadInit()
    ! Optional. Register region name and initialize hash table entries.
    call likwid_markerRegisterRegion("example")
!$OMP END PARALLEL

!$OMP PARALLEL
    print '(a,i0,a,i0,a)', "Thread ", omp_get_thread_num()," sleeps now for ", SLEEPTIME," seconds"
    ! Start measurements inside a parallel region.
    call likwid_markerStartRegion("example")
    ! Insert your code here
    ! Often contains an OpenMP for pragma. Regions can be nested.
    call Sleep(SLEEPTIME)
    ! Stop measurements inside a parallel region.
    call likwid_markerStopRegion("example")
    print '(a,i0,a)', "Thread ", omp_get_thread_num()," wakes up again"
    ! If multiple groups given, you can switch to the next group.
    call likwid_markerNextGroup();
    ! If you need the performance data inside your application, use
    call likwid_markerGetRegion("example", nr_events, events, time, c)
    ! Events is an array of DOUBLE PRECISION with nr_events (INTEGER) entries,
    ! time is a DOUBLE PRECISION and count an INTEGER.
    ! After returning the events array contains maximally nr_events results.
    print '(a,i0,a,f9.3)', "Region example measures ", nr_events, " events, total measurement time is ", time
    print '(a,i0,a)', "The region was called ", c, " times"
    do i=1,nr_events
        print '(a,i0,a,e13.7)', "Event ",i,": ",events(i)
    end do
    
!$OMP END PARALLEL

! Close Marker API and write results to file for further evaluation done
! by likwid-perfctr.
call likwid_markerClose()

end program FmarkerAPI
