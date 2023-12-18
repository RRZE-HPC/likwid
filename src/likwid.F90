! =======================================================================================
!
!     Filename:  likwid.F90
!
!     Description: Marker API f90 module
!
!      Version:   <VERSION>
!      Released:  <DATE>
!
!     Authors:  Jan Treibig (jt), jan.treibig@gmail.com,
!               Thomas Gruber (tr), thomas.roehl@googlemail.com
!     Project:  likwid
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

!> \defgroup Fortran_Interface Likwid Fortran90 Module

!> \ingroup Fortran_Interface
!! Likwid Fortran90 Module for embedding the Marker API into Fortran applications
!! In the basic configuration the module is compiled with the Intel Fortran Compiler
module likwid

interface

!> \ingroup Fortran_Interface
!! \brief Initialize the Likwid Marker API
!! This routine initializes the Marker API for Fortran. It reads some
!! environment commonly set by likwid-perfctr.
!! \note Must be called once in a serial region.
  subroutine likwid_markerInit()
  end subroutine likwid_markerInit

!> \ingroup Fortran_Interface
!! \brief Add current thread to Likwid for Marker API measurements
!! This routine adds the current thread to Likwid that it performs measurements
!! for this thread. If using the daemon access mode, it starts a deamon for the
!! current thread.
!! \note  Must be called once in a parallel region.
  subroutine likwid_markerThreadInit()
  end subroutine likwid_markerThreadInit

!> \ingroup Fortran_Interface
!! \brief Setup performance counters for the next event set
!! If multiple groups should be measured this function
!! switches to the next group in a round robin fashion.
!! Each call reprogramms the performance counters for the current CPU,
!! \note Do not call it while measuring a code region.
  subroutine likwid_markerNextGroup()
  end subroutine likwid_markerNextGroup

!> \ingroup Fortran_Interface
!! \brief Close the Likwid Marker API
!! Close the Likwid Marker API and write measured results to temporary file
!! for evaluation done by likwid-perfctr
!! \note Must be called once in a serial region and no further
!! Likwid calls should be used
  subroutine likwid_markerClose()
  end subroutine likwid_markerClose

!> \ingroup Fortran_Interface
!! \brief Write marker API results to a file
!! Gathers all data of regions and writes them out to file.
!! \note Must be called in serial region of the application.
  subroutine likwid_markerWriteFile( markerfile )
!> \param markerfile [in] The file to write to
  character(*) :: markerfile
  end subroutine likwid_markerWriteFile

!> \ingroup Fortran_Interface
!! \brief Register a code region
!! Initializes the hash table with an empty entry to reduce the overhead
!! at likwid_markerStartRegion()
  subroutine likwid_markerRegisterRegion( regionTag )
!> \param regionTag Name for the code region for later identification
  character(*) :: regionTag
  end subroutine likwid_markerRegisterRegion


!> \ingroup Fortran_Interface
!! \brief Start the measurement for a code region
!! Reads the currently running event set and store the results as start values.
!! for the measurement group identified by regionTag
  subroutine likwid_markerStartRegion( regionTag )
!> \param regionTag Name for the code region for later identification
  character(*) :: regionTag
  end subroutine likwid_markerStartRegion

!> \ingroup Fortran_Interface
!! \brief Stop the measurement for a code region
!! Reads the currently running event set and accumulate the difference between
!! stop and start data in the measurement group identified by regionTag.
  subroutine likwid_markerStopRegion( regionTag )
!> \param regionTag Name for the code region for later identification
  character(*) :: regionTag
  end subroutine likwid_markerStopRegion

!> \ingroup Fortran_Interface
!! \brief Get accumulated measurement results for a code region
!! Get the accumulated data in the measurement group identified by regionTag
!! for the current thread.
!! \warning Experimental
  subroutine likwid_markerGetRegion( regionTag, nr_events, events, time, count )
!> \param regionTag [in] Name for the code region for later identification
!! \param nr_events [in,out] Length of the events array
!! \param events [out] Events array to store intermediate results
!! \param time [out] Accumulated measurement time
!! \param count [out] Call count of the region
  character(*) :: regionTag
  INTEGER :: nr_events
  DOUBLE PRECISION, DIMENSION(*) :: events
  DOUBLE PRECISION :: time
  INTEGER :: count
  end subroutine likwid_markerGetRegion

!> \ingroup Fortran_Interface
!! \brief Reset the counters for a code region to zero
  subroutine likwid_markerResetRegion( regionTag )
!> \param regionTag Name for the code region for later identification
  character(*) :: regionTag
  end subroutine likwid_markerResetRegion

#ifdef LIKWID_WITH_NVMON
!> \ingroup Fortran_NvGPU_Interface
!! \brief Initialize the Likwid NvMarker API
!! This routine initializes the NvMarker API for Fortran. It reads some
!! environment commonly set by likwid-perfctr.
!! \note Must be called once in a serial region.
  subroutine likwid_NvMarkerInit()
  end subroutine likwid_NvMarkerInit

!> \ingroup Fortran_NvGPU_Interface
!! \brief Setup performance counters for the next event set
!! If multiple groups should be measured this function
!! switches to the next group in a round robin fashion.
!! Each call reprogramms the performance counters for the current CPU,
!! \note Do not call it while measuring a code region.
  subroutine likwid_NvMarkerNextGroup()
  end subroutine likwid_NvMarkerNextGroup

!> \ingroup Fortran_NvGPU_Interface
!! \brief Close the Likwid Marker API
!! Close the Likwid Marker API and write measured results to temporary file
!! for evaluation done by likwid-perfctr
!! \note Must be called once in a serial region and no further
!! Likwid calls should be used
  subroutine likwid_NvMarkerClose()
  end subroutine likwid_NvMarkerClose

!> \ingroup Fortran_NvGPU_Interface
!! \brief Register a code region
!! Initializes the hash table with an empty entry to reduce the overhead
!! at likwid_markerStartRegion()
  subroutine likwid_NvMarkerRegisterRegion( regionTag )
!> \param regionTag Name for the code region for later identification
  character(*) :: regionTag
  end subroutine likwid_NvMarkerRegisterRegion


!> \ingroup Fortran_NvGPU_Interface
!! \brief Start the measurement for a code region
!! Reads the currently running event set and store the results as start values.
!! for the measurement group identified by regionTag
  subroutine likwid_NvMarkerStartRegion( regionTag )
!> \param regionTag Name for the code region for later identification
  character(*) :: regionTag
  end subroutine likwid_NvMarkerStartRegion

!> \ingroup Fortran_NvGPU_Interface
!! \brief Stop the measurement for a code region
!! Reads the currently running event set and accumulate the difference between
!! stop and start data in the measurement group identified by regionTag.
  subroutine likwid_NvMarkerStopRegion( regionTag )
!> \param regionTag Name for the code region for later identification
  character(*) :: regionTag
  end subroutine likwid_NvMarkerStopRegion

!> \ingroup Fortran_NvGPU_Interface
!! \brief Get accumulated measurement results for a code region
!! Get the accumulated data in the measurement group identified by regionTag
!! for the current thread.
!! \warning Experimental
!! subroutine likwid_markerGetRegion( regionTag, nr_events, events, time, count )
!> \param regionTag [in] Name for the code region for later identification
!! \param nr_events [in,out] Length of the events array
!! \param events [out] Events array to store intermediate results
!! \param time [out] Accumulated measurement time
!! \param count [out] Call count of the region
!! character(*) :: regionTag
!! INTEGER :: nr_events
!! DOUBLE PRECISION, DIMENSION(*) :: events
!! DOUBLE PRECISION :: time
!! INTEGER :: count
!! end subroutine likwid_markerGetRegion

!> \ingroup Fortran_NvGPU_Interface
!! \brief Reset the counters for a code region to zero
  subroutine likwid_NvMarkerResetRegion( regionTag )
!> \param regionTag Name for the code region for later identification
  character(*) :: regionTag
  end subroutine likwid_NvMarkerResetRegion

#endif /* LIKWID_WITH_NVMON */

#ifdef LIKWID_WITH_ROCMON
!> \ingroup Fortran_RocmGPU_Interface
!! \brief Initialize the Likwid RocmonMarker API
!! This routine initializes the RocmonMarker API for Fortran. It reads some
!! environment commonly set by likwid-perfctr.
!! \note Must be called once in a serial region.
  subroutine likwid_RocmMarkerInit()
  end subroutine likwid_RocmMarkerInit

!> \ingroup Fortran_RocmGPU_Interface
!! \brief Setup performance counters for the next event set
!! If multiple groups should be measured this function
!! switches to the next group in a round robin fashion.
!! Each call reprogramms the performance counters for the current CPU,
!! \note Do not call it while measuring a code region.
  subroutine likwid_RocmMarkerNextGroup()
  end subroutine likwid_RocmMarkerNextGroup

!> \ingroup Fortran_RocmGPU_Interface
!! \brief Close the Likwid RocmonMarker API
!! Close the Likwid Marker API and write measured results to temporary file
!! for evaluation done by likwid-perfctr
!! \note Must be called once in a serial region and no further
!! Likwid calls should be used
  subroutine likwid_RocmMarkerClose()
  end subroutine likwid_RocmMarkerClose

!> \ingroup Fortran_RocmGPU_Interface
!! \brief Register a code region
!! Initializes the hash table with an empty entry to reduce the overhead
!! at likwid_RocmMarkerStartRegion()
  subroutine likwid_RocmMarkerRegisterRegion( regionTag )
!> \param regionTag Name for the code region for later identification
  character(*) :: regionTag
  end subroutine likwid_RocmMarkerRegisterRegion


!> \ingroup Fortran_RocmGPU_Interface
!! \brief Start the measurement for a code region
!! Reads the currently running event set and store the results as start values.
!! for the measurement group identified by regionTag
  subroutine likwid_RocmMarkerStartRegion( regionTag )
!> \param regionTag Name for the code region for later identification
  character(*) :: regionTag
  end subroutine likwid_RocmMarkerStartRegion

!> \ingroup Fortran_RocmGPU_Interface
!! \brief Stop the measurement for a code region
!! Reads the currently running event set and accumulate the difference between
!! stop and start data in the measurement group identified by regionTag.
  subroutine likwid_RocmMarkerStopRegion( regionTag )
!> \param regionTag Name for the code region for later identification
  character(*) :: regionTag
  end subroutine likwid_RocmMarkerStopRegion

!> \ingroup Fortran_RocmGPU_Interface
!! \brief Get accumulated measurement results for a code region
!! Get the accumulated data in the measurement group identified by regionTag
!! for the current thread.
!! \warning Experimental
!! subroutine likwid_markerGetRegion( regionTag, nr_events, events, time, count )
!> \param regionTag [in] Name for the code region for later identification
!! \param nr_events [in,out] Length of the events array
!! \param events [out] Events array to store intermediate results
!! \param time [out] Accumulated measurement time
!! \param count [out] Call count of the region
!! character(*) :: regionTag
!! INTEGER :: nr_events
!! DOUBLE PRECISION, DIMENSION(*) :: events
!! DOUBLE PRECISION :: time
!! INTEGER :: count
!! end subroutine likwid_markerGetRegion

!> \ingroup Fortran_RocmGPU_Interface
!! \brief Reset the counters for a code region to zero
  subroutine likwid_RocmMarkerResetRegion( regionTag )
!> \param regionTag Name for the code region for later identification
  character(*) :: regionTag
  end subroutine likwid_RocmMarkerResetRegion

#endif /* LIKWID_WITH_ROCMON */

end interface

end module likwid

