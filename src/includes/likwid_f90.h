! =======================================================================================
!
!     Filename:  likwid_f90.h
!
!     Description: Marker API f90 module
!
!      Version:   <VERSION>
!      Released:  <DATE>
!
!     Author:  Jan Treibig (jt), jan.treibig@gmail.com
!     Project:  likwid
!
!      Copyright (C) 2012 Jan Treibig 
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

#ifndef LIKWID_F90_H
#define LIKWID_F90_H

      use likwid

#define likwid_markerStart(region) \
 likwid_markerStartRegion(region,len_trim(region))

#define likwid_markerStop(region) \
 likwid_markerStopRegion(region,len_trim(region))

#endif
