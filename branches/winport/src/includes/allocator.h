/*
 * ===========================================================================
 *
 *      Filename:  allocator.h
 *
 *      Description:  Header File allocator Module. 
 *
 *      Version:  1.0
 *      Created:  04/05/2010
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Company:  RRZE Erlangen
 *      Project:  none
 *      Copyright:  Copyright (c) 2010, Jan Treibig
 *
 *      This program is free software; you can redistribute it and/or modify
 *      it under the terms of the GNU General Public License, v2, as
 *      published by the Free Software Foundation
 *     
 *      This program is distributed in the hope that it will be useful,
 *      but WITHOUT ANY WARRANTY; without even the implied warranty of
 *      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *      GNU General Public License for more details.
 *     
 *      You should have received a copy of the GNU General Public License
 *      along with this program; if not, write to the Free Software
 *      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *
 * ===========================================================================
 */

#ifndef ALLOCATOR_H
#define ALLOCATOR_H

#include <types.h>
#include <bstrlib.h>

extern void allocator_init(int numVectors);
extern void allocator_finalize();
extern void allocator_allocateVector(void** ptr,
        int alignment,
        int size,
        int offset,
        DataType type,
        bstring domain);

#endif /*ALLOCATOR_H*/

