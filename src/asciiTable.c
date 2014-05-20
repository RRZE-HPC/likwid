/*
 * ===========================================================================
 *
 *      Filename:  asciiTable.c
 *
 *      Description:  Module implementing output of ascii table.
 *
 *      Version:  <VERSION>
 *      Created:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Company:  RRZE Erlangen
 *      Project:  likwid
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



/* #####   HEADER FILE INCLUDES   ######################################### */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include <types.h>
#include <strUtil.h>
#include <asciiTable.h>

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

TableContainer*
asciiTable_allocate(int numRows,int numColumns, bstrList* headerLabels)
{
    int i;
    TableContainer* container;

    container = (TableContainer*) malloc(sizeof(TableContainer));
    container->numRows = numRows;
    container->numColumns = numColumns;
    container->currentRow = 0;
    container->printed = 0;

    if (numColumns != headerLabels->qty)
    {
        fprintf(stderr, "asciiTable: Number of columns not equal to number of header labels!\n");
        exit(EXIT_FAILURE);
    }

    container->header = bstrListCreate();
    bstrListAlloc (container->header, numColumns);

    for(i=0; i<numColumns; i++)
    {
        container->header->entry[i] = bstrcpy(headerLabels->entry[i]);
    }

    container->rows = (bstrList**) malloc( numRows * sizeof(bstrList*));

    for(i=0; i<numRows; i++)
    {
        container->rows[i] = bstrListCreate();
        bstrListAlloc (container->rows[i], numColumns);
    }

    return container;
}

void 
asciiTable_free(TableContainer* container)
{
    int i;

    if(container == NULL)
    {
        fprintf(stderr, "asciiTable_free: Cannot free NULL reference!\n");
        return;
    }

    bstrListDestroy(container->header);

    for(i=0; i<container->numRows; i++)
    {
        bstrListDestroy(container->rows[i]);
    }

    free(container->rows);
}

void
asciiTable_insertRow(TableContainer* container, int row, bstrList* fields)
{
    int i;

    if (container->numColumns != fields->qty)
    {
        fprintf(stderr, "asciiTable: Number of colummns not equal to number of field labels!\n");
        exit(EXIT_FAILURE);
    }

    if (row >= container->numRows)
    {
        fprintf(stderr, "asciiTable: Number of Rows smaller than requested row index!\n");
        exit(EXIT_FAILURE);
    }

    for(i=0; i<container->numColumns; i++)
    {
        container->rows[row]->entry[i] = bstrcpy(fields->entry[i]);
        container->rows[row]->qty++;
    }
}

void
asciiTable_appendRow(TableContainer* container, bstrList* fields)
{
    asciiTable_insertRow(container, container->currentRow++, fields);
}

void
asciiTable_setCurrentRow(TableContainer* container, int row)
{
    container->currentRow = row;
}

void
asciiTable_print(TableContainer* container)
{
    int i;
    int j;
    int* boxwidth;

    boxwidth = (int*) malloc(container->numColumns * sizeof(int));

    for (j=0; j<container->numColumns; j++) boxwidth[j] = 0;

    for (j=0; j<container->numColumns; j++)
    {
        boxwidth[j] = MAX(boxwidth[j],blength(container->header->entry[j]));
    }

    /* determine maximum label width in each column */
    for (i=0; i<container->numRows; i++)
    {
        for (j=0; j<container->numColumns; j++)
        {
            //           btrimws(container->rows[i]->entry[j]);
            boxwidth[j] = MAX(boxwidth[j],blength(container->rows[i]->entry[j]));
        }
    }

    if (! container->printed)
    {
        /* Increase boxwidth with two spaces */
        for (j=0; j<container->numColumns; j++) boxwidth[j] +=2;
    }

    /* print header */

    for (j=0; j<container->numColumns; j++)
    {
        printf("+");
        for (i=0;i<boxwidth[j];i++)
        {
            printf("-");
        }
    }
    printf("+\n");

    for (j=0; j<container->numColumns; j++)
    {
        printf("|");
        bJustifyCenter(container->header->entry[j],boxwidth[j]);
        printf("%s",bdata(container->header->entry[j]));
    }
    printf("|\n");

    for (j=0; j<container->numColumns; j++)
    {
        printf("+");
        for (i=0;i<boxwidth[j];i++)
        {
            printf("-");
        }
    }
    printf("+\n");

    for (i=0; i<container->numRows; i++)
    {
        for (j=0; j<container->numColumns; j++)
        {
            printf("|");
            bJustifyCenter(container->rows[i]->entry[j],boxwidth[j]);
            printf("%s",bdata(container->rows[i]->entry[j]));
        }
        printf("|\n");
    }

    for (j=0; j<container->numColumns; j++)
    {
        printf("+");
        for (i=0;i<boxwidth[j];i++)
        {
            printf("-");
        }
    }
    printf("+\n");
    container->printed = 1;

    free(boxwidth);
}




