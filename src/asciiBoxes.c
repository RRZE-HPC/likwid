/*
 * ===========================================================================
 *
 *      Filename:  asciiBoxes.c
 *
 *      Description:  Module implementing output of nested ascii art boxes
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

#include <error.h>
#include <types.h>
#include <asciiBoxes.h>


/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

BoxContainer*
asciiBoxes_allocateContainer(int numLines, int numColumns)
{
    int i;
    BoxContainer* container;

    container = (BoxContainer*) malloc(sizeof(BoxContainer));
    container->numLines = numLines;
    container->numColumns = numColumns;

    container->boxes = (Box**) malloc(numLines * sizeof(Box*));

    for(i=0; i<numLines; i++)
    {
        container->boxes[i] = (Box*) malloc(numColumns * sizeof(Box));
    }

#if 0
    for(i=0; i<numLines; i++)
    {
        for(j=0; j<numColumns; j++)
        {
            container->boxes[i][j].width = 0;
            container->boxes[i][j].label = NULL;
        }
    }
#endif


    return container;
}

void 
asciiBoxes_addBox(BoxContainer* container, int line, int column, bstring label)
{
    if( line >= container->numLines)
    {
        ERROR_PMSG(line id %d too large,line);
    }
    if( column >= container->numColumns)
    {
        ERROR_PMSG(column id %d too large,column);
    }

    container->boxes[line][column].width = 1;
    container->boxes[line][column].label = bstrcpy(label);
}


void
asciiBoxes_addJoinedBox(BoxContainer* container,
        int line,
        int startColumn,
        int endColumn,
        bstring label)
{
    if( line >= container->numLines)
    {
        ERROR_PMSG(line id %d too large,line);
    }
    if( endColumn >= container->numColumns)
    {
        ERROR_PMSG(column id %d too large,endColumn);
    }

    container->boxes[line][startColumn].width = (endColumn-startColumn)+1;
    container->boxes[line][startColumn].label = bstrcpy(label);
}

void
asciiBoxes_print(BoxContainer* container)
{
    int i;
    int j;
    int k;
    int width;
    int boxwidth=0; /* box width is inner width of box */

    /* determine maximum label width */
    for (i=0; i<container->numLines; i++)
    {
        for (j=0; j<container->numColumns; j++)
        {
            btrimws(container->boxes[i][j].label);
            boxwidth = MAX(boxwidth,blength(container->boxes[i][j].label));
        }
    }
    boxwidth += 2;  /* add one space each side */

    /* top line */
    printf("+");
    for (i=0;
            i<(container->numColumns * (boxwidth+2) +
                (container->numColumns+1));  /* one space between boxes */
            i++)
    {
        printf("-");
    }
    printf("+\n");

    for (i=0; i<container->numLines; i++)
    {
        /* Box top line */
        printf("| ");
        for (j=0; j<container->numColumns; j++)
        {
            printf("+");
            if(container->boxes[i][j].width == 1)
            {
                for (k=0; k<boxwidth; k++)
                {
                    printf("-");
                }
            }
            else 
            {
                for (k=0;
                        k<(container->boxes[i][j].width * boxwidth +
                            (container->boxes[i][j].width-1)*3);
                        k++)
                {
                    printf("-");
                }
                j+= container->boxes[i][j].width-1;
            }
            printf("+ ");
        }
        printf("|\n");
        printf("| ");

        /* Box label line */
        for (j=0; j<container->numColumns; j++)
        {
            int offset=0;

            /* center label */
            if(container->boxes[i][j].width == 1)
            {
         //       printf("\nblength %d\n",blength(container->boxes[i][j].label));
                width = (boxwidth - blength(container->boxes[i][j].label))/2;

                offset = (boxwidth - blength(container->boxes[i][j].label))%2;
            }
            else
            {
                width = (container->boxes[i][j].width * boxwidth +
                        ((container->boxes[i][j].width-1)*3) -
                        blength(container->boxes[i][j].label))/2;

                offset = (container->boxes[i][j].width * boxwidth +
                        ((container->boxes[i][j].width-1)*3) -
                        blength(container->boxes[i][j].label))%2;
            }

            printf("|");

            for (k=0; k<(width+offset); k++)
            {
                printf(" ");
            }

            printf("%s",container->boxes[i][j].label->data);

            for (k=0; k<width; k++)
            {
                printf(" ");
            }
            printf("| ");

            if(container->boxes[i][j].width != 1)
            {
                j+= container->boxes[i][j].width-1;
            }
        }
        printf("|\n");
        printf("| ");

        /* Box bottom line */
        for (j=0; j<container->numColumns; j++)
        {
            printf("+");
            if(container->boxes[i][j].width == 1)
            {
                for (k=0; k<boxwidth; k++)
                {
                    printf("-");
                }
            }
            else 
            {
                for (k=0;
                        k<(container->boxes[i][j].width * boxwidth +
                            (container->boxes[i][j].width-1)*3);
                        k++)
                {
                    printf("-");
                }
                j+= container->boxes[i][j].width-1;
            }
            printf("+ ");

        }
        printf("|\n");
    }

    /* bottom line */
    printf("+");
    for (i=0; i< (container->numColumns * (boxwidth+2) + 
                container->numColumns+1); i++)
    {
        printf("-");
    }
    printf("+\n");
}




