/*
This file is part of GOTCHA.  For copyright information see the COPYRIGHT
file in the top level directory, or at
https://github.com/LLNL/gotcha/blob/master/COPYRIGHT
This program is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License (as published by the Free
Software Foundation) version 2.1 dated February 1999.  This program is
distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the terms and conditions of the GNU Lesser General Public License
for more details.  You should have received a copy of the GNU Lesser General
Public License along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
*/

#include <stdio.h>
#include <string.h>

extern int init_autotee(char *filename);
extern int close_autotee();

#define OUTPUT_FILE "tee.out"

int main()
{
   int result;

   printf("Every stdout print after this line should also appear in %s:\n", OUTPUT_FILE);

   result = init_autotee(OUTPUT_FILE);
   if (result != 0)
      return -1;

   printf("First line\n");
   printf("Second %s\n", "line");
   fprintf(stdout, "Third line\n");
   fprintf(stdout, "%s line\n", "Forth");
   puts("Fifth line");
   fputs("Sixth ", stdout);
   fputs("line\n", stdout);
   fwrite("Seventh line\n", 1, strlen("Seventh line\n"), stdout);
   fprintf(stderr, "Eighth line is stderr and should not appear in in %s\n", OUTPUT_FILE);
   close_autotee();
   printf("Ninth line is after close and should not appear in %s\n", OUTPUT_FILE);

   return 0;
}
