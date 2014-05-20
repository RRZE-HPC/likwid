      PROGRAM testmarker
     
#include <likwid_f90.h>
     
      IMPLICIT NONE
     
      INTEGER, PARAMETER :: n=10000000, nrep1 = 100, nrep2 = 100
      INTEGER :: i=0
     
      REAL(kind=8)    :: a(n), b(n), c(n), s
     
      DO i = 1, n
      a(i) = 1.0/float(i)
      b(i) = 1.0
      c(i) = float(i)
      END DO
     
      CALL likwid_markerInit()
     
! dummy
      CALL likwid_markerStart("dummy")
      CALL dummy()
      CALL likwid_markerStop("dummy")
     
! sub
      CALL likwid_markerStart("sub")
      DO i = 1, nrep1
      CALL sub(n, a, b, c)
      end do
      CALL likwid_markerStop("sub")
     
! another
      CALL likwid_markerStart("another")
      DO i = 1, nrep2
      CALL another(n, a, s)
      b(i) = b(i) + s
      END DO
      CALL likwid_markerStop("another")
     
! oncemore sub
      CALL likwid_markerStart("sub2")
      DO i = 1, nrep1
      CALL sub(n, a, b, c)
      END DO
      CALL likwid_markerStop("sub2")
     
      CALL likwid_markerClose()
     
      PRINT *,'job done'
      STOP
      END
     
      SUBROUTINE sub(n, a, b, c)
     
      REAL*8 a(n), b(n), c(n)
     
      s = 0.0
      DO i = 1, n
      a(i) = sin( sqrt( exp( b(i) * c(i) - dble(i) ) ) )
      END DO
     
      RETURN
      END
     
      SUBROUTINE another(n, a, s)
     
      REAL*8 a(n), s
     
      s = 0.0
      DO i = 1, n
      s = s + sin( sqrt( exp ( a(i) ) ) )
      END DO
     
      RETURN
      END
     
      SUBROUTINE dummy()
      RETURN
      END

