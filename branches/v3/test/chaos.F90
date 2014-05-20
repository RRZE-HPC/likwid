program testmarker

#include <likwid_f90.h>

implicit none

integer, parameter :: n=10000000, nrep1 = 100, nrep2 = 100
integer :: i=0

real(kind=8)    :: a(n), b(n), c(n), s

do i = 1, n
   a(i) = 1.0/float(i)
   b(i) = 1.0
   c(i) = float(i)
end do

call likwid_markerInit()

! dummy
call likwid_markerStart("dummy")
call dummy()
call likwid_markerStop("dummy")

! sub
call likwid_markerStart("sub")
do i = 1, nrep1
  call sub(n, a, b, c)
end do
call likwid_markerStop("sub")

! another
call likwid_markerStart("another")
do i = 1, nrep2
  call another(n, a, s)
  b(i) = b(i) + s
end do
call likwid_markerStop("another")

! oncemore sub
call likwid_markerStart("sub2")
do i = 1, nrep1
  call sub(n, a, b, c)
end do
call likwid_markerStop("sub2")

call likwid_markerClose()

print *,'job done'
stop
end

subroutine sub(n, a, b, c)

real*8 a(n), b(n), c(n)

s = 0.0
do i = 1, n
  a(i) = sin( sqrt( exp( b(i) * c(i) - dble(i) ) ) )
end do

return
end

subroutine another(n, a, s)

real*8 a(n), s

s = 0.0
do i = 1, n
  s = s + sin( sqrt( exp ( a(i) ) ) )
end do

return
end

subroutine dummy()
return
end

