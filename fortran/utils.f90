!!! Time-stamp: <utils.f90 11:46, 17 Mar 2004 by P Sunthar>

!_______________________________________
!

!!! $Log: utils.f90,v $
!!! Revision 1.1  2004/01/29 06:40:09  pxs565
!!! Initial revision
!!!

module random_numbers
   !****************************************************************!***********
   ! This module contains utlities to generate and set random numbers
   ! Has a few internal seeds, as well as a utility to reset the seeds to some
   ! given value.
   !****************************************************************!***********

   Use Global_parameters_variables_and_types

   implicit none

   private

   public GaussRand, ran_1
   public set_seed, reset_RNG_with_seed, get_all_parameters

   Integer(k4b), Save :: ix=1, iy = -1, seed = 123

contains

   subroutine set_seed(seed_in)
      Integer(k4b), intent(in) :: seed_in

      seed = seed_in

      !print *, "my RNG seed is: ", seed

   end subroutine

   subroutine reset_RNG_with_seed(seed_in, ix_in, iy_in)
      Integer(k4b), intent(in) :: seed_in
      Integer(k4b), intent(in), optional :: ix_in, iy_in
      Integer(k4b), parameter :: ix_default = 1, iy_default = -1

      if (present(ix_in)) then
         ix = ix_in
      else
         ix = ix_default
      end if

      if (present(iy_in)) then
         iy = iy_in
      else
         iy = iy_default
      end if

      seed = seed_in

   end subroutine

   subroutine get_all_parameters(seed_out, ix_out, iy_out)
      Integer(k4b), intent(out) :: ix_out, iy_out, seed_out

      ix_out = ix
      iy_out = iy
      seed_out = seed

   end subroutine

   Subroutine GaussRand(N,GX)
      !!! Returns a normalised gaussian random variable with mean zero
      !!! and variance 1: f(r) = exp(-r^2/2) / sqrt(2*Pi)
      !!! To obtain a dist with a given sigma use:  sqrt(sigma) * Gx

      Integer(k4b), Intent (in) :: N
!          Integer(k4b), Intent (inout) :: seed
      Real (DBprec), Intent (out), Dimension(N) :: GX

      Real (DBprec) rnd(2)
      Real (DBprec) x,y, rsq, fac
      Integer i
      ! Integer :: n2
      Real (DBprec) :: rtemp(N+1)


      Do i = 1,N,2
         rsq = 0.d0
         Do While (rsq .Ge. 1.0  .Or.  rsq .Eq. 0.0)
            Call ran_1(2,rnd)
            x = 2.d0*rnd(1) -1.d0
            y = 2.d0*rnd(2) -1.d0
            rsq = x*x + y*y
         End Do
         fac = Sqrt(-2.d0*Log(rsq)/rsq);
         rtemp(i) = x*fac
         rtemp(i+1) = y*fac
      End Do

      GX = rtemp(1:N)

   End Subroutine GaussRand

   Subroutine ran_1(n, X)
      !Integer(k4b), Intent(inout) ::idum
      Integer(k4b), Intent(in) :: n
      Real (DBprec), Intent(inout) :: X(n)
      !---------------------------------------------------------------------c
      !     Random number generator based on procedure given in             c
      !     Numerical Recipes in Fortran 90, Chapter B7, pp. 1141-1143      c
      !     This generator is supposed to have a period of about 3.1E18.    c
      !                                                                     c
      !     The calling sequence is the same as that of RANL,               c
      !     the BLAS random number generator. The arguments are:            c
      !     n - gives the dimension of vector R                             c
      !     X - 1D array of dimension n; on exit                            c
      !         contains n unifromly distributed random numbers in (0,1)    c
      !     idum |- seeds which are updated on exit                         c
      !---------------------------------------------------------------------c

      Integer(k4b),Parameter :: IA=16807,IM=2147483647,IQ=127773,IR=2836
      Real, Save :: am
      Integer(k4b) :: k
      Integer(k4b) count

      If ((seed.Le.0) .Or. (iy.Lt.0)) Then
         am = Nearest(1.0,-1.0)/real(IM)
         iy = Ior(Ieor(888889999,Abs(seed)),1)
         ix = Ieor(777755555,Abs(seed))
         seed = Abs(seed) + 1
      End If

      Do count = 1,n
         ix = Ieor(ix,Ishft(ix,13))
         ix = Ieor(ix, Ishft(ix,-17))
         ix = Ieor(ix,Ishft(ix, 5))
         k = iy/IQ
         iy = IA*(iy-k*IQ)-IR*k
         If (iy.Lt.0) iy = iy + IM
         X(count) = am*Ior(Iand(IM,Ieor(ix,iy)),1)
      End Do

   End Subroutine ran_1

end module

module simulation_utilities
   ! This module contains random utilities which aren't necessarily specific to
   ! this BD simulation

   Use Global_parameters_variables_and_types
   Implicit None

contains

   SUBROUTINE PRINT_MATRIX( DESC, M, N, A)
      ! tries to print a matrix in column-major form
      CHARACTER*(*), intent(in)  ::  DESC
      INTEGER, intent(in)   ::       M, N
      REAL(DBprec), intent(in) :: A( :, : )

      INTEGER :: I, J

      WRITE(*,*)
      WRITE(*,*) DESC
      DO I = 1, M
         print *,  ( A( I, J ), J = 1, N )
      END DO

      RETURN
   END



   subroutine hunt(xx, x, jlo)
      ! Given an array xx(1:N), and a given value x, this returns an index jlo such that
      ! x is between xx(jlo) and xx(jlo+1). xx must be monotonic, increasing or decreasing.
      ! jlo=0 or jlo = N means that x is out of the range. The input jlo is taken as the
      ! initial guess for output jlo

      implicit none
      Integer(k4b), intent(inout) :: jlo
      Real(DBprec), intent(in) :: x
      Real(Dbprec), intent(in), dimension(:) :: xx

      Integer(k4b) :: n, inc, jhi, jm
      logical :: ascnd
      n = size(xx)
      ascnd = (xx(n) >= xx(1))
      if (jlo <= 0 .or. jlo > n) then
         ! go straight to bisection
         jlo = 0
         jhi = n+1
      else
         ! begin hunt algorithm
         inc = 1
         if (x >= xx(jlo) .eqv. ascnd) then
            do
               jhi = jlo + inc
               if (jhi > n) then
                  jhi = n+1
                  exit
               else
                  if (x < xx(jhi) .eqv. ascnd) exit
                  jlo = jhi
                  inc = inc + inc
               end if
            end do
         else
            jhi = jlo
            do
               jlo = jhi - inc
               if (jlo < 1 ) then
                  jlo = 0
                  exit
               else
                  if (x >= xx(jlo) .eqv. ascnd) exit
                  jhi = jlo
                  inc = inc + inc
               end if
            end do
         end if
      end if

      ! Hunt is done, do bisection
      do
         if (jhi - jlo <= 1) then
            if (x == xx(n)) jlo = n-1
            if (x == xx(1)) jlo = 1
            exit
         else
            jm = (jhi + jlo)/2
            if (x >= xx(jm) .eqv. ascnd) then
               jlo = jm
            else
               jhi = jm
            end if
         end if
      end do

   end subroutine

   subroutine polint(xa,ya,x,y,dy)
      Implicit none
      Real(DBprec) , Dimension(:), Intent(in) :: xa, ya
      Real(DBprec), intent(in) :: x
      Real(Dbprec), intent(out) :: y, dy
      ! Given arrays xa and ya of length N, and a given value x, this routine returns a value y, and an
      ! error estimate dy. If P(x) is the polynomial of degree N-1 such that P(xa_i) = ya_i, i=1,...N, then
      ! the returned value y = P(x)
      ! This subroutine is from numerical recipes in fortran 90, pg 1043.
      Integer(k4b) :: m,n,ns, ns_loc(1)
      Real(DBprec), dimension(size(xa)) :: c, d, den, ho

      n = size(xa)
      c = ya
      d = ya
      ho = xa-x;
      ns_loc = minloc(abs(x-xa))
      ns = ns_loc(1)
      y = ya(ns)
      ns = ns-1
      do m=1,n-1
         den(1:n-m) = ho(1:n-m)-ho(1+m:n)
         if (any(den(1:n-m) == 0)) then
            print *, 'polint: calculation failure'
            stop
         end if
         den(1:n-m) = (c(2:n-m+1)-d(1:n-m))/den(1:n-m)
         d(1:n-m) = ho(1+m:n)*den(1:n-m)
         c(1:n-m) = ho(1:n-m)*den(1:n-m)
         if (2*ns < n-m) then
            dy = c(ns+1)
         else
            dy = d(ns)
            ns = ns-1
         end if
         y = y + dy
      end do
   end subroutine

   function lin_interp(xin, x_table, y_table) result(yout)
      ! given a value xin and a table of x-y associated values x_table and y_table,
      ! determine the linearly interpolated value yout.
      ! x_table must be strictly ascending.
      real(DBprec), dimension(:), intent(in) ::  x_table, y_table
      real(DBprec), intent(in) :: xin
      real(DBprec) :: yout
      integer :: i, n
      real(DBprec) :: x0,x1,y0,y1

      n = size(x_table)

      ! ideally we would only do this check when debugging!
      if (xin < x_table(1) .or. xin > x_table(n)) then
         print*, "Error: x is out of bounds"
         stop
      end if

      ! hunt for the right value
      i = int(n/2)
      call hunt(x_table, xin, i)

      y0 = y_table(i)
      y1 = y_table(i + 1)
      x0 = x_table(i)
      x1 = x_table(i + 1)
      yout = y0 + (y1 - y0)*(xin - x0)/(x1 - x0)
      !print *, xin, i, x0, x1, y0, y1, yout

      !   do i=1,n
      !       if (xin >= x_table(i)) then
      !           y0 = y_table(i)
      !           y1 = y_table(i + 1)
      !           x0 = x_table(i)
      !           x1 = x_table(i + 1)
      !           yout = y0 + (y1 - y0)*(xin - x0)/(x1 - x0)
      !           print *, x0, x1, y0, y1, yout
      !           return
      !       end if
      !   end do

   end function lin_interp

   function lin_interp_vector(xin, x_table, y_table) result(yout)
      ! given a value xin and a table of x-y associated values x_table and y_table,
      ! determine the linearly interpolated value yout.
      ! x_table must be strictly ascending.
      ! y_table must be of size (:, length(x_table))
      ! yout will be of length (second dimension of y_table - 1)
      real(DBprec), dimension(:), intent(in) ::  x_table
      real(DBprec), dimension(:,:), intent(in) :: y_table
      real(DBprec), intent(in) :: xin
      real(DBprec), dimension(size(y_table,1)) :: yout
      integer :: i, n, j, m
      real(DBprec) :: x0, x1
      real(DBprec), dimension(size(y_table,1)) :: y0, y1

      n = size(x_table)
      m = size(y_table, 1)  ! length of output vector

      ! ideally we would only do these checks when debugging!
      if (xin < x_table(1) .or. xin > x_table(n)) then
         print*, "Error: x is out of bounds"
         stop
      end if
      if (n .ne. size(y_table,2)) then
         print*, "Error: the size of y doesn't match the size of x"
         stop
      end if

      ! hunt for the right value
      i = int(n/2)
      call hunt(x_table, xin, i)

      ! Get the y values at the two x points for interpolation
      y0 = y_table(:, i)
      y1 = y_table(:, i + 1)
      x0 = x_table(i)
      x1 = x_table(i + 1)

      ! Perform linear interpolation for each column
      do j = 1, m
         yout(j) = y0(j) + (y1(j) - y0(j))*(xin - x0)/(x1 - x0)
      end do
      !print *, xin, i, x0, x1, y0, y1, yout

      !   do i=1,n
      !       if (xin >= x_table(i)) then
      !           y0 = y_table(i)
      !           y1 = y_table(i + 1)
      !           x0 = x_table(i)
      !           x1 = x_table(i + 1)
      !           yout = y0 + (y1 - y0)*(xin - x0)/(x1 - x0)
      !           print *, x0, x1, y0, y1, yout
      !           return
      !       end if
      !   end do

   end function lin_interp_vector

   function inverse_lin_interp(x, fx, fxval)
      real(DBprec), dimension(:), intent(in) :: x, fx
      real(DBprec), intent(in) :: fxval
      real(DBprec) :: inverse_lin_interp
      integer :: i

      do i=1,size(x)
         if (fx(i) > fxval) then
            inverse_lin_interp = (fxval-fx(i-1))/(fx(i)-fx(i-1))*(x(i)-x(i-1)) + x(i-1)
            EXIT
         end if
      end do

   end function inverse_lin_interp

   pure function find_roots_cubic(coeff, lower_bound, upper_bound)
      real(DBprec), intent(in) :: lower_bound, upper_bound
      real(DBprec), intent(in) :: coeff(4)
      real(DBprec) :: find_roots_cubic
      real(DBprec) :: Q, R, theta, x, Au, Bu
      real(DBprec) :: a, b, c
      integer :: i, iter

      a = coeff(3)/coeff(4)
      b = coeff(2)/coeff(4)
      c = coeff(1)/coeff(4)

      Q = (a**2 - 3.D0*b)/9.D0
      R = (2.D0*a**3 - 9.D0*a*b + 27.D0*c)/54.D0

      !If roots are all real, get root in range
      if (R**2 .lt. Q**3) then
         iter = 0
         theta = acos(R/sqrt(Q**3))
         do i=-1,1
            iter = iter+1
            x = -2.D0*sqrt(Q)*cos((theta + dble(i)*PI*2.D0)/3.D0)-a/3.D0
            if ((x.ge.lower_bound).and.(x.le.upper_bound)) then
               find_roots_cubic = x
               return
            end if
         end do
         !Otherwise, two imaginary roots and one real root, return real root
      else
         Au = -sign(1.D0, R)*(abs(R)+sqrt(R**2-Q**3))**(1.D0/3.D0)
         if (Au.eq.0.D0) then
            Bu = 0.D0
         else
            Bu = Q/Au
         end if
         find_roots_cubic = (Au+Bu)-a/3.D0
         return
      end if

   end function

   function rtsafe(funcd, lower_bound, upper_bound, root_accuracy)
      ! Finds the root of a cubic equation bounded between lower and upper bounds to
      ! within +- root_accuracy. Modelled on rtsafe on page 1190 of numerical recipes in Fortran 90

      Real(DBprec), intent(in) :: lower_bound, upper_bound, root_accuracy
      real(DBprec) :: rtsafe
      interface
         subroutine funcd(x,fval,fderiv)
            ! some 1D function returning function value and derivative at x
            use Global_parameters_variables_and_types
            implicit none
            Real (DBprec), intent(in) :: x
            Real (DBprec), intent(out) :: fval, fderiv
         end subroutine
      end interface
      Integer(k4b), parameter :: MAXIT = 10000      !maximum iterations
      ! following parameters are directly copied from numerical recipes, to follow them

      Integer(k4b) :: j
      Real(DBprec) :: x1, x2, xacc
      Real(DBprec) :: df, dx, dxold, f, fh, fl, temp, xh, xl

      x1 = lower_bound
      x2 = upper_bound
      xacc = root_accuracy

      call funcd(x1,fl,df)
      call funcd(x2,fh,df)

      if ((fl > 0.0d0 .and. fh > 0.0d0) .or. &
         (fl < 0.0d0 .and. fh < 0.0d0)) then
         print *, "root must be bracketed in rtsafe"
         stop
      end if

      if (fl == 0.d0) then
         rtsafe = x1
         return
      else if (fh == 0.d0) then
         rtsafe = x2
         return
         ! make sure that f(x1) < 0
      else if (fl < 0.d0) then
         xl = x1
         xh = x2
      else
         xh = x1
         xl = x2
      end if
      rtsafe = 0.5d0*(x1+x2)
      !stepsize before last
      dxold = abs(x2-x1)
      ! stepsize of last step
      dx = dxold
      call funcd(rtsafe, f, df)
      do j=1,MAXIT
         if (((rtsafe-xh)*df-f)*((rtsafe-xl)*df-f) >= 0.d0 .or. &
            abs(2.0d0*f) > abs(dxold*df) ) then
            !bisection
            dxold = dx
            dx = 0.5d0*(xh - xl)
            rtsafe = xl + dx
            if (xl == rtsafe) return
         else
            dxold = dx
            dx = f/df
            temp = rtsafe
            rtsafe = rtsafe - dx
            if (temp == rtsafe) return
         end if
         if (abs(dx) < xacc) return
         call funcd(rtsafe,f,df)
         if (f < 0.d0) then
            xl = rtsafe
         else
            xh = rtsafe
         end if
      end do

      print *, "Max iterations exceeded in find_cubic_root_safe", dx
      stop
   end function

   Subroutine chbyshv (L, d_a, d_b, a)
      Integer L
      Real (DBprec) d_a, d_b, a(0:MAXCHEB)
      !---------------------------------------------------------------------c
      !     This routine calculates the Chebyshev coefficients for the      c
      !     Chebyshev polynomial approximation of a square root function.   c
      !     The routine computes L+1 coefficients, which are returned in    c
      !     in the one-dimensional array a, for the approximation of the    c
      !     square root of any variable that lies in the interval           c
      !     (d_a, d_b).                                                     c
      !---------------------------------------------------------------------c


      Integer j, k
      Real (DBprec) xks(0:L)


      !     Calculate the shift factors
      !	d_a = 2/(lmax-lmin)
      !      -d_b/d_a = (lmax+lmin)/2

      !     Calculate the collocation points
      Do k = 0,L
         xks(k) = Cos(PI*(k+0.5d0)/(L+1))/d_a -d_b/d_a
      End Do

      !     Calculate the Chebyshev coefficients
      a = 0.d0
      Do j = 0,L
         Do k = 0,L
            a(j) = a(j)+(xks(k)**0.5)*Cos(j*(k+0.5)*PI/(L+1))
         End Do
         a(j) = 2.d0/(L+1)*a(j)
      End Do
      a(0) = a(0)/2.d0

   End Subroutine chbyshv

   Subroutine maxminev_fi(n, A, maxev, minev)
      Use Global_parameters_variables_and_types
      Integer n
      Real (DBprec) A(:,:,:,:), maxev, minev
      !---------------------------------------------------------------------c
      !     This routine approximates the maximum and minimum eigen values  c
      !     of a symmetric matrix nxn matrix A using Fixman's suggestion.   c
      !---------------------------------------------------------------------c
      Integer lda, incx, incy, i
      Real (DBprec) F(n+1), G(n), alpha, beta
      !Real (SNGL) sdot
      Real (DOBL) ddot

      external :: dsymv

      F = 1.d0
      G = 0.d0

      lda = n
      alpha = 1.d0
      beta = 0.d0
      incx = 1
      incy = 1

      Call dsymv('U',n,alpha,A,lda,F,incx,beta,G,incy)
      maxev = ddot(n, F, 1, G, 1)
      maxev = 2.d0*maxev/n

      !Forall (i = 1:n) F(i) = (-1.0)**i
      Do i = 1,n,2
         F(i)   = -1.d0
         F(i+1) =  1.d0
      End Do

      Call dsymv('U',n,alpha,A,lda,F,incx,beta,G,incy)
      minev = ddot(n, F, 1, G, 1)
      minev = minev/2.d0/n
      ! write (*,*) maxev, minev

   End Subroutine maxminev_fi

   Subroutine polish_poly_root(c,xin,atol)
      Use Global_parameters_variables_and_types
      ! use newton raphson to polish the root of a polynomial
      Integer, Parameter :: n=4 ! presently only for cubic
      Real (dbprec), Intent (in) :: c(n)
      Real (dbprec), Intent (inout) :: xin
      Real (dbprec), Intent (in) :: atol

      Real (dbprec) ::  p, p1,x0,x

      Integer i,iter
      Integer, Parameter :: IMAX = 30

      x = xin

      ! algo from NR: techniques for polishing, roots of polynomials
      Do iter=1,IMAX
         x0 = x
         p =  c(n)*x + c(n-1)
         p1 = c(n)
         Do i=n-2,1,-1
            p1 = p + p1*x
            p  = c(i) + p*x
         End Do

         !if (abs(p1) < atol) then ! f' = 0 is not solvable in newt-raph
         If (Abs(p1) .Eq. 0.d0) Then ! f' = 0 is not solvable in newt-raph
            !! for WLC, in this case f'' also = 0, therefore
            p1 = 6 * c(4) ! f'''
            p1 = 6*p/p1
            ! note sign(a,b) returns sgn(b) * mod(a)
            x = x - Sign(1._DBprec,p1) * Abs(p1)**(1.d0/3.d0)
            !! we omit considering cases for ILC and FENE, as they
            !! donot seem to have this singularity
         Else
            x = x - p/p1
         End If

         If (Abs(p) .Le. atol) Exit
      End Do
      if (iter>IMAX) then
         !write (5,*) 'poly-root: loop exceeded'
      end if


      xin = x
   End Subroutine polish_poly_root

   Subroutine numint(f,dx,nmin,nmax,nord,sumf)
      Use Global_parameters_variables_and_types

      Real (DBprec), Intent(in), Dimension(:) :: f
      Real (DBprec) , Intent(in) :: dx
      Real (DBprec), Intent(out) :: sumf
      Integer, Intent (in) :: nmin, nmax,nord



      Integer, Parameter :: stdout=5, stderr=6
      Integer nbound,nint,j

      nbound = Ubound(f,1)
      nint = nmax-nmin

      !write (*,*) 'nbound = ', nbound


      If (nint > nbound .Or. nmin < 1 .Or. nmax > nbound ) Then
         Write(stderr,*) 'Array out of bounds: ', nmin,nmax,nbound
         Return
      End If

      sumf = 0.d0

      Select Case (nord)
       Case (1) ! trapezoidal rule
         sumf = 0.5d0 * (f(nmax) + f(nmin))

       Case(2)
         sumf = 5.d0/12.d0 * (f(nmin) + f(nmax)) + &
            13.d0/12.d0 *(f(nmin+1) + f(nmax-1))
       Case (3)
         sumf = 3.d0/8.d0 * (f(nmin) + f(nmax)) + &
            7.d0/6.d0 *(f(nmin+1) + f(nmax-1)) + &
            23.d0/24.d0 *(f(nmin+2) + f(nmax-2))
      End Select

      Do j = nmin+nord, nmax-nord
         sumf = sumf + f(j)
      End Do
      sumf =  dx*sumf

   End Subroutine numint

   subroutine meanerr(vec,mean,err)
      Use Global_parameters_variables_and_types
      real (DBprec) , intent(in) :: vec(:)
      real (DBprec) , intent(out) :: mean
      real (DBprec), intent(out) :: err ! standard error of mean

      integer n

      n = ubound(vec,1)
      mean = sum(vec)/n

      ! though the following is correct,
      ! err = sqrt((sum(vec*vec) - mean*mean*n)/(n-1))
      ! for very small errors, such as in the case of diffusivity.
      ! so use the sure shot formula.
      err = 2*sqrt(sum((vec-mean)*(vec-mean))/(n-1)/n)
      ! the standard error of mean is approximately in the 95% confidence
      ! interval, giving the factor 2 (from the t-distribution)


   end subroutine meanerr

   function normeqpr(Stype, r, b)
      Integer, intent (in) :: Stype
      Real (DBprec), intent (in) :: r, b
      Real (DBprec) normeqpr

      Integer, save :: norm_computed = 0
      Real (DBprec)  , save :: norm_b, bsav

      Integer , parameter :: Ln = 21 ! 21 always see below for X,W data
      real (DOBL) :: X(Ln), W(Ln)
      !real (DOBL) :: scratch(Ln)
      !real (DOBL)  endpts(2)

      Integer n
      real (DBprec)  xfact,rg, fb , M

      if ( norm_computed .eq. 0  .and. bsav .ne. b) then

         !! generate the gauss abscissa and weights for legendre
         !call gaussq(1, Ln, 0, 0, 0, endpts, scratch, X, W) ;


         !! generated from c code
         X(01) = -0.993752170620389;   W(01) = 0.016017228257774
         X(02) = -0.967226838566306;   W(02) = 0.03695378977085309
         X(03) = -0.920099334150401;   W(03) = 0.05713442542685689
         X(04) = -0.853363364583318;   W(04) = 0.0761001136283793
         X(05) = -0.768439963475678;   W(05) = 0.09344442345603385
         X(06) = -0.667138804197413;   W(06) = 0.1087972991671478
         X(07) = -0.55161883588722;   W(07) = 0.1218314160537286
         X(08) = -0.424342120207439;   W(08) = 0.1322689386333376
         X(09) = -0.288021316802401;   W(09) = 0.1398873947910733
         X(10) = -0.145561854160895;   W(10) = 0.14452440398997
         X(11) = -2.4782829604619e-16;   W(11) = 0.1460811336496907
         X(12) = 0.145561854160895;   W(12) = 0.1445244039899697
         X(13) = 0.288021316802401;   W(13) = 0.1398873947910732
         X(14) = 0.424342120207439;   W(14) = 0.1322689386333371
         X(15) = 0.55161883588722;   W(15) = 0.1218314160537288
         X(16) = 0.667138804197413;   W(16) = 0.1087972991671492
         X(17) = 0.768439963475678;   W(17) = 0.09344442345603389
         X(18) = 0.853363364583317;   W(18) = 0.07610011362837897
         X(19) = 0.920099334150401;   W(19) = 0.05713442542685797
         X(20) = 0.967226838566306;   W(20) = 0.03695378977085323
         X(21) = 0.993752170620389;   W(21) = 0.01601722825777395



         M = 18  ! some large number, obtained by trial and error

         ! limit the upper bound of M
         if (M > b/2) M = b/2

         xfact = sqrt(2*M/b)

         norm_b = 0.d0
         fb = 0.d0
         do n=1,Ln
            rg = (X(n) + 1)/2 * xfact
            norm_b = norm_b +  W(n) * expphi(Stype,rg,b) * rg * rg
            fb = fb +  W(n) * expphi(Stype,rg,b) * rg**4
         end do

         !write (*,*) '# b = ', b
         !write (*,*) '# computed normb = ', norm_b
         !write (*,*) '# f(b) = ', b*fb/norm_b

         bsav = b
         norm_computed = 1
      end if

      normeqpr = r*r*expphi(Stype,r,b)/norm_b

   end function normeqpr

   function expphi(Stype, r, b)
      !!! return the distribution function (without normalisation)
      Integer, intent (in) :: Stype
      Real (DBprec) , intent (in) :: r, b
      Real (DBprec) expphi

      select case (Stype)
       case (HOOK)
         write (*,*) 'Hookean case called in expb: Check the code'
       case (FENE)
         expphi = (1-r*r)**(b/2)

       case (WLC)
         expphi = exp( -b/6.0 * ( 2*r*r + 1/(1-r) -r -1 ));
      end select
   end function expphi

   function  lam1_th(hs, N)
      Real (DBprec) lam1_th
      Real (DBprec) , intent (in) :: hs
      Integer ,  intent (in) :: N


      real (DBprec) s,b,aj


      b   = 1 - 1.66*hs**0.78;
      s   =   - 1.40*hs**0.78;

      aj = 4*sin(Pi/2./N)*sin(Pi/2./N);
      lam1_th = 2./( aj * b / N**s);

   end function lam1_th

end module simulation_utilities


Module flock_utils
!****************************************************************!***********
! This module contains utilities to lock files to prevent other programs
! to open and write data into it.  This is done by creating another file in
! the directory called lock-<file>, where <file> is the file to be locked.
! The module contains routines for locking, unlocking (removing the lock-<file>
! and for returning the status of a lock through islocked()
! The unlocking is done by 'deleting' the lock-<file>
!****************************************************************!***********
   Implicit none

!--string prefix for the lock file
   Character(5), parameter :: prefx = 'lock-'
!--a constant number to be added to the unit of the file to be locked
   Integer, parameter :: added = 100

Contains

   Subroutine islocked(file,lstat)

      Character(*), intent(in) :: file
      Logical, intent(out) :: lstat

      Character(40) :: lfile

      lfile = prefx // file

      Inquire (file=lfile, exist=lstat)

   end Subroutine islocked

   Subroutine lockfile(unit,file)
      Integer, intent(in) :: unit
      Character(*), intent(in) :: file

      Integer lunit
      Character(40) :: lfile

      lfile = prefx // file

      lunit = unit + added
      open(unit=lunit,file=lfile)

   end Subroutine lockfile

   Subroutine unlockfile(unit,file)

      Integer, intent(in) :: unit
      Character(*), intent(in) :: file

      Integer lunit
      Character(40) :: lfile

      lfile = prefx // file

      lunit = unit + added
      close(unit=lunit,status='delete')


   end Subroutine unlockfile
end Module flock_utils
