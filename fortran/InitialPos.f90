! A module containing some utilities for generating the initial positions
! of the chains

module Initial_Position_Utilities
   use Global_parameters_variables_and_types
   use random_numbers
   use simulation_utilities
   Implicit none

   private

   public Initial_position
   public Initial_position_FENE_Fraenkel
   public FENE_Fraenkel_Aligned_x_axis

contains

   Subroutine Initial_position(SType,Nbeads,L0s,Q0s,R)

      Integer, Intent (in) :: Stype
      Integer(k4b), Intent (in) :: Nbeads
      Real (DBprec), intent (in) :: L0s, Q0s ! L0s is sqrtb, Q0s is natscl
      Real (DBprec), Intent (out), Dimension(Ndim,Nbeads) :: R

      Integer mu, done, dummy
      Real (DBprec), parameter :: upbound = 2.d0
      Real (DBprec) modr,b2b(Ndim),rv,eqpr, rvchk
      Real (DBprec) b, b2b_length

      ! Separate procedure for bounded WLC
!      if (SType.eq.WLC_bounded) then
!         R = 0.d0
!         do mu=2,Nbeads
!            b2b_length = -5.d0
!            if (Q0s>L0s) then
!               print *, "Initial position fails, Q0s>L0s"
!               stop
!            end if
!            do while ((b2b_length<max(0.d0,(2.d0*Q0s-L0s))).or.(b2b_length>L0s))
!               ! Gaussian random number with mean Q0s and std (L0s-Q0s)
!               call GaussRand(Ndim,b2b)
!               b2b = b2b*(L0s-Q0s)+Q0s
!               b2b_length = sqrt(b2b(1)**2 + b2b(2)**2 + b2b(3)**2)
!            end do
!            R(:,mu) = R(:,mu-1) + b2b
!         end do
!      end if

      Call GaussRand(Ndim*Nbeads,R)

      !print *, "R init", R

      R(:,1) = 0.D0

      ! intrinsic ran() modifies the seed for every call
      ! so use a separate varible seed, so that the ran_1 sequence is
      ! not altered by this variation.
      !varseed = seed

      b = L0s*L0s

      !!$  do rv = 0,0.99,.01
      !!$     eqpr = normeqpr(Stype,rv,b);
      !!$     write (*,*) rv,eqpr
      !!$  end do

      Do mu = 2,Nbeads
        b2b = R(:,mu) ! * sqrt(sigma=1), gaussian dist betwn two adj beads

        !!!!!!!!!! Change Spring type
        ! if (SType .ne. HOOK) then
        If (Stype .eq. Hook) then
            dummy = 1
        Else If (Stype .eq. Fraenkel) then
            dummy = 1
        End If
        if (dummy .ne. 1) then
           done = 0;

           do while (done .ne. 1)

             ! rv = ran(varseed)
              call Random_Number(rv)
              eqpr = normeqpr(Stype,rv,b);


              ! use another uniform random number to decide acceptance
              ! ie, accept only eqpr fraction at this r
              !rvchk = ran(varseed)

               call Random_Number(rvchk)
              !write (*,*) rv, rvchk , eqpr
              if (rvchk .le. eqpr/upbound) done = 1
           end do

           modr = Sqrt(Sum(b2b*b2b))
           b2b = b2b/modr * (rv * L0s)
        end if

        R(:,mu) = R(:,mu-1) + b2b

        !print *, "R middle", R

        ! in the case of a finitely extensible spring, a gaussian distribution
        ! can lead to incorrect forces some exceptional b2b vectors, we limit
        ! this by altering the distance when preseving the random direction.
        ! and an additional random factor between 0 and 1
        ! A good guess of the factor cud b obtained more rigorously.
      End Do

      !print *, "R final", R
   End Subroutine Initial_position

   Subroutine Initial_position_FENE_Fraenkel(SType,Nbeads,dQ,sigma,R)
      ! Subroutine to generate the initial position of a chain to conform with the distribution
      ! function of a FENE-Fraenkel dumbbell at equilibrium.
      ! Connector vector directions are chosen randomly on a unit sphere.
      ! This can be used even without a FENE-Fraenkel spring, such that
      ! sigma is natural length, dQ is extensiblity about that length

      Integer, Intent(in) :: SType                    ! Spring type
      Integer(k4b), Intent(in) :: Nbeads                   ! Number of beads in chain
      Real(DBprec), Intent(in) :: dQ                  ! Finite extensibility about sigma, equivalent to sqrtb
      Real(DBprec), Intent(in) :: sigma               ! Natural spring length, equivalent to Q0 for Fraenkel spring
      Real(DBprec), Intent(out), Dimension(Ndim,Nbeads) :: R  ! Output bead locations in chain

      Integer :: nu
      Real(DBprec), dimension(Ndim, Nbeads-1) :: b2bvecs         ! bead to bead vectors generated

      b2bvecs = generate_Q_FF(sigma, dQ**2.d0, 10000, Nbeads-1)

      R(:,1) = 0.d0

      do nu = 2,Nbeads
         R(:, nu) = R(:,nu-1) + b2bvecs(:,nu-1)
      end do

   End Subroutine

   subroutine FENE_Fraenkel_Aligned_x_axis(SType,Nbeads,dQ,sigma,R)
      ! Subroutine to generate the initial position of a chain to conform with the distribution
      ! function of a FENE-Fraenkel dumbbell at equilibrium.
      ! Connector vector directions are aligned along x-axis
      ! This can be used even without a FENE-Fraenkel spring, such that
      ! sigma is natural length, dQ is extensiblity about that length

      Integer, Intent(in) :: SType                    ! Spring type
      Integer(k4b), Intent(in) :: Nbeads                   ! Number of beads in chain
      Real(DBprec), Intent(in) :: dQ                  ! Finite extensibility about sigma, equivalent to sqrtb
      Real(DBprec), Intent(in) :: sigma               ! Natural spring length, equivalent to Q0 for Fraenkel spring
      Real(DBprec), Intent(out), Dimension(Ndim,Nbeads) :: R  ! Output bead locations in chain
      real(DBprec) :: Ql(Nbeads-1)                           ! Lengths of FENE-Fraenkel dumbbells

      Integer :: nu, i
      Real(DBprec), dimension(Ndim, Nbeads-1) :: b2bvecs         ! bead to bead vectors generated

      Ql(:) = generate_Ql_eq_FF(Nbeads-1, dQ**2.d0, sigma, 10000)

      do i=1,NBeads-1
         b2bvecs(:,i) = (/1,0,0/)*Ql(i)
      end do

      R(:,1) = 0.d0

      do nu = 2,Nbeads
         R(:, nu) = R(:,nu-1) + b2bvecs(:,nu-1)
      end do
   end subroutine

   subroutine spherical_unit_vectors(output_unit_vector, N)
      ! Outputs a spherically symmetrically distributed unit vector

      integer(k4b), intent(in) :: N
      integer :: i
      real(DBprec) :: x1, x2, r1(N), r2(N)
      real(DBprec), dimension(Ndim, N) :: output_unit_vector

      do i=1,N
         do while (.True.)
            call ran_1(1, r1)
            call ran_1(1, r2)
            x1 = r1(1)*2.D0 - 1.D0
            x2 = r2(1)*2.D0 - 1.D0
            if ((x1**2 + x2**2).lt.1.D0) EXIT
         end do
         output_unit_vector(1,i) = 2.D0*x1*sqrt(1.D0-x1**2-x2**2)
         output_unit_vector(2,i) = 2.D0*x2*sqrt(1.D0-x1**2-x2**2)
         output_unit_vector(3,i) = 1.D0-2.D0*(x1**2 + x2**2)
      end do

   end subroutine

   function psiQ_FF(Q, alpha, Q0, Jeq)
      implicit none
      real(DBprec), intent(in) :: Q, alpha, Q0, Jeq
      real(DBprec) :: psiQ_FF

      !    Jeq = (1.D0/(alpha+3.D0)+Q0**2/alpha)*beta(0.5D0,(alpha+2.D0)/2.D0)*alpha**(1.5D0)

      psiQ_FF = Q**2*(1.D0-(Q-Q0)**2.D0/alpha)**(alpha/2.D0)/Jeq
   end function psiQ_FF

   function integral_psiQ_FF(Q, alpha, Q0, Jeq)
      !Simple cumulative trapezoidal integral of psiQ_FF at
      !points specified in Q
      implicit none
      real(DBprec), dimension(:), intent(in) :: Q
      real(DBprec), intent(in) :: alpha, Q0, Jeq
      real(DBprec), dimension(size(Q)) :: integral_psiQ_FF
      integer :: k

      integral_psiQ_FF(1) = 0.D0
      do k=2,size(Q)
         integral_psiQ_FF(k) = integral_psiQ_FF(k-1) + &
            (psiQ_FF(Q(k-1),alpha,Q0, Jeq) + psiQ_FF(Q(k),alpha,Q0, Jeq))*(Q(k)-Q(k-1))/2.D0
      end do
      !Trapezoidal rule is far from perfect, but we must have int from 0 to 1
      !    integral_psiQ_FF = integral_psiQ_FF/integral_psiQ_FF(size(Q))

   end function integral_psiQ_FF

   function generate_Ql_eq_FF(N, alpha, Q0, Nsteps)
      implicit none
      real(DBprec), intent(in) :: alpha, Q0
      ! N is number of Qls to generate
      ! Nsteps is the number of widths to use for trapezoidal rule integration
      integer(k4b), intent(in) :: N, Nsteps
      integer :: k
      real(DBprec), dimension(N) :: generate_Ql_eq_FF, rands
      real(DBprec), dimension(Nsteps) :: Q, intpsiQ
      real(DBprec) :: width, Jeq

      !Generate a Q vector between Q0-sqrt(alpha) and Q0+sqrt(alpha) with Nsteps steps
      !Round off a little at the end to prevent singularities
      Q = 0.D0
      if (Q0-sqrt(alpha).le.0.D0) then
         width = (Q0+sqrt(alpha))/(Nsteps-1)
         Q(1) = 0.D0
      else
         width = 2.D0*sqrt(alpha)/(Nsteps-1)
         Q(1) = Q0-sqrt(alpha)
      end if
      do k=2,Nsteps
         Q(k) = Q(k-1) + width
      end do
      Q(1) = Q(1) + 0.0000001D0
      Q(Nsteps) = Q(Nsteps) - 0.0000001D0

      !   determine normalisation constant
      !   Is this needed? It's normalised anyway... TODO
      intpsiQ = integral_psiQ_FF(Q,alpha,Q0, 1.D0)
      Jeq = intpsiQ(size(Q))
      !   perform actual integration and normalise to 1
      intpsiQ = integral_psiQ_FF(Q,alpha,Q0,Jeq)
      Jeq = intpsiQ(size(Q))
      intpsiQ = intpsiQ/Jeq

      call ran_1(N,rands)

      !Inverse of integral returns distribution of psiQ_FF
      do k=1,N
         generate_Ql_eq_FF(k) = inverse_lin_interp(Q,intpsiQ,rands(k))
      end do

   end function generate_Ql_eq_FF

   function generate_Q_FF(Q0,alpha, Nsteps, N)
      implicit none
      real(DBprec), intent(in) :: Q0, alpha
      integer(k4b), intent(in) :: N, Nsteps
      real(DBprec), dimension(Ndim, N) :: generate_Q_FF
      real(DBprec) :: Ql(N)

      call spherical_unit_vectors(generate_Q_FF, N)

      Ql(:) = generate_Ql_eq_FF(N, alpha, Q0, Nsteps)

      generate_Q_FF(1,:) = generate_Q_FF(1,:)*Ql
      generate_Q_FF(2,:) = generate_Q_FF(2,:)*Ql
      generate_Q_FF(3,:) = generate_Q_FF(3,:)*Ql

   end function generate_Q_FF

end module
