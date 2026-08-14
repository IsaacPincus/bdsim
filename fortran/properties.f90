!!! Time-stamp: <properties.f90 09:12, 30 Jan 2004 by P Sunthar>



!!! $Log: properties.f90,v $
!!! Revision 1.1  2004/01/29 22:12:18  pxs565
!!! Initial revision
!!!

module properties
!________________________________________________________________
!   utilities to extract properties from a given configuration
!________________________________________________________________

   use Global_parameters_variables_and_types
   implicit none

   Character(10) :: Prop_names(NProps)

   Character(10) , Parameter :: Correl_names(1) = (/"Sxy"/)

   type calculated_variables
      Real (DBprec), allocatable, dimension(:,:) :: samples
      Real (DBprec), allocatable, dimension(:) :: true_times_at_sample_points
      Real (DBprec), allocatable, dimension(:,:,:) :: chain_configuration_at_sample_points
      Real (DBprec), allocatable, dimension(:,:,:) :: total_force_at_sample_points
      Real (DBprec), allocatable, dimension(:,:) :: center_of_mass_cumulative

   end type

   contains

   Subroutine chain_props(NBeads, Rbead, F, props)
     Integer, intent (in) :: NBeads
     Real (DBprec), intent (in), dimension(:,:)  :: Rbead, F
     Real (DBprec), intent (out),dimension(:) ::props

     !---------------------------------------------------------------------
     !    This routine estimates the contributions of a single bead-spring
     !    chain in the configuration specified by the vector R, to the
     !    equilibrium and non-equilibrium properties of the solution.
     !    (Time correlations are evaluated by a separate procedure.)
     !
     !    The idea behind using this routine is to have a code wherein
     !    the main program can be run using different property estimation
     !    routines (such as this) without any serious changes to the
     !    main program itself. In addition, the user should have the
     !    freedom to choose properties of interest.
     !
     !    In the present routine, the properties are returned in the
     !    vector props whose elements are given below:
     !
     !    props(1)  ---  square of end-to-end vector
     !    props(2)  ---  1,1 component of shape tensor
     !    props(3)  ---  1,2 component of shape tensor
     !    props(4)  ---  1,3 component of shape tensor
     !    props(5)  ---  2,2 component of shape tensor
     !    props(6)  ---  2,3 component of shape tensor
     !    props(7)  ---  3,3 component of shape tensor
     !    props(8)  ---  First normal stress difference
     !    props(9)  ---  Second normal stress difference
     !    props(10) ---  1,2 component of stress tensor
     !    props(11) ---  Stretch in 1 direction
     !    props(12) ---  stretch in 2 direction
     !    props(13) ---  stretch in 3 direction
     !    props(14) ---  radius of gyration
     !--------------------------------------------------------------------
     Integer i, j, k, mu, nu
     Real (DBprec) R(Ndim,NBeads), Rc(Ndim),dist(Ndim)

     props = 0.D0
     R = RBead

     !     Calculation of square of end-to-end vector

     dist = R(:,Nbeads) - R(:,1)

     props(1) = sum(dist*dist)


     !     Calculation of shape tensor

     !       First, calculate position of centre of mass
     Rc = 0.D0
     Do mu = 1, NBeads
        Rc = Rc + R(:,mu)
     End Do
     Rc = Rc/NBeads

     ! shift the origin to the current center of mass
     Forall (mu = 1: NBeads)
        R(:,mu) = R(:,mu) - Rc
     End Forall


     !       Calculate shape tensor
     ! props (2:7)
     Do nu = 1, NBeads
        k = 2
        Do i = 1,Ndim             ! Since the tensor is symmetric
           Do j = i,Ndim          ! calculate only the upper diagonal components
              props(k) = props(k) + R(i,nu) * R(j,nu)
              k = k + 1
           End Do
        End Do
     End Do
     props(2:7) = props(2:7)/NBeads

     !     Calculation of polymer's contribution to the stress tensor
     !     using the Kramers-Kirkwood expression - Eq. C in Table 15.2-1 of DPL - II

     ! props (8:10)
     Do nu = 1, NBeads
        k = 8
        props(k) = props(k) + R(1,nu) * F(1,nu) - R(2,nu) * F(2,nu)
        k = k + 1
        props(k) = props(k) + R(2,nu) * F(2,nu) - R(3,nu) * F(3,nu)
        k = k + 1
        props(k) = props(k) + R(1,nu) * F(2,nu)
     End Do

     ! props (11:13)
     !     Calculation of "stretch" in x, y and z directions
     Do i = 1,Ndim
        props(10+i) = Maxval(R(i,:)) - Minval(R(i,:))
     End Do

     ! prop (14)
     ! radius of gyration
     props(14) = 0.D0
     Do mu = 1, NBeads
        props(14) = props(14) + sum(R(:,mu)*R(:,mu))
     end Do
     props(14) =  props(14)/NBeads



   !!$        write (*,98) R,F
   !!$ 98     format (3(F10.6,1x))
   !!$        write (*,72) props(1:10)
   !!$72      format (5(E10.4,2x))

     !     ENSURE THAT THE NProps PARAMETER IN MODULE globals IS GREATER THAN
     !     THE NUMBER OF PROPERTIES DEFINED

   End Subroutine chain_props

   Subroutine time_correl(NBeads, R, F, t0, t, correl)
     Integer, intent (in) :: NBeads
     Real (DBprec), intent (in), dimension(:,:)  :: R, F
     Real (DBprec), intent (out)  :: correl
     Real (DBprec), intent (in) :: t0, t
     !Real R(3*NBeads), F(3*NBeads), correl(:), t0, t

     !--------------------------------------------------------------------c
     !    This routine estimates time correlations between properties     c
     !    at times t0 and t.                                              c
     !                                                                    c
     !    In the present routine, the correlations are returned in the    c
     !    vector correl whose elements are given below:                   c
     !                                                                    c
     !    correl     ---  the autocorrelation for calculation of          c
     !                    linear viscoelastic properties in simple shear  c
     !--------------------------------------------------------------------c
     Integer  nu, dir
     Real (DBprec) Rc(Ndim), temp
     Real (DBprec), Save :: t0props

     correl = 0.D0

     !     First, calculate position of centre of mass
     Rc = 0.D0
     Do nu = 1, NBeads
        Rc = Rc + R(:,nu)
     End Do
     Rc = Rc/NBeads

     !     Calculation of Sxy
     temp = 0.D0
     dir = 1
     Do nu = 1,NBeads
        temp = temp + (R(dir,nu)-Rc(dir)) * F(dir+1,nu)
     End Do

     If (t.Eq.t0) t0props = temp

     correl = temp*t0props


     !     ENSURE THAT THE NCorrels PARAMETER IN MODULE globals IS GREATER THAN
     !     OR EQUAL TO THE NUMBER OF PROPERTIES DEFINED

   End Subroutine time_correl

end module properties
