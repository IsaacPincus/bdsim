!!! Time-stamp: <gsipc.f90 12:45, 06 Apr 2004 by P Sunthar>

!________________________________________________
!   Bead Spring Configuration Space Utilities
!________________________________________________

!!! $Log: gsipc.f90,v $
!!! Revision 1.3  2004/02/09 05:38:36  pxs565
!!! pcerr made relative
!!!
!!! Revision 1.2  2004/02/03 02:02:35  pxs565
!!! cofm initialised correctly
!!!
!!! Revision 1.1  2004/01/29 22:11:08  pxs565
!!! Initial revision
!!!


Module csputls
!!! This module contains subroutines used to manipulate the
!!! configuration variable, R and its derivatives
   Use Global_parameters_variables_and_types
   Implicit none

Contains

   Subroutine b2bvector_sym_up(N,R,b2b)
      ! Calling arguments
      Integer, Intent(in)  :: N
      Real (DBprec), Intent(in)  :: R(:,:)
      Real (DBprec), Intent(out) :: b2b(:,:,:)

!!!_________________________________________________________|
!!! This routine returns the vector one bead to another bead
!!! as a symmetric matrix but only the upper elements filled
!!!_________________________________________________________|

      Integer mu,nu

      ! according to /usr/bin/prof, this takes max time. so commenting out
      ! since only upper diagonal is anyway reqd
      ! b2b = 0

      ! note only upper half (nu > mu) of the matrix is filled
      ! and the convention is R_12 = R_2 - R_1
      ! where R_12 is the vector from 1 to 2
      Do nu = 2,N
         Do mu = 1,nu-1
            b2b(:,mu,nu) = R(:,nu) - R(:,mu)
         End Do
      End Do
   End Subroutine b2bvector_sym_up

   Subroutine modr_sym_up(N,b2bvec,deltaR)
      ! calling arguments
      Integer, Intent (in)  :: N
      Real (DBprec), Intent (in)  :: b2bvec(:,:,:)
      Real (DBprec), Intent (out) :: deltaR(:,:)
!!!_________________________________________________________|
!!! This subroutine returns the magnitude of each of the bead
!!! to bead vector
!!!_________________________________________________________|

      Integer mu,nu,i
      Real (DBprec) r12(Ndim),modr


      deltaR = 0.d0

      ! note that we cant use a forall since dot_product is a
      ! transformational function
      Do nu = 2,N
         Do mu = 1,nu-1
            r12 = b2bvec(:,mu,nu)
            modr = 0.d0
            Do i=1,Ndim
               modr = modr + r12(i) * r12(i)
            End Do
            If (modr < 1.0d-12) modr = 1.0d-12   !so that initial dr is != 0
            deltaR(mu,nu) = Sqrt(modr)
         End Do
      End Do
   End Subroutine modr_sym_up

   subroutine Q_mag_from_R_vector(N, R, Q_mag)
      ! N is the length of the R vector, or number of beads
      Integer, intent(in) :: N
      Real(DBprec), intent(in) :: R(:,:)
      Real(DBprec), intent(out) :: Q_mag(N-1)
      Real(DBprec) :: Q_tmp(3)

      Integer nu

      Do nu = 1,N-1
         Q_tmp = R(:,nu+1) - R(:,nu)
         Q_mag(nu) = sqrt(Q_tmp(1)**2 + Q_tmp(2)**2 + Q_tmp(3)**2)
      End Do

   end subroutine

   subroutine get_unit_vectors_from_connector_vectors_and_distances(unit_vectors,&
      bead_to_bead_vector_superdiagonal_matrix, bead_to_bead_distances, Nbeads)
      ! Calculates the unit vectors pointing between beads given the bead distances and connecting vectors

      Integer, intent(in) :: Nbeads
      Real (DBprec), Intent (in)  :: bead_to_bead_vector_superdiagonal_matrix(:,:,:) ! Ndim x Nbeads x Nbeads
      Real (DBprec), Intent (in)  :: bead_to_bead_distances(:,:) ! Nbeads x Nbeads
      Real (DBprec), Intent (out)  :: unit_vectors(:,:) ! Ndim x (Nbeads - 1)

      Integer mu

      do mu = 1,Nbeads-1
         unit_vectors(:,mu) = bead_to_bead_vector_superdiagonal_matrix(:,mu,mu+1)/bead_to_bead_distances(mu,mu+1)
      end do

   end subroutine

   subroutine get_cos_theta_from_unit_vectors(cos_theta_values, unit_vectors, Nbeads)

      Integer, Intent(in) :: NBeads
      Real (DBprec), Intent(in)  :: unit_vectors(:,:) ! Ndim x (Nbeads - 1)
      Real (DBprec), Intent(out)  :: cos_theta_values(:) ! (Nbeads - 2)

      cos_theta_values = sum(unit_vectors(:,2:)*unit_vectors(:,1:Nbeads-2),1)

   end subroutine


   Subroutine tensor_prod_of_vec(alpha,vec,tensor)
      Implicit None
      Real (DBprec), Intent (in) :: alpha
      Real (DBprec), Intent (in) :: vec(:)
      Real (DBprec), Intent (out) ::  tensor(:,:)
      Integer i,j
      Real (DBprec) temp

      Do j = 1,Ndim
         Do i = 1,j
            temp = alpha * vec(i) * vec(j)
            tensor(i,j) = temp
            tensor(j,i) = temp
         End Do
      End Do
   End Subroutine tensor_prod_of_vec
End Module csputls

Module Excluded_Volume_Calculations
   use csputls
   use simulation_utilities
   use Global_parameters_variables_and_types
   Implicit none

   private

   public get_excluded_volume_force
   public set_EV_parameters

   type, public :: Excluded_Volume_Inputs
      Integer :: excluded_volume_type = noEV
      Real (DBprec) :: dimensionless_EV_energy = 0.d0
      Real (DBprec) :: dimensionless_EV_radius = 0.d0
      Real (DBprec) :: minimum_EV_cutoff = 0.7d0
      Real (DBprec) :: maximum_EV_cutoff = 1.5d0
      Real (DBprec), allocatable, dimension(:,:) :: bead_attractive_interaction_strengths
      Real (DBprec) :: backbone_attractive_strength = 0.d0
      Integer :: spacer_length = 5
      Integer :: contour_dist_for_EV = 1
   end type

   ! internal paramters for EV
   Real (DBprec) :: zsbyds5,pt5bydssq
   Real (DBprec) :: LJa,LJb,LJpa,LJpb,Rmin,Rcutg,Rcutp
   Real (DBprec) :: SDKa, SDKb, RcutpSDK  ! for SDK
   Real (DBprec) :: phi_bb
   Real (DBprec), parameter :: alpha=1.530633312d0, beta=1.213115524d0
   Real (DBprec), allocatable, dimension(:,:) :: phi
   logical, save :: phi_allocated = .false.
   Integer :: EV, l, cd

contains

   subroutine set_EV_parameters(nbeads,EV_inputs)
      ! Sets internal EV parameters to calculate force
      ! Should be called before actually calculating the force

      Integer, intent(in) :: Nbeads
      type(Excluded_Volume_Inputs), intent(in) :: EV_inputs
      Real (DBprec) :: Zstar,Dstar   ! EV parameters (non-dim)

      zstar = EV_inputs%dimensionless_EV_energy
      dstar = EV_inputs%dimensionless_EV_radius
      EV = EV_inputs%excluded_volume_type

      ! Set phi to input if SDK, else to 0
      if (EV_inputs%excluded_volume_type.eq.SDK) then
         phi = EV_inputs%bead_attractive_interaction_strengths
      else
         if (phi_allocated) then
            deallocate(phi)
         else
            phi_allocated = .true.
         end if
         allocate(phi(nbeads,nbeads))
         phi = 0.d0
      end if

      phi_bb = EV_inputs%backbone_attractive_strength
      l = EV_inputs%spacer_length
      cd = EV_inputs%contour_dist_for_EV

      ! For EV, Gauss
      zsbyds5 = Zstar/(Dstar**5.0)
      pt5bydssq = 0.5/(Dstar*Dstar)

      ! for EV, LJ
      LJpa = 12.D0  !LJ purely repulsive interaction potential power.
      LJpb = 6.D0  !LJ purely attractive interaction potential power.
      LJa  = LJpa*4*Zstar*(Dstar**LJpa)
      LJb  = LJpb*4*Zstar*(Dstar**LJpb)

      ! for EV, SDK
      SDKa  = LJpa*4*(Dstar**LJpa)
      SDKb  = LJpb*4*(Dstar**LJpb)
      RcutpSDK = EV_inputs%maximum_EV_cutoff

      Rmin = EV_inputs%minimum_EV_cutoff
      !Rcutg = EV_inputs%maximum_EV_cutoff
      Rcutp = EV_inputs%maximum_EV_cutoff
!      RcutpSDK = 1.82D0*Dstar
!
!      Rmin = 0.7D0*Dstar
      Rcutg = Dstar*(2.D0**(1.D0/6.D0))
!      Rcutp = 1.5D0*Dstar

   end subroutine

   Subroutine get_excluded_volume_force(N,b2bvec,dr,Fev,FlowType)
!!! This routine returns excluded volume force
      Integer, intent (in) :: N
      Integer, intent (in) :: FlowType
      Real (DBprec), Intent (in) :: b2bvec(:,:,:)
      Real (DBprec), Intent (in) :: dr(:,:)
!      Integer, Intent (in) :: EV
      Real (DBprec), Intent (out) :: Fev(:,:)     ! Ndim x Nbeads
!      Real (DBprec), Intent (in) :: phi(:,:)

      Integer nu,mu, inteq
      Real (DBprec) :: Fpair12(Ndim)
      Integer :: stickbead(N)
      !Real (DBprec) :: modr, r12(Ndim)

      stickbead = 0
      Fev  = 0.d0
      inteq = 1
      Select Case (FlowType)
       case (EQ) ! Equilibrium without attractive potential in LJ and SDK
         inteq  = 1 !Initial configuration in good solvent
       case (UA) ! uniaxial extension
         inteq  = 0 !Initial configuration in poor solvent
       case (PL) ! planar extension
         inteq  = 0 !Initial configuration in poor solvent
       case (SH) ! planar shear
         inteq  = 0 !Initial configuration in poor solvent
       case (UR) ! uniaxial extension followed by relaxation
         inteq  = 0 !Initial configuration in poor solvent
       case (PU) ! periodic uniaxial extension
         inteq  = 0 !Initial configuration in poor solvent
       case (PP) ! periodic planar extension
         inteq  = 0 !Initial configuration in poor solvent
       case (PR)  ! Production run without flow
         inteq = 0
      end Select

      If (EV .EQ. Gauss) Then !For Gaussian potential
         Do nu = 2,N
            Do mu  = 1,nu-1
               ! force between the pair mu,nu, since the EV force is
               ! repulsive, we take it to be in the opposite direction
               ! of the connector vector mu -> nu
               ! convention followed: force is positive for attraction
               if ((nu-mu) .lt. cd) then
                  continue
               else
                  Fpair12 = - b2bvec(:,mu,nu) &
                     * zsbyds5*Exp(-dr(mu,nu)*dr(mu,nu) * pt5bydssq)
                  Fev(:,mu) = Fev(:,mu) + Fpair12
                  Fev(:,nu) = Fev(:,nu) - Fpair12
               end if
            End Do
         End Do

      Else if (EV .EQ. LJ) Then  !For LJ
         If (inteq .eq. 1) Then
            Do nu = 2,N
               Do mu  = 1,nu-1
                  if ((nu-mu) .lt. cd) then
                     continue
                  else
                     If (dr(mu,nu) .le. Rcutg .and. dr(mu,nu) .ge. Rmin) Then
                        Fpair12 = - b2bvec(:,mu,nu) &
                           *(LJa/(dr(mu,nu)**(LJpa+2.d0))-LJb/(dr(mu,nu)**(LJpb+2.d0)))
                        Fev(:,mu) = Fev(:,mu) + Fpair12
                        Fev(:,nu) = Fev(:,nu) - Fpair12
                     Else If (dr(mu,nu) .lt. Rmin) Then
                        Fpair12 = - b2bvec(:,mu,nu) &
                           *(LJa/(Rmin**(LJpa+2.d0))-LJb/(Rmin**(LJpb+2.d0)))
                        Fev(:,mu) = Fev(:,mu) + Fpair12
                        Fev(:,nu) = Fev(:,nu) - Fpair12
                     End If
                  end if
               End Do
            End Do
         Else
            Do nu = 2,N
               Do mu  = 1,nu-1
                  if ((nu-mu) .lt. cd) then
                     continue
                  else
                     If (dr(mu,nu) .le. Rcutp .and. dr(mu,nu) .ge. Rmin) Then
                        Fpair12 = - b2bvec(:,mu,nu) &
                           *(LJa/(dr(mu,nu)**(LJpa+2.d0))-LJb/(dr(mu,nu)**(LJpb+2.d0)))
                        Fev(:,mu) = Fev(:,mu) + Fpair12
                        Fev(:,nu) = Fev(:,nu) - Fpair12
                     Else If (dr(mu,nu) .lt. Rmin) Then
                        Fpair12 = - b2bvec(:,mu,nu) &
                           *(LJa/(Rmin**(LJpa+2.d0))-LJb/(Rmin**(LJpb+2.d0)))
                        Fev(:,mu) = Fev(:,mu) + Fpair12
                        Fev(:,nu) = Fev(:,nu) - Fpair12
                     End If
                  End If
               End Do
            End Do
         End If

      Else if (EV .EQ. SDK) Then  !For SDK
         Do nu = 2,N
            Do mu  = 1,nu-1
               if ((nu-mu) .lt. cd) then
                  continue
               else
                  If (dr(mu,nu) .le. Rcutg .and. dr(mu,nu) .ge. Rmin) Then
                     Fpair12 = - b2bvec(:,mu,nu) &
                        *(SDKa/(dr(mu,nu)**(LJpa+2.d0))-SDKb/(dr(mu,nu)**(LJpb+2.d0)))
                     Fev(:,mu) = Fev(:,mu) + Fpair12
                     Fev(:,nu) = Fev(:,nu) - Fpair12

                  Else If (dr(mu,nu) .lt. Rmin) Then
                     Fpair12 = - b2bvec(:,mu,nu) &
                        *(SDKa/(Rmin**(LJpa+2.d0))-SDKb/(Rmin**(LJpb+2.d0)))
                     Fev(:,mu) = Fev(:,mu) + Fpair12
                     Fev(:,nu) = Fev(:,nu) - Fpair12
                  End IF
                  If (inteq .eq. 0) Then
                     If (dr(mu,nu) .le. RcutpSDK .and. dr(mu,nu) .gt. Rcutg) Then
                        Fpair12= -b2bvec(:,mu,nu) &
                           *alpha*phi(mu,nu)*sin(alpha*dr(mu,nu)*dr(mu,nu) + beta)
                        Fev(:,mu) = Fev(:,mu) + Fpair12
                        Fev(:,nu) = Fev(:,nu) - Fpair12
                     End If
                  End If ! inteq
               End If
            End Do
         End Do

      Else if (EV .EQ. SDK_stickers) Then !For SDK with stickers
         Do nu = 2,N
            Do mu  = 1,nu-1
               if ((nu-mu) .lt. cd) then
                  continue
               else
                  If (dr(mu,nu) .le. Rcutg .and. dr(mu,nu) .ge. Rmin) Then
                     Fpair12 = - b2bvec(:,mu,nu) &
                        *(SDKa/(dr(mu,nu)**(LJpa+2.d0))-SDKb/(dr(mu,nu)**(LJpb+2.d0)))
                     Fev(:,mu) = Fev(:,mu) + Fpair12
                     Fev(:,nu) = Fev(:,nu) - Fpair12

                  Else If (dr(mu,nu) .lt. Rmin) Then
                     Fpair12 = - b2bvec(:,mu,nu) &
                        *(SDKa/(Rmin**(LJpa+2.d0))-SDKb/(Rmin**(LJpb+2.d0)))
                     Fev(:,mu) = Fev(:,mu) + Fpair12
                     Fev(:,nu) = Fev(:,nu) - Fpair12
                  End IF
                  ! initial flowtype = EQ sims use only bb interactions
                  If (inteq .eq. 1) Then
                     If (dr(mu,nu) .le. Rcutp .and. dr(mu,nu) .gt. Rcutg) Then
                        Fpair12= -b2bvec(:,mu,nu) &
                           *alpha*phi_bb*sin(alpha*dr(mu,nu)*dr(mu,nu) + beta)
                        Fev(:,mu) = Fev(:,mu) + Fpair12
                        Fev(:,nu) = Fev(:,nu) - Fpair12
                     End If
                  End If
                  If (inteq .eq. 0) Then
                     ! identify the stickers (mu - stickerbead no.)
                     If (MOD((mu-l),l) .eq. 0 .and. MOD((nu-l),l) .eq. 0) Then
                        ! for functionality of one - can be changed later
                        If (StickBead(mu) .le. 0 .and. StickBead(nu) .le. 0) Then
                           ! write(*,*) 'stick initial', StickBead
                           ! write(*,*) Rcutg, dr(mu,nu), Rcutp
                           If (dr(mu,nu) .le. Rcutp .and. dr(mu,nu) .gt. Rcutg) Then
                              ! write(*,*)'Force'
                              Fpair12= -b2bvec(:,mu,nu) &
                                 *alpha*phi(mu,nu)*sin(alpha*dr(mu,nu)*dr(mu,nu) + beta)
                              Fev(:,mu) = Fev(:,mu) + Fpair12
                              Fev(:,nu) = Fev(:,nu) - Fpair12

                              StickBead(mu) = StickBead(mu) + 1
                              StickBead(nu) = StickBead(nu) + 1
                              ! write(*,*) 'inside'
                           End If
                        Else
                           ! backbone
                           If (dr(mu,nu) .le. Rcutp .and. dr(mu,nu) .gt. Rcutg) Then
                              Fpair12= -b2bvec(:,mu,nu) &
                                 *alpha*phi_bb*sin(alpha*dr(mu,nu)*dr(mu,nu) + beta)
                              Fev(:,mu) = Fev(:,mu) + Fpair12
                              Fev(:,nu) = Fev(:,nu) - Fpair12
                           End If
                        End If
                     Else
                        If (dr(mu,nu) .le. Rcutp .and. dr(mu,nu) .gt. Rcutg) Then
                           Fpair12= -b2bvec(:,mu,nu) &
                              *alpha*phi_bb*sin(alpha*dr(mu,nu)*dr(mu,nu) + beta)
                           Fev(:,mu) = Fev(:,mu) + Fpair12
                           Fev(:,nu) = Fev(:,nu) - Fpair12
                        End If
                     End if
                  End If ! inteq
               End if
            End Do
         End Do
      End If ! EV
   End Subroutine get_excluded_volume_force

End Module

Module Flow_Strength_Calculations
   use csputls
   use simulation_utilities
   use Global_parameters_variables_and_types
   implicit none

   private

!   public set_flow_parameters
   public get_kappa
   public get_harmonic_trap_force

   type, public :: flow_inputs
      Integer :: flow_type = EQ

      Real (DBprec) :: flow_strength = 0.d0   ! Usually referred to as gdots
      Real (DBprec) :: flow_cessation_time = 5.d0 ! Used for extension-relaxation
      Real (DBprec) :: flow_period = 10.d0 ! For periodic flows
      ! variable_flow_strength is for if flow strength is read from file
      ! if we're using transientExtension, it will be of dimension (:,2), where column 1 is times and column 2 is gdots
      ! Otherwise, it will be (:,10), with time, then (1,1), (1,2), (1,3), (2,1) ... (3,3) components of Kappa
      Real (DBprec), allocatable :: variable_flow_strength(:,:)

      ! When using harmonic traps
      Real (DBprec) :: trapOneStrength = 0.d0   ! First harmonic trap strength
      Real (DBprec) :: trapTwoStrength = 0.d0   ! Second harmonic trap strength
      Real (DBprec), dimension(Ndim) :: trapOneInitialPosition = (/0.d0, 0.d0, 0.d0/)
      Real (DBprec), dimension(Ndim) :: trapTwoInitialPosition = (/0.d0, 0.d0, 0.d0/)
      Real (DBprec), dimension(Ndim) :: trapTwoVelocity = (/0.d0, 0.d0, 0.d0/)

   end type

contains

   subroutine get_harmonic_trap_force(time, flow_inputs_dum, Nbeads, R_bead, F_trap)
      real(DBprec), intent(in) :: time
      type(flow_inputs), intent(in) :: flow_inputs_dum
      Integer, intent(in) :: Nbeads
      real(DBprec), intent(in) :: R_bead(Ndim,Nbeads)
      real(DBprec), intent(out) :: F_trap(Ndim,Nbeads)
      real(DBprec) :: trap_two_position(Ndim), trap_one_position(Ndim)

      F_trap = 0.d0

!      if (flow_inputs%flow_type .NE. TrapConstVel) then
!         return
!      end if

      trap_one_position = flow_inputs_dum%trapOneInitialPosition
      if (flow_inputs_dum%flow_type .EQ. TrapConstVel) then
         call get_trap_two_position(time, flow_inputs_dum, trap_two_position)
      else
         trap_two_position = flow_inputs_dum%trapTwoInitialPosition
      end if

      ! Force on first and last beads, harmonic
      F_trap(:,1) = flow_inputs_dum%trapOneStrength*(trap_one_position - R_bead(:,1))
      F_trap(:,Nbeads) = flow_inputs_dum%trapTwoStrength*(trap_two_position - R_bead(:,Nbeads))
   end subroutine

   subroutine get_trap_two_position(time, flow_inputs_dum, trap_two_position)
      real(DBprec), intent(in) :: time
      type(flow_inputs), intent(in) :: flow_inputs_dum
      real(DBprec), intent(out) :: trap_two_position(Ndim)

      trap_two_position = flow_inputs_dum%trapTwoInitialPosition + &
         flow_inputs_dum%trapTwoVelocity*time
   end subroutine

   subroutine get_kappa(t,K, flow_params)
      type(flow_inputs), intent(in) :: flow_params
      Real (DBprec), intent (in) :: t
      Real (DBprec), intent (out), dimension(:,:) :: K
      Real (DBprec), dimension(Ndim*Ndim) :: flow_tensor_as_vector

      Real (DBprec) :: omgs,var,T0, gdots

      gdots = flow_params%flow_strength
      K = 0

      Select Case (flow_params%flow_type)
       case (EQ) ! Equilibrium
         K  = 0
       case (UA) ! uniaxial extension
         K(1,1) = 1
         K(2,2) = -0.5
         K(3,3) = -0.5
       case (PL) ! planar extension
         K(1,1) = 1
         K(2,2) = -1
       case (SH) ! planar shear
         K(1,2) = 1
       case (UR) ! uniaxial extension followed by relaxation
         if (t .le. flow_params%flow_cessation_time) then
            K(1,1) = 1
            K(2,2) = -0.5
            K(3,3) = -0.5
         end if
       case (PU) ! periodic uniaxial extension

         T0 = flow_params%flow_period ! time period of oscilation in strain units

         !omgs = 2*PI/T0 * gdots
         omgs = 2*PI/T0
         var = sin(omgs*t)
         K(1,1) = 1
         K(2,2) = -0.5
         K(3,3) = -0.5
         K = K * var

       case (PP) ! periodic planar extension

         T0 = 10.0d0 ! time period of oscilation in strain units

         omgs = 2*PI/T0 * gdots
         var = sin(omgs*t)
         K(1,1) = 1
         K(2,2) = -1
         K = K * var

       case(PR)
         K = 0

       case(TrapConstVel)
         K = 0

       case(ShearRelaxation)
         if (t .le. flow_params%flow_cessation_time) then
            K(1,2) = 1
         end if

       case(TransientExtension)
         ! flow is just uniaxial extension
         K(1,1) = 1
         K(2,2) = -0.5
         K(3,3) = -0.5
         ! flow rate from linear interpolation of table
         gdots = lin_interp(t, flow_params%variable_flow_strength(1,:), &
            flow_params%variable_flow_strength(2,:))
         !print *, t, gdots

       case(ArbitraryFlow)
         ! We read in some arbitrary flow strength tensor
         ! Each tensor component is linearly interpolated in time
         ! print *, t, size(flow_params%variable_flow_strength(1,:)), size(flow_params%variable_flow_strength(2:,:),2), &
         !    size(flow_params%variable_flow_strength(2:,:),1)
         flow_tensor_as_vector = lin_interp_vector(t, flow_params%variable_flow_strength(1,:), &
            flow_params%variable_flow_strength(2:,:))

         K = reshape(flow_tensor_as_vector, [3, 3])
         gdots = 1.0d0

         ! print *, t, K*gdots

      end Select

      K = K * gdots

   end subroutine get_kappa

End Module

Module Spring_Force_calculatons
   use csputls
   use simulation_utilities
   use Global_parameters_variables_and_types
   implicit none

   private

   public get_spring_force
   public solve_implicit_r
   public generate_lookup_table
   public stop_using_lookup_table, start_using_lookup_table
   public print_lookup_table
   !public input_lookup_table

   type, public :: spring_inputs
      Integer :: spring_type = HOOK
      Real (DBprec) :: finite_extensibility_parameter = 10.d0
      Real (DBprec) :: natural_length = 0.d0
   end type

   ! This is a really bad way of implementing rtsafe calculations
   ! using a general function
   Real (DBprec) :: gama_internal, dt_internal, sigma_internal

   ! for lookup table implementation
   Logical :: lookup_table_used = .false.
   Integer(k4b), parameter :: MaxLookupEntries = 10000
   Real (Dbprec), allocatable, dimension(:) :: gamma_vals, sol_vals
   Integer(k4b), save :: lookup_table_index = 2

contains

   subroutine print_lookup_table()
      print *, gamma_vals
      !print *,
      print *, sol_vals
   end subroutine

   Subroutine force_sans_hookean(sptype,Q,ff,Q0s)
      Integer, Intent (in) :: sptype
      Real (DBprec), Intent (in) :: Q        ! Bead-to-bead distance
      Real (DBprec), Intent (in) :: Q0s       ! Natural spring length
!    Real (DBprec), Intent (in) :: sqrtb     ! extensibility parameter
      Real (DOBL), Intent (out) :: ff         ! Force factor for deviation from Hookean
      ! F = Q*ff, where Q is connector vector

      ! TODO : Make it so sqrtb is an actual input to this function, and fix up the rest of the code
      ! to explicitly feed in sqrtb without dividing through by it constantly.
      ! This will make adding new force laws not an absolute nightmare, and also
      ! easier to extend with additional parameters in the future

      ! compute the force factor, ff
      Select Case (sptype)
       Case (HOOK)
         ff = 1.d0

       Case (FENE)
         ff = 1.d0/(1.d0 - (Q)*(Q))

       Case (ILC)  ! Inverse Langevin, Pade approx
         ff = (3.d0-(Q)*(Q))/(1.d0 - (Q)*(Q))/3.d0

       Case (WLC) ! Worm-like Chain, Marko-Siggia interpolation
         ff = 1._dp/(6._dp*(Q)) * ( 4._dp*(Q)+ 1._dp/(1._dp-(Q))/(1._dp-(Q)) - 1._dp)

       Case (Fraenkel)
         ff = 1.d0*(1.d0 -(Q0s/Q))

       case (FENEFraenkel)
         ff = (1.d0-Q0s/Q)/(1.d0-(Q-Q0s)*(Q-Q0s))

       case (WLC_bounded)
         ff = (2.d0/(3.d0*Q))*&
            ( &
            (((1.d0-(Q-Q0s)/(1.d0-Q0s))**(-2.d0))-1.d0)/4.d0 &
            + (Q-Q0s)/(1.d0-Q0s) &
            - Q0s*(((1.d0+(Q-Q0s)/(1.d0-Q0s))**(-2.d0))-1.d0)/4.d0 &
            + Q0s*(Q-Q0s)/(1.d0-Q0s) &
            )

       case default
         ff = 1.d0
         print *, "spring type not correctly set"
      End Select

   End Subroutine force_sans_hookean

   Subroutine get_spring_force(N,b2bvec,dr,Fs,spring_inputs_dum)
      type(spring_inputs), intent(in) :: spring_inputs_dum
      Integer, Intent (in)  :: N
      Real (DBprec), Intent (in)  :: b2bvec(Ndim,N,N) ! Ndim x Nbeads x Nbeads
      Real (DBprec), Intent (in) :: dr(N,N)       ! Nbeads x Nbeads
      Real (DBprec), Intent (out) :: Fs(Ndim,N)      ! Ndim x Nbeads

!!! Purpose:
      ! computes the net spring force on a bead due to
      ! connector springs of neighbouring beads
      ! Both the bead to bead connector vector superdiagonal matrix and the
      ! bead distances superdiagonal matrix are fed in so that these don't have
      ! to be calculated each time the procedure is called.

      Integer :: sptype
      Integer nu
      Real (DBprec) r, sqrtb, Q0sp ! Q0s = Q0/sqrtb
      Real (DOBL) Fcnu(Ndim), Fcnu1(Ndim), nonhook
      sptype = spring_inputs_dum%spring_type
      sqrtb = spring_inputs_dum%finite_extensibility_parameter

      Fs  = 0.d0
      Fcnu1 = 0.d0 ! Fc of nu-1

      Do nu = 1,N-1 ! over each of the connector

         ! only upper diagonal (beads nu and nu+1),
         ! in other words, r is distance between adjacent beads
         ! This is actually not needed for Hookean springs
         r = dr(nu,nu+1)/sqrtb
!        r = dr(nu,nu+1)
         Q0sp = spring_inputs_dum%natural_length/sqrtb
         If (Abs(r - 1.d0) .Lt. MYEPS) r = 1 - MYEPS

         Call force_sans_hookean(sptype,r,nonhook,Q0sp)

         ! connector force between adjacent beads, for each connector vector
         Fcnu = b2bvec(:,nu,nu+1) * nonhook

         ! total spring force on a bead
         Fs(:,nu) = Fcnu - Fcnu1
         Fcnu1 = Fcnu ! save for the next bead
      End Do

      ! and the last bead
      Fs(:,N) = - Fcnu1

   End Subroutine get_spring_force

   Subroutine solve_implicit_r(sptype,dtby4,gama,natscl,r,ff)
      Integer, Intent (in) :: sptype
      Real (DBprec), Intent (in) :: dtby4             ! delta t divided by 4
      Real (DBprec), Intent (in) :: gama              ! Gamma*, RHS of semi-implicit equation equation
      Real (DBprec), Intent (in) :: natscl            ! Q0 or sigma, natural spring length
      Real (DBprec), Intent (out) :: r                ! implicit solution for connector vector length
      Real (DOBL), Intent (out) :: ff                 ! connector vector force from implicit solution

      Real (DBprec) coeff(4), denom

      Real (DBprec) :: r_err

      ! Note that the equations appear particularly confusing since
      ! they are generally written in terms of gamma* and r*, which are
      ! r* = r/sqrtb and gamma* = gamma/sqrtb
      ! So the inputs and outputs from this function should actually be
      ! gamma*, r* and natscl*

      !! Purpose
      ! solves the equation r ( 1 + dt/4 ff ) == gama
      ! where ff depends on spring force law F = H Q ff,
      ! for r and returns, r and ff(r)

      if (lookup_table_used) then
         call hunt(gamma_vals, gama, lookup_table_index)

         if (lookup_table_index .eq. 0) then
            print *, "gama value beyond minimum for table"
            r = sol_vals(1)
         else if (lookup_table_index .eq. size(gamma_vals)) then
            print *, "gama value beyond maximum for table"
            r = sol_vals(size(sol_vals))
         else if (lookup_table_index .eq. 1) then
            call polint(gamma_vals(lookup_table_index:lookup_table_index+2), &
               sol_vals(lookup_table_index:lookup_table_index+2), &
               gama, r, r_err)
         else if (lookup_table_index .eq. size(gamma_vals)-1) then
            call polint(gamma_vals(lookup_table_index-2:lookup_table_index), &
               sol_vals(lookup_table_index-2:lookup_table_index), &
               gama, r, r_err)
         else
            call polint(gamma_vals(lookup_table_index-1:lookup_table_index+1), &
               sol_vals(lookup_table_index-1:lookup_table_index+1), &
               gama, r, r_err)
         end if

         Call force_sans_hookean(sptype,r,ff,natscl)
         return
      end if

      ! set up the polynomial equation (cubic), and the guess for
      ! gama >> 1, obtained by the asymptotic behaviour

      coeff(4) = 1.d0

      Select Case (sptype)
       Case (HOOK) ! Hookean
         r = gama/(1.d0 +dtby4)
         ff = 1.d0
         Return

       Case (FENE) ! FENE
         coeff(1) = gama
         coeff(2) = -(1.d0 + dtby4)
         coeff(3) = -gama
         r = 1.d0 - dtby4/2.d0/gama

       Case (ILC) ! ILC
         denom = 3.d0 + dtby4
         coeff(1) = 3.d0 * gama / denom
         coeff(2) = -(3.d0 + 3.d0* dtby4) / denom
         coeff(3) = -3.d0 * gama / denom
         r = 1.d0 - dtby4/3.d0/gama

       Case (WLC) ! WLC
         denom = 2.d0*(1.5d0 + dtby4)
         coeff(1) = -3.d0 *gama/denom
         coeff(2) =  3.d0 * (1.d0 + dtby4 + 2.d0*gama) / denom
         coeff(3) = -1.5d0 * (4.d0 + 3.d0*dtby4 + 2.d0*gama) / denom
         r = 1.d0 - Sqrt(dtby4/6.d0/gama)

       Case (Fraenkel) ! Fraenkel spring
         r = (gama + (dtby4*natscl))/(1.d0+(dtby4))
         ff = 1.d0*(1.d0-(natscl/r))
         Return

       Case (FENEFraenkel)
         coeff(1) = dtby4*natscl+gama*(1.d0-natscl**2)
         coeff(2) = -1.d0+natscl**2-dtby4+2.d0*gama*natscl
         coeff(3) = -(2.d0*natscl+gama)
         r = find_roots_cubic(coeff, max(natscl-1.d0,0.d0), natscl+1.d0)
         !r = find_cubic_root_safe(coeff, max(natscl-1.d0,0.d0), natscl+1.d0, r, 1.d-12)
         !call force_sans_hookean(FENEFraenkel,r,ff,natscl)
         !Return

       Case (WLC_bounded)
         ! This is honestly such a dodgy way of doing this, but I don't know any
         ! other way in fortran without making things VERY complicated
         dt_internal = dtby4*4.d0
         gama_internal = gama
         sigma_internal = natscl

         if (natscl .gt. 1.d0) then
            print *, "sigma/L cannot be greater than 1 for WLC bounded"
            stop
         end if

         r = rtsafe(MS_alt_val_deriv_func, &
            max(0.d0,2.d0*natscl-1.d0+epsilon(1.d0)),&
            1.0d0-epsilon(1.d0), 1d-15)
         call force_sans_hookean(sptype,r,ff,natscl)
         return

      End Select

      ! the Hookean guess is same for all the force laws
      If ((gama < 1.0_dp).and.(sptype /= FENEFraenkel)) Then
         r = gama/(1.d0 +dtby4)
      End If

      ! all the common forces laws yield a cubic in the implicit form
      ! polish the guessed root by a Newt-Raph for polynomials

      ! for WLC, the newt raph, does not converge to the correct root,
      ! for gama > 100 , therefore simply ignore polishing
      !print *, "before polish, r = ", r
      If (sptype .Eq. WLC .And. gama > 100._dp) Then
         r = r
!    Else if (sptype /= FENEFraenkel) then
      else
         Call polish_poly_root(coeff,r,1.d-14)
      End If
      !print *, "after polish, r = ", r

      Call force_sans_hookean(sptype,r,ff,natscl)

   End Subroutine solve_implicit_r

   subroutine MS_alt_val_deriv_func(x,fval,fderiv)
      ! prints the value of the semi-implicit eqn to find the root of for
      ! a MS-modified spring force law
      use Global_parameters_variables_and_types
      implicit none
      Real (DBprec), intent(in) :: x
      Real (DBprec), intent(out) :: fval
      Real (DBprec), intent(out) :: fderiv

      Real (DBprec) :: dt, gama, s

      dt = dt_internal
      s = sigma_internal
      gama = gama_internal

      fval = 6.d0*x/dt + 1.d0/4.d0*(1.d0 - s)**2.d0/(1.d0 - x)**2.d0 &
         - 1.d0/4.d0 + (x - s)/(1.d0 - s) &
         - s/4.d0*(1.d0 - s)**2.d0/(1.d0 + x - 2.d0*s)**2.d0 &
         + s/4.d0 + s*(x - s)/(1.d0 - s) &
         - 6.d0*gama/(dt)

      fderiv = 6.d0/dt + 1.d0/(1.d0 - s) + s/(1.d0 - s) &
         + (1.d0 - s)**2.d0/(2.d0*(1.d0 - x)**3.d0) &
         + ((1.d0 - s)**2.d0*s)/(2.d0*(1.d0 - 2.d0*s + x)**3.d0)


   end subroutine

   subroutine stop_using_lookup_table()
      lookup_table_used = .false.
   end subroutine

   subroutine start_using_lookup_table()
      lookup_table_used = .true.
   end subroutine

   subroutine generate_lookup_table(spring_inputs_dum, dt, max_gama, tol)
      ! Generates a lookup table such that the 2nd-order polynomial interpolation error is always
      ! smaller than tol for the range of gamma_vals
      type(spring_inputs), intent(in) :: spring_inputs_dum
      Real(DBprec), intent(in) :: dt, tol, max_gama

      Real(DBprec), dimension(MaxLookupEntries) :: G_vals_dum, sol_vals_dum
      Real(DBprec) :: dG, force_dum, G_half, sol_half, interp_val, interp_err, err_rel

      Integer(k4b) :: i

      if (allocated(gamma_vals)) then
         deallocate(gamma_vals, sol_vals)
      end if

      dG = 0.01d0

      G_vals_dum(1) = 0.d0
      call solve_implicit_r(spring_inputs_dum%spring_type, dt/4.d0, G_vals_dum(1),&
         spring_inputs_dum%natural_length/spring_inputs_dum%finite_extensibility_parameter,&
         sol_vals_dum(1), force_dum)
      G_vals_dum(2) = dG
      call solve_implicit_r(spring_inputs_dum%spring_type, dt/4.d0, G_vals_dum(2),&
         spring_inputs_dum%natural_length/spring_inputs_dum%finite_extensibility_parameter,&
         sol_vals_dum(2), force_dum)

      do i=3,MaxLookupEntries
         do while (.true.)
            G_vals_dum(i) = G_vals_dum(i-1) + dG
            call solve_implicit_r(spring_inputs_dum%spring_type, dt/4.d0, G_vals_dum(i),&
               spring_inputs_dum%natural_length/spring_inputs_dum%finite_extensibility_parameter,&
               sol_vals_dum(i), force_dum)
            G_half = G_vals_dum(i-1) + dG/2.d0
            call solve_implicit_r(spring_inputs_dum%spring_type, dt/4.d0, G_half,&
               spring_inputs_dum%natural_length/spring_inputs_dum%finite_extensibility_parameter,&
               sol_half, force_dum)
            call polint(G_vals_dum(i-2:i),sol_vals_dum(i-2:i), G_half, interp_val, interp_err)
            err_rel = abs(sol_half-interp_val)/tol
            dG = dG*0.7d0
            if (err_rel < 1.d0) then
               dG = dG*1.6d0
               exit
            end if
         end do
         if (G_vals_dum(i) > max_gama) exit
      end do

      allocate(gamma_vals(i), sol_vals(i))
      gamma_vals(:) = G_vals_dum(1:i)
      sol_vals(:) = sol_vals_dum(1:i)
      lookup_table_used = .true.

   end subroutine

End Module

Module Bending_Force_calculations
   use Global_parameters_variables_and_types
   use csputls
   use simulation_utilities
   implicit none

   private
   public get_bending_force

   type, public :: bending_potential_inputs
      Integer :: bending_potential_type = NoBendingPotential
      Real (DBprec) :: bending_stiffness = 0.d0
      Real (DBprec), allocatable, dimension(:) :: natural_angles !max(2, Nbeads-2)

   end type

contains

   subroutine get_bending_force(bending_force, bending_potential_params, &
      bead_to_bead_vector_superdiagonal_matrix, bead_to_bead_distances, Nbeads)

      type(bending_potential_inputs) :: bending_potential_params

      Integer, Intent(in) :: NBeads
      Real (DBprec), Intent (in)  :: bead_to_bead_vector_superdiagonal_matrix(:,:,:) ! Ndim x Nbeads x Nbeads
      Real (DBprec), Intent (in)  :: bead_to_bead_distances(:,:) ! Nbeads x Nbeads
      Real (DBprec), Intent (out)  :: bending_force(:,:) ! Ndim x Nbeads
      Integer  :: mu
      !Real (DBprec)  :: ForceMu(Ndim), ForceMuMinusOne(Ndim), ForceMuPlusOne(Ndim)
      Real (DBprec)  :: unit_vectors(Ndim,Nbeads-1) ! Ndim x (Nbeads - 1)
      Real (DBprec)  :: cos_theta_values(Nbeads-2) ! (Nbeads - 2)
      !Real (DBprec)  :: angles(Nbeads-2)
      real(DBprec) :: bending_stiffness

      bending_force = 0
      bending_stiffness = bending_potential_params%bending_stiffness

      if (bending_potential_params%bending_potential_type.eq.NoBendingPotential) return
      ! two-bead case leads to zero forces. For this case, we don't need to allocate natural_angles
      if (Nbeads.eq.2) return

      call get_unit_vectors_from_connector_vectors_and_distances(unit_vectors,&
         bead_to_bead_vector_superdiagonal_matrix, bead_to_bead_distances, Nbeads)

      call get_cos_theta_from_unit_vectors(cos_theta_values, unit_vectors, Nbeads)

      !angles = acos(cos_theta_values)

      ! calculate bending forces on each bead
      if (bending_potential_params%bending_potential_type.eq.OneMinusCosTheta) then

         ! For Nbeads>2, force on first and last bead is always the same
         bending_force(:,1) = bending_stiffness/bead_to_bead_distances(1,2)&
            *(cos_theta_values(1)*unit_vectors(:,1) - unit_vectors(:,2))
         bending_force(:,Nbeads) = bending_stiffness/bead_to_bead_distances(Nbeads-1,Nbeads)&
            *(-cos_theta_values(Nbeads-2)*unit_vectors(:,Nbeads-1) + unit_vectors(:,Nbeads-2))

         ! For Nbeads = 3, the middle bead only feels force from its own angle
         if (Nbeads.eq.3) then
            bending_force(:,2) = bending_stiffness*(&
               1/bead_to_bead_distances(2,3)*(cos_theta_values(1)*unit_vectors(:,2) - unit_vectors(:,1)) &
               + 1/bead_to_bead_distances(1,2)*(-cos_theta_values(1)*unit_vectors(:,1) + unit_vectors(:,2)) &
               )
            ! For Nbeads > 3, the next two beads feel forces from adjacent beads as well
         else if (Nbeads.gt.3) then
            bending_force(:,2) = bending_stiffness*(&
               1/bead_to_bead_distances(2,3)*(cos_theta_values(1)*unit_vectors(:,2) - unit_vectors(:,1)) &
               + 1/bead_to_bead_distances(1,2)*(-cos_theta_values(1)*unit_vectors(:,1) + unit_vectors(:,2)) &
               + 1/bead_to_bead_distances(2,3)*(cos_theta_values(2)*unit_vectors(:,2) - unit_vectors(:,3)) &
               )
            bending_force(:,Nbeads-1) = bending_stiffness*(&
               1/bead_to_bead_distances(Nbeads-1,Nbeads)&
               *(cos_theta_values(Nbeads-2)*unit_vectors(:,Nbeads-1) - unit_vectors(:,Nbeads-2)) &
               + 1/bead_to_bead_distances(Nbeads-2,Nbeads-1)&
               *(-cos_theta_values(Nbeads-2)*unit_vectors(:,Nbeads-2) + unit_vectors(:,Nbeads-1)) &
               + 1/bead_to_bead_distances(Nbeads-2,Nbeads-1)&
               *(-cos_theta_values(Nbeads-3)*unit_vectors(:,Nbeads-2) + unit_vectors(:,Nbeads-3)) &
               )
            ! When Nbeads > 5, we also have to consider internal beads which feel forces from 3 total angles
            if (Nbeads.gt.4) then
               do mu=3,Nbeads-2
                  bending_force(:,mu) = bending_stiffness*(&
                     1/bead_to_bead_distances(mu,mu+1)&
                     *(cos_theta_values(mu-1)*unit_vectors(:,mu) - unit_vectors(:,mu-1)) &
                     + 1/bead_to_bead_distances(mu-1,mu)&
                     *(-cos_theta_values(mu-1)*unit_vectors(:,mu-1) + unit_vectors(:,mu)) &
                     + 1/bead_to_bead_distances(mu-1,mu)&
                     *(-cos_theta_values(mu-2)*unit_vectors(:,mu-1) + unit_vectors(:,mu-2)) &
                     + 1/bead_to_bead_distances(mu,mu+1)&
                     *(cos_theta_values(mu)*unit_vectors(:,mu) - unit_vectors(:,mu+1)) &
                     )
               end do
            end if
         end if
      end if

   end subroutine

End Module

Module Hydrodynamic_interaction_calculations
   use Global_parameters_variables_and_types
   use csputls
   use simulation_utilities
   use random_numbers
   implicit none

   private

   type, public :: Hydrodynamic_interaction_inputs
      Integer :: EigenvalueCalcMethod = EigsFixman
      Integer :: delSCalcMethod = Chebyshev
      Integer :: ChebUpdateMethod = UpdateChebNew
      Real (DBprec) :: nchebMultiplier = 1.d0
      Real (DBprec) :: hstar = 0.d0
      Real (DBprec) :: fd_err_max = 0.0025d0
   end type

   public get_diffusion_tensor_with_HI
   public set_Hydrodynamic_interaction_parameters
   public set_chebyshev_parameters
   public get_dels_approx
   public get_delS_cholesky
   public get_delS_exact

   ! Constants to calculate several quantities
   Real (DBprec), parameter :: RPI = sqrt(PI), TRPI = 2.d0*RPI, TPI = 2.d0*PI
   Real (DBprec), parameter :: C1 = TPI/3.d0, C3 = 0.75d0/RPI*0.375d0
   Real (DBprec), parameter :: C7 = 0.09375d0/RPI, RPI3by4 = RPI*0.75d0
   Real (DBprec), parameter :: C5 = 14.14855378d0, C6 = 1.21569221d0
   Integer, parameter :: incx = 1
   Integer, parameter :: incy = 1
   Real (DBprec) :: fd_err_max = 0.0025d0

   Real (DBprec) :: L_max, L_min, d_a, d_b, cheba(0:MAXCHEB), alpha, beta
   Integer :: ncheb, ncheb_old
   logical :: ncheb_flag, fd_check

   Integer :: EigCalcMethod = EigsFixman
   Integer :: nChebUpdateMethod = UpdateChebNew
   Real (DBprec) :: nchebMult = 1.d0

contains

   subroutine set_Hydrodynamic_interaction_parameters(HI_params)
      type(Hydrodynamic_interaction_inputs), intent(in) :: HI_params

      EigCalcMethod = HI_params%EigenvalueCalcMethod
      nchebMult = HI_params%nchebMultiplier
      fd_err_max = HI_params%fd_err_max
      nChebUpdateMethod = HI_params%ChebUpdateMethod

   end subroutine

   subroutine get_diffusion_tensor_with_HI(hstar, deltaR_sup, b2b_sup, Diffusion_sup, Nbeads)
      real (DBprec), intent(in) :: hstar
      real (DBprec), intent(in), dimension(:,:) :: deltaR_sup
      real (DBprec), intent(in), dimension(:,:,:) :: b2b_sup
      real (DBprec), intent(inout), dimension(:,:,:,:) :: Diffusion_sup
      Integer, intent(in) :: Nbeads

      real (DBprec) :: rs, hsbyrs, rsbyhs, hsbyrs2, om1, om2
      Integer :: nu, mu, i

      Diffusion_sup = 0.0d0
      ! diagonal elements
      Forall (mu = 1:Nbeads, i = 1:Ndim )
         Diffusion_sup(i,mu,i,mu) = 1.d0
      End Forall

      ! Calculate the RPY diffusion tensor
      If (Hstar.Gt.0) Then
         Do nu = 2,NBeads
            Do mu = 1,nu-1
               rs = deltaR_sup(mu,nu)
               hsbyrs = Hstar/rs
               rsbyhs = rs/Hstar
               hsbyrs2 = hsbyrs*hsbyrs
               If (rsbyhs.Ge.TRPI) Then
                  om1 = RPI3by4*hsbyrs*(1.0+C1*hsbyrs2)
                  om2 = RPI3by4*(hsbyrs/(rs*rs))*(1._dp-TPI*hsbyrs2)
               Else
                  om1 = 1._dp-C3*rsbyhs
                  om2 = C7*rsbyhs/(rs*rs)
               End If

               ! store only the upper triangular in the mu,nu index.
               Call tensor_prod_of_vec(om2,b2b_sup(:,mu,nu), &
                  Diffusion_sup(:,mu,:,nu) )

               ! additional factor for diagonal elements, in each mu,nu
               Forall (i = 1:Ndim )
                  Diffusion_sup(i,mu,i,nu) = Diffusion_sup(i,mu,i,nu) + om1
               End Forall
            End Do
         End Do
      End If ! Hstar


   end subroutine

   subroutine get_delS_cholesky(Nbeads, hstar, Diffusion_sup, delts, Dels, X_input)
      Integer, intent(in) :: Nbeads
      Real (DBprec), intent(in) :: diffusion_sup(:,:,:,:)
      Real (DBprec), intent(in) :: delts, hstar
      Real (DBprec), intent(out) :: Dels(:,:)
      Real (DBprec), intent(in), optional :: X_input(:,:)

      Real (DBprec), dimension(Ndim, Nbeads) :: X_0
      Real (DBprec), dimension(Ndim,Nbeads,Ndim,Nbeads) :: chol_matrix
      !Real (DBprec), dimension(Ndim*Nbeads, Ndim*Nbeads) :: test
      Integer :: Ndof, lda, info

      external :: dpotrf, dgemv

      Ndof = Ndim*Nbeads
      lda = Ndof

      DelS = 0.d0

      if (present(X_input)) then
         X_0 = X_input
      else
         X_0 = 0.d0
         Call ran_1(Ndof, X_0)
         X_0 = X_0 - 0.5d0
         X_0 = (X_0*X_0*C5 + C6)*X_0! Element-wise multiplications
      end if

      If (Hstar.Gt.0) Then
         ! get the cholesky factorisation
         chol_matrix = diffusion_sup
         call dpotrf('U', Ndof, chol_matrix, lda, INFO)

         ! ---- LOCAL MODIFICATION (not in upstream SingleChainBD) -------------
         ! INFO was declared and never read. dpotrf returns INFO > 0 when the
         ! leading minor of that order is not positive definite, and it stops
         ! there: the factor is only partially computed and the remainder of
         ! chol_matrix still holds the original diffusion tensor. The dgemv below
         ! then produces a finite but WRONG Brownian displacement, and the run
         ! carries on with no indication that anything happened.
         !
         ! For the RPY tensor this can only occur if the configuration itself has
         ! gone wrong -- typically beads driven on top of one another after the
         ! implicit corrector failed to converge -- so the message points there.
         if (INFO .gt. 0) then
            write (*,1025) INFO
         else if (INFO .lt. 0) then
            write (*,1026) -INFO
         end if
1025     format ('WARNING: dpotrf reports the diffusion tensor is not positive', &
            ' definite at leading minor ', I5, /, &
            '         delS comes from a partial factorisation and is WRONG.', /, &
            '         The configuration is degenerate; look for an earlier', &
            ' "Loop exceeded" message.')
1026     format ('WARNING: dpotrf rejected argument ', I3)
         ! ---- end local modification -----------------------------------------

         alpha = 1.D0
         beta = 0.D0

         ! multiply lower cholesky matrix by dW
         call dgemv('t', Ndof, Ndof, alpha,chol_matrix,lda, X_0,incx, &
            beta,delS,incy)
      else
         DelS = X_0
      end if

      DelS = DelS * sqrt(delts)

   end subroutine

   subroutine get_delS_exact(Nbeads, hstar, Diffusion_sup, delts, Dels, X_input)
      Integer, intent(in) :: Nbeads
      Real (DBprec), intent(in) :: diffusion_sup(:,:,:,:)
      Real (DBprec), intent(in) :: delts, hstar
      Real (DBprec), intent(out) :: Dels(:,:)
      Real (DBprec), intent(in), optional :: X_input(:,:)

      Real (DBprec) :: dummy_D(Ndim,Nbeads,Ndim,Nbeads)
      Real (DBprec), dimension(Ndim, Nbeads) :: X_0
      Real (DBprec), dimension(Ndim*Nbeads,Ndim*Nbeads) :: eig_vec_matrix, sqrtD
      Real (DBprec), dimension(Ndim*Nbeads) :: eig_vector
      Integer :: Neigs
      !Real (DBprec), dimension(Ndim*Nbeads, Ndim*Nbeads) :: test
      Integer :: Ndof, lda, info, ii, ldz
      Integer, save :: lwork, liwork
      Integer, parameter :: LWMAX = 100000
      Real (DBprec), dimension(LWMAX) :: work
      Integer, dimension(LWMAX) :: iwork
      ! Integer, dimension(Ndim*Nbeads) :: isuppz
      Integer, dimension(Ndim*Nbeads) :: ifail
      Real (DBprec) :: eigenvalues_tolerance = -1.d-4

      logical, save :: workspace_queried = .false.

      external ::  dger, dgemv
      external :: dsyevx

      Ndof = Ndim*Nbeads
      lda = Ndof
      ldz = Ndof

      dummy_D = diffusion_sup

      DelS = 0.d0
      !Generate the random vector X_0

      if (present(X_input)) then
         X_0 = X_input
      else
         X_0 = 0.d0
         Call ran_1(Ndof, X_0)
         X_0 = X_0 - 0.5d0
         X_0 = (X_0*X_0*C5 + C6)*X_0! Element-wise multiplications
      end if

      ! print *, "X_0 is"
      ! print *, X_0

      If (Hstar.Gt.0) Then
         ! first find the optimal workspace with a dummy run, and save results
         if (.not.workspace_queried) then
            workspace_queried = .true.
            lwork = -1
            liwork = -1
            call dsyevx('V', 'A', 'U', Ndof, dummy_D, lda, 1.d0, 2.d0, 1, 2, eigenvalues_tolerance, &
               Neigs,eig_vector,eig_vec_matrix,ldz,work,lwork,iwork,ifail,INFO)
         end if
         lwork = min(LWMAX, INT(work(1)))
         ! do the real calculation
         call dsyevx('V', 'A', 'U', Ndof, dummy_D, lda, 1.d0, 2.d0, 1, 2, eigenvalues_tolerance, &
            Neigs,eig_vector,eig_vec_matrix,ldz,work,lwork,iwork,ifail,INFO)

         ! if we get an error, try re-calculating the number of terms
         if (info.ne.0) then
            lwork = -1
            liwork = -1
            call dsyevx('V', 'A', 'U', Ndof, dummy_D, lda, 1.d0, 2.d0, 1, 2, eigenvalues_tolerance, &
               Neigs,eig_vector,eig_vec_matrix,ldz,work,lwork,iwork,ifail,INFO)

            lwork = min(LWMAX, INT(work(1)))

            call dsyevx('V', 'A', 'U', Ndof, dummy_D, lda, 1.d0, 2.d0, 1, 2, eigenvalues_tolerance, &
               Neigs,eig_vector,eig_vec_matrix,ldz,work,lwork,iwork,ifail,INFO)
         end if

         ! this next step can probably be vectorised in a clever way, if performance is important
         sqrtD = 0.d0
         do ii=1,Ndof
            call dger(Ndof, Ndof, eig_vector(ii)**(0.5d0), &
               eig_vec_matrix(:,ii), 1, eig_vec_matrix(:,ii), 1, &
               sqrtD, lda)
         end do

         alpha = 1.D0
         beta = 0.D0

         ! multiply sqrtD matrix by X_0
         call dgemv('n', Ndof, Ndof, alpha,sqrtD,lda, X_0,incx, &
            beta,delS,incy)

      else
         DelS = X_0
      end if

      DelS = DelS * sqrt(delts)

   end subroutine

   subroutine get_exact_eigenvalues(n, A, maxev, minev)
      Integer, intent(in) :: n
      Real (DBprec), intent(in) :: A(:,:,:,:)
      Real (DBprec), intent(out) :: maxev, minev

      Real (DBprec), dimension(n,n) :: eig_vec_matrix
      Real (DBprec), dimension(n) :: eig_vector
      Integer :: Neigs
      Integer :: Ndof, lda, info, ldz
      Integer, save :: lwork, liwork
      Integer, parameter :: LWMAX = 50000
      Real (DBprec), dimension(LWMAX) :: work
      Integer, dimension(LWMAX) :: iwork
      ! Integer, dimension(Ndim*Nbeads) :: isuppz
      Integer, dimension(n) :: ifail
      Real (DBprec) :: eigenvalues_tolerance = -1.d-4
      logical, save :: workspace_queried = .false.

      external :: dsyevx

      ! call PRINT_MATRIX( 'diff matrix', n, n, reshape(A, (/n,n/)))

      ! find eigenvalues directly
      Ndof = n
      lda = n
      ldz = n
      if (.not.workspace_queried) then
         workspace_queried = .true.
         lwork = -1
         liwork = -1
         ! print *, lwork
         call dsyevx('N', 'A', 'U', Ndof, A, lda, 1.d0, 2.d0, 1, 2, eigenvalues_tolerance, &
            Neigs,eig_vector,eig_vec_matrix,ldz,work,lwork,iwork,ifail,INFO)
      end if
      lwork = min(LWMAX, INT(work(1)))
      print *, lwork

      call dsyevx('N', 'A', 'U', Ndof, A, lda, 1.d0, 2.d0, 1, 2, eigenvalues_tolerance, &
         Neigs,eig_vector,eig_vec_matrix,ldz,work,lwork,iwork,ifail,INFO)

      ! if we get an error, try re-calculating the number of terms
      if (info.ne.0) then
         lwork = -1
         liwork = -1
         call dsyevx('N', 'A', 'U', Ndof, A, lda, 1.d0, 2.d0, 1, 2, eigenvalues_tolerance, &
            Neigs,eig_vector,eig_vec_matrix,ldz,work,lwork,iwork,ifail,INFO)

         lwork = min(LWMAX, INT(work(1)))

         call dsyevx('N', 'A', 'U', Ndof, A, lda, 1.d0, 2.d0, 1, 2, eigenvalues_tolerance, &
            Neigs,eig_vector,eig_vec_matrix,ldz,work,lwork,iwork,ifail,INFO)

         print *, lwork
      end if

      ! print *, "info", info
      minev = eig_vector(1)
      maxev = eig_vector(Neigs)

      ! print *, "minev and maxev: ", minev, maxev
   end subroutine

   subroutine set_chebyshev_parameters(Nbeads, diffusion_sup, hstar)
      Integer, intent(in) :: Nbeads
      Real (DBprec), intent(in) :: hstar
      Real (DBprec), intent(in) :: diffusion_sup(:,:,:,:)
      Integer :: Ndof
      Real (DBprec) :: dummy_D(Ndim,Nbeads,Ndim,Nbeads)
      logical, save :: first_run = .true.
      Integer, save :: upper_limit_ncheb = MAXCHEB

      !     Get initial estimates of L_max and L_min using
      !     Kroeger et al's approx. of Zimm theory's preds.;
      !     the number of Chebyshev terms, and the Chebyshev coefficients
      ! L_max = 2.d0*(1.d0 + PI*Sqrt(Dble(NBeads))*Hstar)
      ! L_min = (1.d0 - 1.71d0*Hstar)/2._dp     ! Note Hstar < 0.58
      ! ncheb = Int(Sqrt(L_max/L_min)+0.5d0)+1
      ! ncheb_flag = .False.
      ! d_a = 2.d0/(L_max-L_min)
      ! d_b = -(L_max+L_min)/(L_max-L_min)
      ! Call chbyshv(ncheb, d_a, d_b, cheba)
      dummy_D = diffusion_sup

      Ndof = Ndim*Nbeads
      if (EigCalcMethod==EigsFixman) Then
         Call maxminev_fi(Ndof, dummy_D, L_max, L_min)
      else if (EigCalcMethod==EigsExact) Then
         Call get_exact_eigenvalues(Ndof, dummy_D, L_max, L_min)
      end if

      if ((nChebUpdateMethod==UpdateChebNew).or.(first_run.eqv..true.)) then
         first_run=.false.
         ncheb = Int(Sqrt(L_max/L_min)*nchebMult+0.5)+1
         upper_limit_ncheb = ncheb*2.d0
         ! d_a = 2._dp/(L_max-L_min)
         ! d_b = -(L_max+L_min)/(L_max-L_min)
      else if (nChebUpdateMethod==UpdateChebAddOne) Then
         ncheb = ncheb + 1
         ! fd-theorem isn't perfect, so limit the number of terms to some maximum
         if (ncheb > upper_limit_ncheb) then
            ncheb = upper_limit_ncheb
         end if
      end if

      d_a = 2._dp/(L_max-L_min)
      d_b = -(L_max+L_min)/(L_max-L_min)

      If (ncheb.Gt.MAXCHEB) ncheb = MAXCHEB
      Call chbyshv(ncheb, d_a, d_b, cheba)

      ! print *, 'Lmin and Lmax'
      ! print *, L_min, L_max

      ! print *, 'ncheb'
      ! print *, ncheb

   end subroutine

   subroutine get_fd_error(Nbeads, Diffusion_sup, Dels, X_0, fd_err)
      Integer, intent(in) :: Nbeads
      Real (DBprec), intent(in) :: diffusion_sup(:,:,:,:)
      Real (DBprec), intent(in) :: Dels(:,:)
      Real (DBprec), intent(in) :: X_0(:,:)
      Real (DBprec), intent(out) :: fd_err

      Integer :: Ndof, lda
      Real (DBprec) :: X_l(Ndim,Nbeads), temp1
      Real (DOBL) :: ddot
      external :: dsymv

      Ndof = Nbeads*Ndim
      lda = Ndof

      fd_err = ddot(Ndof, DelS, 1, DelS, 1) ! BLAS-1 function
      ! print *, "fd_err 1", fd_err
      alpha = 1.D0
      beta = 0.D0

      !Use D:X_0X_0 = X_0.D.X_0
      !Get D.X_0 first, and then get the dot product of X_0 with the
      ! resulting vector. X_l is reused
      Call dsymv('U', Ndof,alpha,Diffusion_sup,lda, &
         X_0,incx, beta,X_l,incy)
      temp1 = ddot(Ndof, X_0, 1, X_l, 1)

      fd_err = Abs((fd_err - temp1)/temp1)

   end subroutine

   subroutine get_dels_approx(Nbeads, hstar, Diffusion_sup, delts, Dels, ncheb_count, fd_err_count, X_input)

      Integer, intent(in) :: Nbeads
      Real (DBprec), intent(in) :: diffusion_sup(:,:,:,:)
      Real (DBprec), intent(in) :: delts, hstar
      Real (DBprec), intent(out) :: Dels(:,:)
      Integer, intent(out), optional :: ncheb_count
      Real (DBprec), intent(out), optional :: fd_err_count
      Real (DBprec), intent(in), optional :: X_input(:,:)

      Real (DBprec), dimension(Ndim, Nbeads) :: X_0, X_l_1, X_l, X_lp1
      Real (DBprec), dimension(Ndim, Nbeads) :: Dels_exact
      Real (DBprec), dimension(Ndim, Nbeads, Ndim, Nbeads) :: Dp_sym_up
      Integer :: Ndof, lda
      Integer :: mu, i, l
      Integer, save :: cheb_update_step = 1
      Real (DBprec) :: fd_err, temp1, exact_err

      logical, save :: update_chebyshev_terms = .true.

      ! external BLAS functions
      !Real (SNGL) snrm2, sdot
      Real (DOBL) :: ddot
      external :: dsymv

      Ndof = Ndim*Nbeads
      lda = Ndof

      DelS = 0.d0

      if (present(X_input)) then
         X_0 = X_input
      else
         X_0 = 0.d0
         Call ran_1(Ndof, X_0)
         X_0 = X_0 - 0.5d0
         X_0 = (X_0*X_0*C5 + C6)*X_0! Element-wise multiplications
      end if

      ! Has check for deviation from fluctuation-dissipation theorem
      ! been performed?
      fd_check = .False.

      if (update_chebyshev_terms) then
         call set_chebyshev_parameters(Nbeads, diffusion_sup, hstar)
         update_chebyshev_terms = .false.
      end if

      if (nChebUpdateMethod==UpdateChebAddOne) Then
         if (mod(cheb_update_step,20)==0) then
            ncheb = ncheb - 1
         end if
         cheb_update_step = cheb_update_step + 1
      end if

      If (Hstar.Gt.0) Then
         FDloop: Do
            ! Update DelS vector
            DelS = cheba(0) * X_0

            ! Shift the D matrix
            Dp_sym_up = d_a * Diffusion_sup

            ! diagonal elements
            Forall (mu = 1:Nbeads, i = 1:Ndim )
               Dp_sym_up(i,mu,i,mu) = Dp_sym_up(i,mu,i,mu) + d_b
            End Forall

            ! Calculate the second Chebyshev vector
            X_l_1 = X_0
            alpha = 1.D0
            beta = 0.D0

            ! BLAS2 symmetric matrix-vector multiplication
            Call dsymv('U', Ndof, alpha,Dp_sym_up,lda,  X_l_1,incx, &
               beta,X_l,incy)

            ! Update DelS vector
            DelS = DelS + cheba(1)*X_l

            Do l = 2,ncheb
               alpha = 2.D0
               beta = 0.D0
               Call dsymv('U', Ndof, alpha,Dp_sym_up,lda, X_l,incx,  &
                  beta,X_lp1,incy)

               X_lp1 = X_lp1-X_l_1
               X_l_1 = X_l

               ! The l-th Chebyshev vector
               X_l = X_lp1

               ! Update DelS vector
               DelS = DelS + cheba(l)*X_l

            End Do

            ! Calculate the deviation from
            ! the fluctuation-dissipation theorem
            If (.Not.fd_check) Then

               call get_fd_error(Nbeads, Diffusion_sup, Dels, X_0, fd_err)

            End If

            If ((fd_err.Le.fd_err_max).Or.fd_check) Then
               Exit FDloop
            Else
               fd_check = .True.
               call set_chebyshev_parameters(Nbeads, diffusion_sup, hstar)
            End If
         End Do FDloop

         if (present(ncheb_count)) then
            ncheb_count = ncheb
         end if
         if (present(fd_err_count)) then
            fd_err_count = fd_err
         end if
      Else ! Hstar == 0, the free-draining case
         DelS = X_0
      End If

      DelS = DelS * sqrt(delts)

   end subroutine

End Module

Module Physics_subroutines
   ! This module contains any subroutines (and required variables) related to the
   ! actual physics of the simulation, such as forces, HI, EV, flow fields etc

   Use Global_parameters_variables_and_types
   Use csputls
   use simulation_utilities
   use properties
   use random_numbers
   use Excluded_Volume_Calculations
   use Flow_Strength_Calculations
   use Spring_Force_calculatons
   use Bending_Force_calculations
   use Hydrodynamic_interaction_calculations
   Implicit None

   private

   public Time_Integrate_Chain

   type, public :: physical_parameters
      type(Excluded_Volume_inputs) :: EV_inputs
      type(flow_inputs) :: Flow_inputs
      type(spring_inputs) :: spring_inputs
      type(bending_potential_inputs) :: bend_inputs
      type(Hydrodynamic_interaction_inputs) :: HI_params
      Integer :: number_of_beads = 10

      Real (DBprec) :: hstar = 0.d0
   end type

   type, public :: simulation_parameters

      Integer :: number_of_samples_to_take = 0
      Integer (k4b) :: simulation_seed = 1

      Real (DBprec) :: time_at_simulation_start = 0.d0
      Real (DBprec) :: time_at_simulation_end = 1.d0
      Real (DBprec) :: simulation_timestep = 0.01d0
      Real (DBprec) :: implicit_loop_exit_tolerance = 1.d-3
      Integer :: update_center_of_mass = 1
      Real (DBprec), dimension(Ndim) :: initial_center_of_mass = 0.d0

      !Real (DBprec), allocatable, dimension(:) :: times_to_take_samples
      Integer (kind=k4b), allocatable :: sample_indexes(:)

   end type

Contains

   Subroutine Time_Integrate_Chain(R_Bead, phys_params, sim_params, output_variables)
      !Use blas_interfaces unable to obtain proper interface
      !which allows matrices of arbitrary rank to b passed

      Real (DBprec), Intent (inout) :: R_Bead(:,:) ! Initial positions of Beads
      type(physical_parameters), intent(in) :: phys_params
      type(simulation_parameters), intent(in) :: sim_params
      type(calculated_variables), intent(out) :: output_variables

      Integer :: NBeads

      Integer :: spring_type ! Spring force law type
      ! Hookean = 1, FENE = 2, ILC = 3, WLC = 4
      Integer :: FlowType
      Real (DBprec)  :: gdots

      Real (DBprec) :: tcur,tmax,Delts  ! Integration time interval specs

      Real (DBprec) :: Hstar         ! HI parameter (non-dim)
!     Real (DBprec) :: Zstar,Dstar   ! EV parameters (non-dim)
      Real (DBprec) :: Q0s
      real (DBprec) :: sqrtb
      real (DBprec) :: imploop_tol

      ! external BLAS functions
      !Real (SNGL) snrm2, sdot
      Real (DOBL) dnrm2
      external :: dsymv

      Integer (k4b) :: seed1 ! seed for rnd number

      Integer :: Nsamples     ! Number of sampling points
      !Real (DBprec), Dimension(sim_params%number_of_samples_to_take) :: times    ! Sampling instances
      Real (DBprec), Dimension(NProps,sim_params%number_of_samples_to_take) :: samples
      Real (DBprec), Dimension(sim_params%number_of_samples_to_take) :: time_cdf
      Real (DBprec), Dimension(sim_params%number_of_samples_to_take,&
         Ndim, phys_params%number_of_beads) :: confi
      Real (DBprec), Dimension(sim_params%number_of_samples_to_take,&
         Ndim, phys_params%number_of_beads) :: grad
      Real (DBprec), Dimension(sim_params%number_of_samples_to_take, Ndim) :: center_of_mass_samples

      !!!_________________________________________________________|
      !     This driver uses:
      !     Predictor-corrector        scheme for excluded volume
      !     Fully implicit scheme for the spring force
      !           based on Ottinger's and Somasi et al.'s suggestions
      !     Warner spring
      !     Rotne-Prager-Yamakawa HI tensor with Chebyshev polynomial
      !         approx. using Fixman's approximation to get conditioner
      !     Narrow Gaussian EV potential
      !!!_________________________________________________________|

      Real (DBprec) time, gama_mag, lt_err, dtsby4, pcerr, delx2
      Real (DBprec)&
      ! various scratch storages for position vector
         cofm(Ndim), & ! center of mass
         cofm_cumulative(Ndim), & !cumulative center of mass
         R_pred(Ndim,phys_params%number_of_beads), &
         DR_pred(Ndim,phys_params%number_of_beads), &
         Ups_pred(Ndim,phys_params%number_of_beads), &
         R_pred_old(Ndim,phys_params%number_of_beads), &
         R_corr(Ndim,phys_params%number_of_beads), &
         Rtemp(Ndim,phys_params%number_of_beads), &
         delta_R(phys_params%number_of_beads,phys_params%number_of_beads),&
         b2b_sup(Ndim,phys_params%number_of_beads,phys_params%number_of_beads), & ! bead to bead vector (super diagonal)
         deltaR_sup(phys_params%number_of_beads,phys_params%number_of_beads), & ! bead to bead dist (super diagonal)
      ! forces
         F_ev(Ndim,phys_params%number_of_beads), &     ! excluded volume forces
         F_spring(Ndim,phys_params%number_of_beads), & ! spring forces
         F_bend(Ndim,phys_params%number_of_beads), &   ! bending potential forces
         F_tot(Ndim,phys_params%number_of_beads), &    ! total forces
         F_trap(Ndim,phys_params%number_of_beads), &   ! forces from trap
         Force_besides_spring(Ndim,phys_params%number_of_beads), & !forces from everything but the spring
      ! symmetric diffusion tensor, only upper diagonal elements stored
         Diffusion_sup(Ndim,phys_params%number_of_beads,Ndim,phys_params%number_of_beads), &
         DelS(Ndim,phys_params%number_of_beads),&
         kappa(Ndim,Ndim),&  !flow field divergence
         gama_mu(Ndim), &
         diffD(Ndim,phys_params%number_of_beads-1,Ndim,phys_params%number_of_beads), &
         diffUps(Ndim,phys_params%number_of_beads-1), &
      !diffD_reshaped(phys_params%number_of_beads-1,Ndim,Ndim*phys_params%number_of_beads), &
         Ql_corr(phys_params%number_of_beads-1), &
         Ql_pred(phys_params%number_of_beads-1)

      ! some connector forces and related requiring higher precision
      Real (DOBL) ff,  &
         F_con_mu(Ndim), &    ! connector force between beads
         F_con_mu1(Ndim)    ! connector force, save for mu-1

      !Logical ncheb_flag

      !     Other definitions...
      Integer  :: mu, nu, isample, lt_count, ncheb_count
      Integer(kind=k4b) :: step
      Real (DBprec) :: sqrt2inv, fd_err_count
      Real (DBprec) :: r

      !     Definitions for the BLAS-2 routine "ssymv"
      Integer :: Ndof, lda, ldadiff, incx, incy
      Real (DBprec) :: alpha, beta
      Real (DBprec) :: R_list_save(Ndim,phys_params%number_of_beads)
      !Real (DBprec) :: R0(Ndim,phys_params%number_of_beads)

      !call set_seed(seed1)

      ! Put input variables into dummy variables to make naming/equations easier
      Nbeads = phys_params%number_of_beads
      spring_type = phys_params%spring_inputs%spring_type
      hstar = phys_params%HI_params%hstar
      Q0s = phys_params%spring_inputs%natural_length
      sqrtb = phys_params%spring_inputs%finite_extensibility_parameter
      FlowType = phys_params%Flow_inputs%flow_type
      gdots = phys_params%Flow_inputs%flow_strength

      tcur = sim_params%time_at_simulation_start
      !print *, "tcur is : ", tcur
      tmax = sim_params%time_at_simulation_end
      Delts = sim_params%simulation_timestep
      !times = sim_params%times_to_take_samples
      seed1 = sim_params%simulation_seed
      Imploop_tol = sim_params%implicit_loop_exit_tolerance
      Nsamples = sim_params%number_of_samples_to_take

      sqrt2inv = 1.d0/(2.d0**0.5d0)

      incx = 1
      incy = 1
      Ndof = Ndim*NBeads  ! degrees of freedom
      lda = Ndof

      pcerr = 0.0d0
      lt_count = 0
      fd_err_count = 0.d0
      ncheb_count = 1

      delta_R = 0.0d0
      Ql_pred = 0.d0
      Ql_corr = 0.d0

      isample = 1
      delx2 = 0

      time = tcur
      step = 1

      time_cdf = 0.d0
      confi = 0.d0
      grad = 0.d0
      samples = 0.d0

      cofm_cumulative = sim_params%initial_center_of_mass

      call set_EV_parameters(phys_params%number_of_beads,phys_params%EV_inputs)
      call set_Hydrodynamic_interaction_parameters(phys_params%HI_params)
      ! call set_chebyshev_parameters(Nbeads, hstar)

      !___________________________________________________________________|
      !                The Time integration loop begins here              |
      !___________________________________________________________________|


      Overtime: Do While (time.Le.Tmax+Delts/2.d0)
         ! find the center of mass
         cofm = 0.d0
         ! Saving absolute position for Neighbour list
         R_list_save = R_bead
         Do mu = 1, NBeads
            cofm = cofm + R_bead(:,mu)
         End Do

         cofm = cofm/NBeads
         cofm_cumulative = cofm_cumulative + cofm

         ! shift the origin to the current center of mass
         if (sim_params%update_center_of_mass.eq.1) then
            Forall (mu = 1: NBeads)
               R_Bead(:,mu) = R_Bead(:,mu) - cofm
            End Forall
         end if

         ! calculate the bead to bead vector and its magnitude
         ! for use in the forces
         Call b2bvector_sym_up(NBeads,R_Bead,b2b_sup)
         Call modr_sym_up(NBeads,b2b_sup,deltaR_sup)

         ! change the nature of the spring, in the specific subroutine
         ! in get_spring_force(), in module Bead_spring_model
         Call get_spring_force(NBeads,b2b_sup,deltaR_sup,F_spring,phys_params%spring_inputs)
         Call get_excluded_volume_force(NBeads,b2b_sup,deltaR_sup,F_ev,FlowType)
         Call get_bending_force(F_bend, phys_params%bend_inputs, b2b_sup, deltaR_sup, NBeads)
         call get_harmonic_trap_force(time, phys_params%Flow_inputs, Nbeads, R_Bead, F_trap)

         ! Therefore, total force on each bead...
         F_tot = F_spring + F_ev + F_bend + F_trap

         ! Calculate the RPY diffusion tensor
         call get_diffusion_tensor_with_HI(hstar, deltaR_sup, b2b_sup, Diffusion_sup, Nbeads)

         ! mean square displacement of cofm
         ! since the bead positions are centered w.r.t. cofm at every
         ! instant, cofm is the incremental displacement
         if (time .gt. tcur) then
            delx2 = delx2 + Sum(cofm * cofm)
         end if

         !___________________________________________________________________|
         !       Take samples when required                                  |
         !___________________________________________________________________|

         If (Nsamples.Gt.0) Then
            ! ideally it should be delts/2, but owing to precision errors
            ! we keep it slighly greater than 0.5
            If(sim_params%sample_indexes(isample) .eq. step) Then

               Call chain_props (NBeads, R_Bead, F_tot, samples(1:14,isample))

               ! Save information on the error between the predictor
               ! and the final corrector, as well as counts
               samples(15,isample) = pcerr
               samples(18,isample) = lt_count

               If (phys_params%HI_params%delSCalcMethod==Chebyshev) Then
                  samples(19, isample) = ncheb_count
                  samples(20, isample) = fd_err_count
               end if

               !! obtain time correlations
               If (gdots .Eq. 0) Then
                  Call time_correl(NBeads, R_Bead, F_tot, tcur, time, &
                     samples(16,isample))
               End If

               ! Diffusivity
               If (time > 0.0d0) samples(17,isample) = delx2/2/Ndim/time


               time_cdf(isample) = time
               confi(isample, :, :) = R_Bead(:,:)
               grad(isample, :, :) = F_tot(:,:)
               center_of_mass_samples(isample, :) = cofm_cumulative
               isample = isample + 1
            End If
         End If

         !____________________________________________________________________|
         !    Chebyshev polynomial approximation of DelS begins here          |
         !____________________________________________________________________|
         If (phys_params%HI_params%delSCalcMethod==Chebyshev) Then
            call get_dels_approx(NBeads, hstar, Diffusion_sup, Delts, Dels, ncheb_count, fd_err_count)
         Else If (phys_params%HI_params%delSCalcMethod==Cholesky) Then
            call get_delS_cholesky(NBeads, hstar, Diffusion_sup, Delts, Dels)
         Else If (phys_params%HI_params%delSCalcMethod==ExactSqrt) Then
            call get_delS_exact(NBeads, hstar, Diffusion_sup, Delts, Dels)
         End If

         !____________________________________________________________________|
         !        The predictor step                                          |
         !____________________________________________________________________|

         If (Hstar.Gt.0) Then
            alpha = 0.25d0*Delts
            beta = 0.D0
            ! Assigns DR_pred <- 0.25*Delts* D.F

            Call dsymv('U', Ndof,alpha,Diffusion_sup,lda, F_tot,incx, &
               beta,DR_pred,incy)

         Else
            DR_pred = 0.25d0 * Delts * F_tot
         End If


         ! Add the K.R vector
         call get_kappa(time,kappa, phys_params%Flow_inputs)
         DR_pred = DR_pred + Delts * Matmul(kappa,R_Bead)

         ! Eq (14)
         R_pred = R_Bead + DR_pred + sqrt2inv*DelS

         R_pred_old = R_pred

         !____________________________________________________________________|
         !        The first semi-implicit corrector step                      |
         !____________________________________________________________________|

         ! initialise with part of Eq (18)
         Ups_pred = R_Bead + 0.5d0*DR_pred + sqrt2inv*DelS

         If ((phys_params%EV_inputs%excluded_volume_type > 0)&
            .OR.(phys_params%bend_inputs%bending_potential_type.NE.NoBendingPotential)&
            .OR.(phys_params%Flow_inputs%flow_type.EQ.TrapConstVel)) Then
            ! Calculate distances between beads using R_pred
            ! reuse variables
            Call b2bvector_sym_up(NBeads,R_pred,b2b_sup)
            Call modr_sym_up(NBeads,b2b_sup,deltaR_sup)
            Call get_excluded_volume_force(NBeads,b2b_sup,deltaR_sup,F_ev,FlowType)
            Call get_bending_force(F_bend, phys_params%bend_inputs, b2b_sup, deltaR_sup, NBeads)
            call get_harmonic_trap_force(time, phys_params%Flow_inputs, Nbeads, R_pred, F_trap)

            Force_besides_spring = (F_ev+F_bend+F_trap)

            ! Calculate D.FEV using R_pred and update
            If (Hstar.Gt.0) Then
               alpha = 0.125D0*Delts      ! The prefactor is 1/8 and not 1/4
               beta = 1.D0               ! Add to existing
               ! Select Case (DBprec)
               !  Case(SNGL)
               !     Call ssymv('U', Ndof, alpha,Diffusion_sup,lda, F_ev,incx,  &
               !       beta,Ups_pred,incy)
               ! Case(DOBL)
               Call dsymv('U', Ndof, alpha,Diffusion_sup,lda, Force_besides_spring,incx,  &
                  beta,Ups_pred,incy)
               !End Select

            Else
               Ups_pred = Ups_pred + 0.125d0 * Delts * (F_ev + F_bend + F_trap)
            End If
         End If

         !        Calculate the 0.5*Delts*K.R_pred vector

         ! Add the K.R vector
         call get_kappa(time+delts,kappa, phys_params%Flow_inputs)
         Ups_pred = Ups_pred + 0.5_dp * Delts * Matmul(kappa,R_pred)

         ! Eq (18) is completely assembled now

         ! Generating the matrix obtained by using the D_nu
         ! operator on the D super-matrix

         ! (in the following comments indices i,j are omitted for clarity)
         !    diffD(mu,nu) = D(mu+1,nu) - D(mu,nu)
         ! this is true only for nu > mu, since D's elements are computed only
         ! for nu >= mu, and the RHS depends on mu+1.  for nu <= mu+1, the RHS
         ! is rewritten in terms of the symmetric matrix D
         !    diffD(mu,nu) = D(nu,mu+1) - D(nu,mu)
         ! so that all the elements of the RHS are the computed ones

         Forall (mu=1:NBeads-1)
            diffUps(:,mu) = Ups_pred(:,mu+1) - Ups_pred(:,mu)
         End Forall

         Forall (mu=1:NBeads-1, nu=1:Nbeads, nu > mu)
            diffD(:,mu,:,nu) = Diffusion_sup(:,mu+1,:,nu) - Diffusion_sup(:,mu,:,nu)
         End Forall
         Forall (mu=1:NBeads-1, nu=1:Nbeads, nu <= mu)
            diffD(:,mu,:,nu) = Diffusion_sup(:,nu,:,mu+1) - Diffusion_sup(:,nu,:,mu)
         End Forall


         ! Start the loop for solving for connector vectors
         R_corr = R_pred
         ! Prabhakar and Prakash (2004) suggest using R_bead, more numerically stable
         !R_corr = R_bead

         ldadiff = Ndim*(Nbeads-1)

         lt_count = 0

         dtsby4 = Delts/4.0D0

         !     do mu=1,NBeads-1
         !        diffD_reshaped(mu,:,:) = Reshape(DiffD(:,mu,:,:), (/Ndim, Ndim*Nbeads/))
         !     end do

         Keepdoing: Do

            F_con_mu1 = 0.d0



            oversprings: Do mu = 1,NBeads-1

               ! the connector force for this spring is obtained from
               ! Fs(mu) = Fc(mu) - Fc(mu-1)

               F_con_mu = F_con_mu1 + F_spring(:,mu)

               !print *, "F_con_mu", F_con_mu
               ! note:  F_spring contains forces evaluated with the
               !          predictor Q for all beads < mu
               !          corrector Q for all beads > mu

               If ( Hstar.Gt.0.d0 ) Then
                  gama_mu = 0.125d0*Delts* &
                     Matmul( &
                     Reshape(diffD(:,mu,:,:),(/ Ndim, Ndim*Nbeads /) ),  &
                     Reshape(F_spring,             (/ Ndim*Nbeads /)) &
                     )
                  ! Neither of the following optimisations make the code faster on gfortran
                  !call dgemv('N',Ndim, Ndof, 0.125d0*Delts, DiffD(:,mu,:,:), Ndim, F_spring, incx, 0, gama_mu, incy)
                  !                gama_mu = 0.125d0*Delts* &
                  !                   Matmul( &
                  !                   diffD_reshaped(mu,:,:),  &
                  !                   Reshape(F_spring,(/ Ndim*Nbeads /)) &
                  !                   )
                  !                gama_mu =
               Else
                  gama_mu  = 0.125d0*Delts*(F_spring(:,mu+1) - F_spring(:,mu))
               End If

               ! the remaining terms on the RHS of Eq.(20)
               gama_mu = gama_mu + diffUps(:,mu) + 0.25d0 * F_con_mu * Delts

               gama_mag = dnrm2(Ndim,gama_mu,1) ! BLAS single normal two

               ! r = Q_nu_mag/sqrt(b) varies from (0,1)
               ! print *, gama_mag
               Call solve_implicit_r(spring_type,dtsby4,gama_mag/sqrtb,Q0s/sqrtb,r,ff)
               ! implicit solution of Eq.(21)
               !   r ( 1 + deltat/4 ff) -  |gama_mu|/sqrtb == 0
               ! and returns r and ff, the factor in the spring force other
               ! than hookean, F = H Q ff

               If (Abs(r-1._dp) .Lt. 1d-6) r = 1 - 1d-6 ! TODO, y this

               ! unit vector for Q_mu same as for Gama_mu
               gama_mu = gama_mu/gama_mag
               gama_mu = gama_mu*r*sqrtb; ! connector vector update

               !Store results of r
               !Ql_corr(mu) = r*sqrtb

               ! corrector positions vector update
               R_corr(:,mu+1) = R_corr(:,mu) + gama_mu

               ! Connector force
               F_con_mu1 = gama_mu * ff ! to b used for next spring

               ! update the spring forces, and connector force for this spring
               ! which will b used at the start of the loop
               F_spring(:,mu)   = F_spring(:,mu)   + (F_con_mu1 - F_con_mu)
               F_spring(:,mu+1) = F_spring(:,mu+1) - (F_con_mu1 - F_con_mu)


            End Do oversprings


            Rtemp = R_corr - R_pred

            ! root sum of squares of bead vector difference components
            lt_err = dnrm2(Ndof,Rtemp,1)/NBeads

            ! Residuals as defined in Prakash and Prabhakar (2004), but definition of |R| is unclear. Dimensionless
            !lt_err = dnrm2(Ndof,Rtemp,1)/dnrm2(Ndof,R_Bead,1)

            ! Residuals as defined by Larson (J. Chem. Phys 2006, Larson, Hsieh, Jain), straight sum of squares
            !        call Q_mag_from_R_vector(NBeads, R_pred, Ql_pred)
            !        lt_err = sum((Ql_pred - Ql_corr)**2)

            lt_count = lt_count + 1

            !If ((lt_err.Lt.imploop_tol).Or.(lt_count.Gt.2*Nbeads*Nbeads)) Exit
            !If ((lt_err.Lt.imploop_tol)) Exit
            If ((lt_err.Lt.imploop_tol).Or.(lt_count > 100*Nbeads)) Exit

            R_pred = R_corr
         End Do Keepdoing

         if (lt_count > 100*NBeads) then
            write (*,1024) lt_count, Delts , time
1024        format ('Loop exceeded ', I4, ' iterations for dt = ', F11.4, ' at time ', G11.4)
            ! print *, ncheb_count, fd_err_count
         end if

         ! if (ncheb_count > 20) then
         ! print *, ncheb_count, fd_err_count
         ! end if


         ! Final Major updates, new position vector and time
         R_Bead = R_corr
         time = time + Delts
         ! write(*,*) 'after crossing loop', R_bead
         Rtemp = R_Bead - R_pred_old
         pcerr = dnrm2(Ndof,Rtemp,1)/NBeads
         !pcerr = dnrm2(Ndof,Rtemp,1)/dnrm2(Ndof,R_Bead,1)
         ! write(*,*) "AFter 1 time step rtemp", Rtemp
         ! If (ncheb_flag) Then
         ! ncheb = ncheb_old
         ! ncheb_flag = .False.
         ! print *, 'changed ncheb back to prev'
         ! End If

         step = step + 1

      End Do    Overtime

      ! have to re-calculate everything one last time
      cofm = 0.d0
      ! Saving absolute position for Neighbour list
      R_list_save = R_bead
      Do mu = 1, NBeads
         cofm = cofm + R_bead(:,mu)
      End Do

      cofm = cofm/NBeads
      cofm_cumulative = cofm_cumulative + cofm

      ! shift the origin to the current center of mass
      if (sim_params%update_center_of_mass.eq.1) then
         Forall (mu = 1: NBeads)
            R_Bead(:,mu) = R_Bead(:,mu) - cofm
         End Forall
      end if

      ! calculate the bead to bead vector and its magnitude
      ! for use in the forces
      Call b2bvector_sym_up(NBeads,R_Bead,b2b_sup)
      Call modr_sym_up(NBeads,b2b_sup,deltaR_sup)

      ! change the nature of the spring, in the specific subroutine
      ! in get_spring_force(), in module Bead_spring_model
      Call get_spring_force(NBeads,b2b_sup,deltaR_sup,F_spring,phys_params%spring_inputs)
      Call get_excluded_volume_force(NBeads,b2b_sup,deltaR_sup,F_ev,FlowType)
      Call get_bending_force(F_bend, phys_params%bend_inputs, b2b_sup, deltaR_sup, NBeads)
      call get_harmonic_trap_force(time, phys_params%Flow_inputs, Nbeads, R_Bead, F_trap)

      ! Therefore, total force on each bead...
      F_tot = F_spring + F_ev + F_bend + F_trap

      ! Calculate the RPY diffusion tensor
      call get_diffusion_tensor_with_HI(hstar, deltaR_sup, b2b_sup, Diffusion_sup, Nbeads)

      ! mean square displacement of cofm
      ! since the bead positions are centered w.r.t. cofm at every
      ! instant, cofm is the incremental displacement
      if (time .gt. tcur) then
         delx2 = delx2 + Sum(cofm * cofm)
      end if

      !___________________________________________________________________|
      !  Take samples when required, this should really be a subroutine   |
      !___________________________________________________________________|

      If (Nsamples.Gt.0) Then
         ! ideally it should be delts/2, but owing to precision errors
         ! we keep it slighly greater than 0.5
         If(sim_params%sample_indexes(isample) .eq. step) Then

            Call chain_props (NBeads, R_Bead, F_tot, samples(1:14,isample))

            ! Save information on the error between the predictor
            ! and the final corrector, as well as counts
            samples(15,isample) = pcerr
            samples(18,isample) = lt_count

            !! obtain time correlations
            If (gdots .Eq. 0) Then
               Call time_correl(NBeads, R_Bead, F_tot, tcur, time, &
                  samples(16,isample))
            End If

            ! Diffusivity
            If (time > 0.0d0) samples(17,isample) = delx2/2/Ndim/time


            time_cdf(isample) = time
            confi(isample, :, :) = R_Bead(:,:)
            grad(isample, :, :) = F_tot(:,:)
            center_of_mass_samples(isample, :) = cofm_cumulative
            isample = isample + 1
         End If
      End If

      output_variables%chain_configuration_at_sample_points = confi
      output_variables%total_force_at_sample_points = grad
      output_variables%samples = samples
      output_variables%true_times_at_sample_points = time_cdf
      output_variables%center_of_mass_cumulative = center_of_mass_samples

      ! This final step is a bit silly - it's purely a hack put in place to make sure that
      ! the code gives identical bead position outputs to the old regression tests. This is
      ! because the code doesn't update R_corr when we do the final calculations for sampling.

      R_bead = R_corr

   End Subroutine Time_Integrate_Chain

End Module Physics_subroutines
