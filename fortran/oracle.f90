! oracle.f90 -- run a reference case with the Fortran core and print the result.
!
! A driver, not a test. It exists so the Fortran physics can be run from this
! repository with no MPI, no NetCDF and no FRUIT: only LAPACK. Each case runs two
! consecutive integrations on ONE continued random stream (the second starting
! from the result of the first, with no reseed) and writes both final
! configurations to stdout at full precision.
!
!   ./fortran_oracle [case] [tolerance] [blocks]
!
!     case      "rouse" (default) or "full"
!     tolerance implicit_loop_exit_tolerance, default 1.0d-6
!     blocks    how many consecutive integrations to run, default 2. The
!               generator is seeded once, so block k is the state after
!               k * steps_per_block steps of one trajectory, and the
!               configuration is printed after each. Raising this shows how the
!               agreement between the two codes evolves over a longer run.
!
! rouse -- the setup of tests.f90::test_rouse_chain_eq: ten Hookean beads, free
!   draining, no flow, seed 5, dt = 0.1, t = 0 -> 1. Reproduces the oracle
!   recorded in tests.f90 and in tests/test_integrator_rouse.cpp.
!
! full -- everything coupled at once, which is what actually exercises the port:
!   fifteen beads, FENE-Fraenkel springs (the cubic implicit solve, with a
!   non-zero natural length), a bending potential, Lennard-Jones excluded volume,
!   hydrodynamic interaction with Cholesky, and simple shear. The starting
!   configuration has overlapping beads (closest pair 0.29 against 2a = 0.89) so
!   the RPY overlapping branch is exercised too. Cholesky rather than Chebyshev
!   on purpose: Chebyshev carries an adaptive fluctuation-dissipation loop whose
!   iteration count can differ between implementations, which would confound an
!   exact comparison.
!
! The tolerance is a command-line argument because the corrector is a
! fixed-point iteration stopped on its increment: at the 1.0d-6 that the recorded
! oracle used, it is still ~3e-8 short of the true solution, so two independent
! implementations cannot agree more closely than that. Running both sides at
! 1.0d-12 removes that floor.
!
! Output is one line per bead per configuration, "x y z" in es26.17.

Program Fortran_Oracle
   Use Global_parameters_variables_and_types   ! Ndim, DBprec, HOOK, noEV, EQ, ...
   Use random_numbers                          ! reset_RNG_with_seed
   Use properties                              ! calculated_variables
   Use Physics_subroutines                     ! physical_parameters,
                                               ! simulation_parameters,
                                               ! Time_Integrate_Chain
   Implicit None

   Type(physical_parameters) :: phys_params
   Type(simulation_parameters) :: sim_params
   Type(calculated_variables) :: output_vars

   Real(DBprec), Allocatable :: PosVecR(:,:)
   Real(DBprec) :: loop_tol
   Integer :: Nbeads, myseed, mu, nblocks, blk
   Character(len=32) :: case_name
   Character(len=64) :: arg

   ! ---- arguments ---------------------------------------------------------
   case_name = 'rouse'
   loop_tol = 1.0d-6
   nblocks = 2
   If (Command_Argument_Count() >= 1) Then
      Call Get_Command_Argument(1, case_name)
   End If
   If (Command_Argument_Count() >= 2) Then
      Call Get_Command_Argument(2, arg)
      Read (arg, *) loop_tol
   End If
   If (Command_Argument_Count() >= 3) Then
      Call Get_Command_Argument(3, arg)
      Read (arg, *) nblocks
   End If

   Select Case (Trim(case_name))
   Case ('rouse')
      Call setup_rouse()
   Case ('full')
      Call setup_full()
   Case ('singular')
      Call setup_singular()
   Case Default
      Write (*, '(A)') 'unknown case "'//Trim(case_name)//'"; use rouse, full or singular'
      Stop 1
   End Select

   sim_params%implicit_loop_exit_tolerance = loop_tol
   sim_params%number_of_samples_to_take = 0
   sim_params%update_center_of_mass = 1
   Allocate (sim_params%sample_indexes(sim_params%number_of_samples_to_take))

   Write (*, '(A,A)')      '# case = ', Trim(case_name)
   Write (*, '(A,ES10.3)') '# implicit_loop_exit_tolerance = ', loop_tol
   Write (*, '(A,I0)')     '# beads = ', Nbeads
   Write (*, '(A,I0)')     '# seed  = ', myseed
   Write (*, '(A,I0)')     '# blocks = ', nblocks
   Write (*, '(A,I0)')     '# steps_per_block = ', &
      Nint((sim_params%time_at_simulation_end - sim_params%time_at_simulation_start) &
           / sim_params%simulation_timestep) + 1

   ! ---- nblocks consecutive integrations on ONE continued stream ----------
   ! The generator is seeded once, before the first block; later blocks carry on
   ! from where the previous one stopped, so block k is the state after
   ! k * steps_per_block steps of a single trajectory. Printing after each gives
   ! the whole growth curve from one run.
   Call reset_RNG_with_seed(myseed)
   Do blk = 1, nblocks
      Call Time_Integrate_Chain(PosVecR, phys_params, sim_params, output_vars)
      Write (*, '(A,I0)') '# ans', blk
      Do mu = 1, Nbeads
         Write (*, '(3ES26.17)') PosVecR(1, mu), PosVecR(2, mu), PosVecR(3, mu)
      End Do
   End Do

   Deallocate (PosVecR)
   Deallocate (sim_params%sample_indexes)

Contains

   ! -----------------------------------------------------------------------
   Subroutine setup_rouse()
      Nbeads = 10
      myseed = 5

      phys_params%number_of_beads = Nbeads
      phys_params%spring_inputs%spring_type = HOOK
      phys_params%spring_inputs%finite_extensibility_parameter = 1000.d0
      phys_params%spring_inputs%natural_length = 0.d0

      phys_params%HI_params%hstar = 0.d0
      phys_params%EV_inputs%excluded_volume_type = noEV
      phys_params%EV_inputs%dimensionless_EV_energy = 0.d0
      phys_params%EV_inputs%dimensionless_EV_radius = 0.d0
      phys_params%bend_inputs%bending_potential_type = NoBendingPotential
      phys_params%bend_inputs%bending_stiffness = 0.d0

      phys_params%Flow_inputs%flow_type = EQ
      phys_params%Flow_inputs%flow_strength = 1.d0

      sim_params%simulation_seed = myseed
      sim_params%time_at_simulation_start = 0.d0
      sim_params%time_at_simulation_end = 1.d0
      sim_params%simulation_timestep = 0.1d0

      Allocate (PosVecR(Ndim, Nbeads))
      PosVecR = Reshape((/ &
         0.000000000000000d00,  0.000000000000000d00,  0.000000000000000d00, &
        -6.586250662803650d-02, -4.911733418703079d-02, 2.791636288166050d-01, &
        -6.753685772418980d-01,  3.519757017493250d-01, -6.340299546718600d-01, &
        -2.844688922166820d00, -2.239686943590640d00, -7.809645235538480d-01, &
        -1.818384915590290d00, -2.201230965554710d00,  9.345296323299410d-01, &
        -3.210149317979810d00, -9.372534379363060d-01, -4.681020081043240d-01, &
        -1.490711003541950d00, -5.971195027232170d-01, -1.509506493806840d00, &
        -1.587433911859990d00, -1.017917685210700d00, -1.456502400338650d00, &
        -6.204553022980690d-01, -1.219155095517640d00, -7.125178799033161d-01, &
        -8.804024234414100d-01, -3.662307761609550d00, -7.348611745983360d-01/), &
         (/Ndim, Nbeads/))
   End Subroutine setup_rouse

   ! -----------------------------------------------------------------------
   Subroutine setup_full()
      Nbeads = 15
      myseed = 11

      phys_params%number_of_beads = Nbeads

      ! FENE-Fraenkel: non-zero natural length, so the implicit solve takes the
      ! cubic branch with a shifted bracket (sigma +/- dQ = 1.5 +/- 3.0).
      phys_params%spring_inputs%spring_type = FENEFraenkel
      phys_params%spring_inputs%finite_extensibility_parameter = 3.d0
      phys_params%spring_inputs%natural_length = 1.5d0

      ! Bending. natural_angles is deliberately left unallocated: the force
      ! routine never dereferences it (see gsipc.f90), matching the C++, which
      ! has no natural-angle input.
      phys_params%bend_inputs%bending_potential_type = OneMinusCosTheta
      phys_params%bend_inputs%bending_stiffness = 1.5d0

      ! Lennard-Jones excluded volume. With a non-EQ flow the Fortran takes the
      ! poor-solvent branch (inteq = 0 -> Rcutp), which is what the C++ does when
      ! PhysParams::equilibration is false.
      phys_params%EV_inputs%excluded_volume_type = LJ
      phys_params%EV_inputs%dimensionless_EV_energy = 1.d0
      phys_params%EV_inputs%dimensionless_EV_radius = 1.d0
      phys_params%EV_inputs%minimum_EV_cutoff = 0.7d0
      phys_params%EV_inputs%maximum_EV_cutoff = 1.5d0
      phys_params%EV_inputs%contour_dist_for_EV = 1

      ! Hydrodynamic interaction. NOTE: the integrator reads HI_params%hstar,
      ! not phys_params%hstar -- setting the latter alone leaves the run free
      ! draining.
      phys_params%HI_params%hstar = 0.25d0
      phys_params%HI_params%delSCalcMethod = Cholesky
      phys_params%HI_params%EigenvalueCalcMethod = EigsFixman
      phys_params%HI_params%ChebUpdateMethod = UpdateChebNew
      phys_params%HI_params%nchebMultiplier = 1.d0
      phys_params%HI_params%fd_err_max = 0.0025d0

      phys_params%Flow_inputs%flow_type = SH
      phys_params%Flow_inputs%flow_strength = 1.d0

      sim_params%simulation_seed = myseed
      sim_params%time_at_simulation_start = 0.d0
      sim_params%time_at_simulation_end = 0.5d0
      sim_params%simulation_timestep = 0.005d0

      ! A fixed starting chain: bonds of 1.4 (inside the FF bracket) in random
      ! directions, centred. Written out as literals rather than generated, so
      ! neither side depends on an RNG or on libm.
      Allocate (PosVecR(Ndim, Nbeads))
      PosVecR = Reshape((/ &
          2.6580494711355812d-01, -1.5153641560908859d+00,  8.8971744579118106d-01, &
         -3.1839438662295377d-01, -2.6456310105703560d+00,  3.0558734373608210d-01, &
         -1.0654694739081119d+00, -2.8105052624338143d+00, -8.6688666238530654d-01, &
         -3.8061825630992757d-01, -3.8254999298223176d+00, -1.8809020178228841d-01, &
         -5.6412438534099429d-01, -2.5987696167518597d+00,  4.6110794947978140d-01, &
         -4.4482844890509982d-01, -1.7507619770038929d+00,  1.5686509610504227d+00, &
          1.0939175038143062d-01, -5.9670721093364754d-01,  1.0020844886808078d+00, &
         -4.4578411617697095d-01,  6.0442809856604995d-01,  5.4486055507256770d-01, &
          2.6495942993909394d-02, -1.6535146869549067d-01, -5.2490155097540081d-01, &
         -9.0605469054047039d-02,  1.1493823693060694d+00, -5.8251580965691385d-02, &
         -2.2299047892838075d-01,  2.4596554567612081d+00,  4.1678383397865953d-01, &
          1.2101068385280689d-01,  2.0211720084425013d+00, -8.6750408761669939d-01, &
          7.0821116000622086d-01,  3.2293922747725059d+00, -4.7329679591009277d-01, &
          1.6916836871696286d+00,  3.5338076714116209d+00, -1.4220396770975463d+00, &
          6.1021684372893303d-01,  2.9107527530423098d+00, -7.8782202105647625d-01/), &
         (/Ndim, Nbeads/))
   End Subroutine setup_full

   ! -----------------------------------------------------------------------
   ! A configuration that makes the Cholesky factorisation fail.
   !
   ! Not synthetic: this is the real state the C++ integrator reached after
   ! 86860 steps of a coarse-grained 10 kbp DNA run, once the implicit corrector
   ! stopped converging and threw the chain apart. The chain has collapsed onto
   ! two points -- 11 beads at one, 10 at the other, 2.615 apart -- with 42 pairs
   ! closer than 1e-6. The RPY tensor built from it has near-duplicate rows, and
   ! its Cholesky factorisation is not positive definite in double precision.
   !
   ! The C++ throws here. Upstream this returns INFO > 0 from dpotrf and says
   ! nothing; with the local modification in gsipc.f90 it prints a warning. The
   ! point of the case is to check that both codes detect the same failure at the
   ! same place.
   !
   ! Only the positions and h* matter: the diffusion tensor does not depend on
   ! the spring law, so the rest is kept minimal.
   Subroutine setup_singular()
      Nbeads = 21
      myseed = 6

      phys_params%number_of_beads = Nbeads
      phys_params%spring_inputs%spring_type = HOOK
      phys_params%spring_inputs%finite_extensibility_parameter = 1000.d0
      phys_params%spring_inputs%natural_length = 0.d0

      phys_params%EV_inputs%excluded_volume_type = noEV
      phys_params%bend_inputs%bending_potential_type = NoBendingPotential
      phys_params%bend_inputs%bending_stiffness = 0.d0

      phys_params%HI_params%hstar = 1.1840879657013359d-01
      phys_params%HI_params%delSCalcMethod = Cholesky

      phys_params%Flow_inputs%flow_type = EQ
      phys_params%Flow_inputs%flow_strength = 0.d0

      sim_params%simulation_seed = myseed
      sim_params%time_at_simulation_start = 0.d0
      sim_params%time_at_simulation_end = 0.d0     ! a single step
      sim_params%simulation_timestep = 0.01d0

      Allocate (PosVecR(Ndim, Nbeads))
      PosVecR = Reshape((/ &
          3.3765051942739451d+03,  1.2495217788360864d+04,  3.2061747562144810d+04, &
          3.3762499570561395d+03,  1.2494272722204056d+04,  3.2059322636380763d+04, &
          3.3765051923658079d+03,  1.2495217787834830d+04,  3.2061747562550663d+04, &
          3.3762499568688331d+03,  1.2494272721976819d+04,  3.2059322636489036d+04, &
          3.3765051925696234d+03,  1.2495217788110200d+04,  3.2061747562421893d+04, &
          3.3762499560215897d+03,  1.2494272722002241d+04,  3.2059322636568308d+04, &
          3.3765051927531099d+03,  1.2495217788104001d+04,  3.2061747562404995d+04, &
          3.3762499567625678d+03,  1.2494272721646401d+04,  3.2059322636629000d+04, &
          3.3765051931086446d+03,  1.2495217788420563d+04,  3.2061747562244200d+04, &
          3.3762499567390928d+03,  1.2494272722504036d+04,  3.2059322636297224d+04, &
          3.3765051934387384d+03,  1.2495217789114515d+04,  3.2061747561938999d+04, &
          3.3762499550760122d+03,  1.2494272722478681d+04,  3.2059322636482153d+04, &
          3.3765051926207616d+03,  1.2495217789159849d+04,  3.2061747562007433d+04, &
          3.3762499565629651d+03,  1.2494272723506823d+04,  3.2059322635924946d+04, &
          3.3765051931253115d+03,  1.2495217788218832d+04,  3.2061747562321067d+04, &
          3.3762499577256367d+03,  1.2494272723414620d+04,  3.2059322635838504d+04, &
          3.3765051921330360d+03,  1.2495217787568272d+04,  3.2061747562679047d+04, &
          3.3762499574748831d+03,  1.2494272722704387d+04,  3.2059322636141696d+04, &
          3.3765051921693489d+03,  1.2495217787371244d+04,  3.2061747562752014d+04, &
          3.3762499568103462d+03,  1.2494272722579521d+04,  3.2059322636260305d+04, &
          3.3765051925460757d+03,  1.2495217787632866d+04,  3.2061747562610402d+04/), &
         (/Ndim, Nbeads/))
   End Subroutine setup_singular

End Program Fortran_Oracle
