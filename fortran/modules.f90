!!! Time-stamp: <modules.f90 17:41, 31 Jan 2004 by P Sunthar>

!________________________________________________________________
!   Global modules and interface headers
!________________________________________________________________

!!! $Log: modules.f90,v $
!!! Revision 1.1  2004/01/29 22:13:11  pxs565
!!! Initial revision
!!!


Module Global_parameters_variables_and_types ! Bead Spring simulation, Global constant (parameters)
   !Save

   !Parameters
   Integer, Parameter :: Ndim = 3 ! Dimension of simulation
   Integer, Parameter ::  MaxXdist = 101
   !!! choose precision
   Integer, Parameter :: k4b = Selected_int_kind(9)
   Integer, Parameter :: SNGL = Selected_real_kind(4)
   Integer, Parameter :: DOBL = Selected_real_kind(8)
   ! Integer, Parameter :: DBprec = SNGL
   Integer, Parameter :: DBprec = DOBL
   Integer, Parameter :: dp = DBprec

   Integer, Parameter :: NProps = 21
   Integer, Parameter :: MAXCHEB = 500
   Real (DBprec), Parameter :: PI = 3.14159265358979323846_DOBL
   Real (DBprec), Parameter :: TINI = 1d-25
   Real (DBprec), Parameter :: MYEPS = 1d-6
   ! Spring force law types
   Integer, Parameter :: HOOK = 1, FENE = 2, ILC = 3, WLC = 4, Fraenkel = 5, FENEFraenkel = 6, WLC_bounded=7
                       !Hookean, FENE, Inverse Langevin, Wormlike, Fraenkel, FENE-Fraenkel
   ! Initial condition parameter
   Integer, Parameter :: RandomSpherical = 1, XAxisAligned = 2
   !
   Integer, Parameter :: NoBendingPotential = 0, OneMinusCosTheta = 1

   Integer, Parameter :: NoLookupTable = 0, UseLookupTable = 1

   ! Public, globally accesible names for EV types
   Integer, Parameter, public :: noEV = 0, Gauss = 1, LJ = 2, SDK = 3, SDK_stickers = 4
   
   Integer, Parameter, public :: Chebyshev = 0,  Cholesky= 1, ExactSqrt = 2
   
   Integer, Parameter, public :: EigsFixman = 0,  EigsExact = 1
   
   Integer, Parameter, public :: UpdateChebNew = 0,  UpdateChebAddOne = 1

   !Flow
   ! EQ equilibrium, no flow and no HI
   ! SH Planar shear
   ! UA Uniaxial Elongational
   ! PL Planar Elongation
   ! UR Uniaxial extension followed by relaxation
   ! PU Periodic uniaxial extension
   ! PR 'production' equilibrium with HI and EV
   ! TrapConstVel no flow, but two harmonic traps, one moving at constant velocity
   Integer, Parameter, public :: EQ = 0, SH = 1, UA = 2, PL = 3,  UR = 4, PU = 5, PP = 6, PR = 7, &
                                 TrapConstVel = 8, ShearRelaxation = 9, TransientExtension = 10, ArbitraryFlow = 11



End Module Global_parameters_variables_and_types

