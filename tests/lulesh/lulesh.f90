! ============================================================================
! VENDORED THIRD-PARTY FIXTURE -- NOT covered by the dace-fortran BSD license.
! This file is GPL-v3 (see the AWE Crown Copyright notice below) and is included
! ONLY as a test fixture for the dace-fortran inliner.
!
!   Upstream : https://github.com/ludgerpaehler/LULESH-Fortran (lulesh.f90)
!   License  : GNU General Public License v3 or later (AWE Crown Copyright 2014)
!
! MODIFICATIONS by the dace-fortran authors (per GPL section 5 -- marking
! changed files), made only to bring the source to standards-conforming,
! parseable Fortran (the upstream is a work-in-progress targeting a patched
! flang):
!   * lulesh.f90: replaced 3 C-style `DO (i=lo,hi)` headers with `DO i=lo,hi`;
!     removed a duplicate `plane,row,col` declaration.
!   * lulesh_comp_kernels.f90: made `m_nodeElemCornerList` rank-1 and rewrote
!     `AllocateNodeElemIndexes` + the two force-gather consumers to the
!     canonical LULESH node->element corner-list algorithm (the upstream left
!     these explicitly broken: "Error in here right now!").
! The driver remains an incomplete WIP (it does not fully compile); it is used
! here only to exercise the inliner's whole-program merge, never executed.
! ============================================================================
!Crown Copyright 2014 AWE.
!
! This file is part of Fortran LULESH.
!
! Fortran LULESH is free software: you can redistribute it and/or modify it under
! the terms of the GNU General Public License as published by the
! Free Software Foundation, either version 3 of the License, or (at your option)
! any later version.
!
! Fortran LULESH is distributed in the hope that it will be useful, but
! WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
! FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
! details.
!
! You should have received a copy of the GNU General Public License along with
! Fortran LULESH. If not, see http://www.gnu.org/licenses/.
!
!
! Authors:
!   Duncan Harris
!   Andy Herdman
!
!
!
!Copyright (c) 2010.
!Lawrence Livermore National Security, LLC.
!Produced at the Lawrence Livermore National Laboratory.
!LLNL-CODE-461231
!All rights reserved.
!
!This file is part of LULESH, Version 1.0.
!Please also read this link -- http://www.opensource.org/licenses/index.php
!
!Redistribution and use in source and binary forms, with or without
!modification, are permitted provided that the following conditions
!are met:
!
!* Redistributions of source code must retain the above copyright
!notice, this list of conditions and the disclaimer below.
!
!* Redistributions in binary form must reproduce the above copyright
!notice, this list of conditions and the disclaimer (as noted below)
!in the documentation and/or other materials provided with the
!distribution.
!
!* Neither the name of the LLNS/LLNL nor the names of its contributors
!may be used to endorse or promote products derived from this software
!without specific prior written permission.
!
!THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
!AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
!IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
!ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC,
!THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
!INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
!BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
!DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
!OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
!NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
!EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
!
!
!Additional BSD Notice

!1. This notice is required to be provided under our contract with the U.S.
!Department of Energy (DOE). This work was produced at Lawrence Livermore
!National Laboratory under Contract No. DE-AC52-07NA27344 with the DOE.
!
!2. Neither the United States Government nor Lawrence Livermore National
!Security, LLC nor any of their employees, makes any warranty, express
!or implied, or assumes any liability or responsibility for the accuracy,
!completeness, or usefulness of any information, apparatus, product, or
!process disclosed, or represents that its use would not infringe
!privately-owned rights.
!
!3. Also, reference herein to any specific commercial products, process, or
!services by trade name, trademark, manufacturer or otherwise does not
!necessarily constitute or imply its endorsement, recommendation, or
!favoring by the United States Government or Lawrence Livermore National
!Security, LLC. The views and opinions of authors expressed herein do not
!necessarily state or reflect those of the United States Government or
!Lawrence Livermore National Security, LLC, and shall not be used for
!advertising or product endorsement purposes.



PROGRAM lulesh

USE lulesh_comp_kernels


IMPLICIT NONE

INTEGER(KIND=4), PARAMETER :: VolumeError = -1
INTEGER(KIND=4), PARAMETER :: QStopError  = -2
INTEGER(KIND=4), PARAMETER :: RLK = 8

! Start of main
TYPE(domain_type) :: domain
!TYPE(domain_type) :: grad_domain  ! Datastruct to store the gradients in  - deactivated for the debugging of the primal
INTEGER :: edgeElems 
INTEGER :: edgeNodes
INTEGER :: opts_its, opts_nx, opts_numReg, opts_showProg,   &
           opts_quiet, opts_viz, opts_balance, opts_cost
REAL(KIND=8) :: opts_numFiles
REAL(KIND=8) :: tx, ty, tz 
INTEGER :: nidx, zidx
INTEGER :: col, row, plane, side
INTEGER :: domElems
INTEGER(KIND=4) :: grad_domElems
INTEGER :: i, j, k
INTEGER :: planeInc, rowInc
REAL(KIND=8),DIMENSION(0:7) :: x_local, y_local, z_local
INTEGER :: gnode, lnode, idx
INTEGER(KIND=4), DIMENSION(:), POINTER :: localNode => NULL()  ! Is this pointer configured correctly
REAL(KIND=8) :: volume
REAL(KIND=8) :: starttim, endtim
REAL(KIND=8) :: elapsed_time

! Initial energy, which is later to be deposited
REAL(KIND=8), PARAMETER :: ebase = 3.948746e+7_RLK
REAL(KIND=8) :: scale, einit


! Needed for boundary conditions
! 2 BCs on each of 6 hexahedral faces (12 bits)
INTEGER, PARAMETER :: XI_M        = z'003' ! 0x003
INTEGER, PARAMETER :: XI_M_SYMM   = z'001' ! 0x001
INTEGER, PARAMETER :: XI_M_FREE   = z'002' ! 0x002

INTEGER, PARAMETER :: XI_P        = z'00c' ! 0x00c
INTEGER, PARAMETER :: XI_P_SYMM   = z'004' ! 0x004
INTEGER, PARAMETER :: XI_P_FREE   = z'008' ! 0x008

INTEGER, PARAMETER :: ETA_M       = z'030' ! 0x030
INTEGER, PARAMETER :: ETA_M_SYMM  = z'010' ! 0x010
INTEGER, PARAMETER :: ETA_M_FREE  = z'020' ! 0x020

INTEGER, PARAMETER :: ETA_P       = z'0c0' ! 0x0c0
INTEGER, PARAMETER :: ETA_P_SYMM  = z'040' ! 0x040
INTEGER, PARAMETER :: ETA_P_FREE  = z'080' ! 0x080

INTEGER, PARAMETER :: ZETA_M      = z'300' ! 0x300
INTEGER, PARAMETER :: ZETA_M_SYMM = z'100' ! 0x100
INTEGER, PARAMETER :: ZETA_M_FREE = z'200' ! 0x200

INTEGER, PARAMETER :: ZETA_P      = z'c00' ! 0xc00
INTEGER, PARAMETER :: ZETA_P_SYMM = z'400' ! 0x400
INTEGER, PARAMETER :: ZETA_P_FREE = z'800' ! 0x800

CHARACTER(len=10) :: arg

INTEGER ::  ElemId

REAL(KIND=8) :: MaxAbsDiff
REAL(KIND=8) :: TotalAbsDiff
REAL(KIND=8) :: MaxRelDiff

REAL(KIND=8) :: AbsDiff, RelDiff

INTEGER :: regionNum, regionVar, binSize, lastReg, elements, &
              runto, costDenominator
INTEGER, DIMENSION(:), ALLOCATABLE :: regBinEnd

ElemId = 0
MaxAbsDiff = 0.0_RLK
TotalAbsDiff = 0.0_RLK
MaxRelDiff = 0.0_RLK

! Symmetry planes have not been set yet
domain%m_symm_is_set = .FALSE.

!CALL GETARG(1, arg)
!READ(arg,*) edgeElems
!edgeElems = 15  ! Fixed for debugging purposes
edgeElems = 2  ! For debugging  - opts.nx
edgeNodes = edgeElems+1
numRanks  = 1   ! Serial execution for now.
myRank    = 0   ! Rank of the executor


! Options as set in the C++ code
! Set defaults that can be overridden by command line args
opts_its      = 9999999
opts_nx       = edgeElems
opts_numReg   = 11
opts_numFiles = (numRanks+10)/9
opts_showProg = 0
opts_quiet    = 0
opts_viz      = 0
opts_balance  = 1
opts_cost     = 1


! Into the domain construction goes:
!  - numRanks
!  - col
!  - row
!  - plane
!  - opts_nx
!  - side
!  - opts_numReg
!  - opts_balance
!  - opts_cost
! C++ call: Domain(numRanks, col, row, plane, opts.nx, side, opts.numReg, opts.balance, opts.cost) ;


! Set up the mesh and decompose. Assumes regular cubes
CALL InitMeshDecomp(numRanks, myRank, col, row, plane, side)


! -----------------------------------------------------
!   Begin the construction of the main data structure
!     and initialize it
! -----------------------------------------------------

! To follow the C++ construction:
edgeElems = opts_nx
edgeNodes = edgeElems+1

! TODO(Ludger & Jan): What is this?!
!this->cost() = cost;

! Store information in the domain
domain%m_tp = tp
domain%m_numRanks = numRanks

! --------------------------------
! Initialize Sedov mesh
! --------------------------------

! Construct a uniform box for the domain
domain%m_sizeX   = edgeElems 
domain%m_sizeY   = edgeElems 
domain%m_sizeZ   = edgeElems 
domain%m_numElem = edgeElems*edgeElems*edgeElems 
domain%m_numNode = edgeNodes*edgeNodes*edgeNodes 

!m_regNumList = new Index_t[numElem()]; // material indexset in C++
ALLOCATE(domain%m_regNumList(0:numElem-1))

! Elem-centered
CALL AllocateElemPersistent(domain, domain%m_numElem)
CALL AllocateElemTemporary (domain, domain%m_numElem) 

! Node-centered
CALL AllocateNodalPersistent(domain, domain%m_numNode) 
CALL AllocateNodesets(domain, edgeNodes*edgeNodes)

!!!!!!!
!TODO(Ludger): SetupCommBuffers needs to be added here.
!!!!!!!

! Basic Field Initialization
DO i=0, numElem-1
   e(i) = 0.0_RLK
   p(i) = 0.0_RLK
   q(i) = 0.0_RLK
   ss(i) = 0.0_RLK
END DO

! v initialized to 1.0
DO i=0, numElem-1
   v(i) = 1.0_RLK
END DO

DO i=0, numNode-1
   xd(i) = 0.0_RLK
   yd(i) = 0.0_RLK
   zd(i) = 0.0_RLK
END DO

DO i=0, numNode-1
   xdd(i) = 0.0_RLK
   ydd(i) = 0.0_RLK
   zdd(i) = 0.0_RLK
END DO

DO i=0, numNode-1
   nodalMass(i) = 0.0_RLK
END DO

! Domain :: BuildMesh
meshEdgeElems = m_tp * opts_nx

nidx = 0
tz = 1.125_RLK * (domain%m_planeLoc*opts_nx) / meshEdgeElems  ! What is domain%m_planeLoc?

DO plane=0, edgeNodes-1
   ty = 1.125_RLK * (domain%m_rowLoc*opts_nx) / meshEdgeElems
   DO row=0, edgeNodes-1
      tx = 1.125_RLK * (domain%m_colLoc*opts_nx) / meshEdgeElems
      DO col=0, edgeNodes-1
         ! Initialize nodal coordinates for the domain
         domain%m_x(nidx) = tx
         domain%m_y(nidx) = ty
         domain%m_z(nidx) = tz

         nidx = nidx+1
         tx = 1.125_RLK * (domain%m_colLoc*opts_nx + col+1) / meshEdgeElems
      END DO
      ty = 1.125_RLK * (domain%m_rowLoc*opts_nx + row+1) / meshEdgeElems
   END DO
   tz = 1.125_RLK * (domain%m_planeLoc*opts_nx + plane+1) / meshEdgeElems
END DO

! embed hexehedral elements in nodal point lattice
nidx = 0
zidx = 0

DO plane=0, edgeElems-1
   DO row=0, edgeElems-1
      DO col=0, edgeElems-1
         localNode => domain%m_nodelist(zidx*8:)
         localNode(0) = nidx
         localNode(1) = nidx                                   + 1
         localNode(2) = nidx                       + edgeNodes + 1
         localNode(3) = nidx                       + edgeNodes
         localNode(4) = nidx + edgeNodes*edgeNodes
         localNode(5) = nidx + edgeNodes*edgeNodes             + 1
         localNode(6) = nidx + edgeNodes*edgeNodes + edgeNodes + 1
         localNode(7) = nidx + edgeNodes*edgeNodes + edgeNodes
         zidx = zidx + 1
         nidx = nidx + 1
      END DO
      nidx = nidx + 1
   END DO
   nidx = nidx + edgeNodes
END DO

!TODO(Ludger): Why was there a nullify here?
!NULLIFY(localNode)

#if _OPENMP
! Setup the thread support structures
numthreads = OMP_GET_MAX_THREADS()

IF (numthreads > 1) THEN
   ALLOCATE(nodeElemCount(0:domain%m_numNode-1))

   DO i=0, domain%m_numNode-1
      nodeElemCount(i) = 0
   END DO

   DO i=0, domain%m_numElem-1
      nl => domain%m_nodelist(i*8)
      DO j=0, 7
         nodeElemCount(nl(j)) = nodeElemCount(nl(j)) + 1
      END DO
   END DO

   ALLOCATE(nodeElemStart(0:domain%m_numNode-1))
   nodeElemStart(0) = 0

   DO i=1, domain%m_numNode
      nodeElemStart(i) = nodeElemStart(i-1) + nodeElemCount(i-1)
   END DO

   ALLOCATE(nodeElemCornerList(0:nodeElemStart(domain%m_numNode)-1))

   DO i=0, domain%m_numNode-1
      nodeElemCount(i) = 0
   END DO

   DO i=0, numElem-1
      nl => domain%m_nodelist(i*8)
      DO j=0, 7
         m = nl(j)
         k = i*8 + j
         offset = nodeElemStart(m) + nodeElemCount(m)
         nodeElemCornerList(offset) = k
         nodeElemCount(m) = nodeElemCount(m) + 1
      END DO
   END DO

   clSize = nodeElemStart(domain%m_numNode)
   DO i=0, clSize-1
      clv = nodeElemCornerList(i)
      IF ((clv.LT.0).OR.(clv.GT.numElem*8))THEN
         PRINT*, "ERROR: clv = ", clv
         PRINT*, "ERROR: clv.LT.0 = ", (clv.LT.0)
         PRINT*, "ERROR: numElem*8 = ", numElem*8
         PRINT*, "ERROR: clv.GT.numElem*8 = ", (clv.GT.numElem*8)
         PRINT*,"AllocateNodeElemIndexes(): nodeElemCornerList entry out of range!"
         CALL luabort(1)
      END IF
   END DO

   DEALLOCATE(nodeElemCount)
ENDIF

#endif

! CreateRegionIndexSets(nr, balance)
! Inputs in our case: opts_numReg, opts_balance

! Equivalent to rand is `rand()`
! binSize = MOD(rand(), 1000) instead of binSize = rand() % 1000

! Setup region index sets. For now, these are constant sized
! throughout the run, but could be changed every cycle to
! simulate effects of ALE on the Lagrange solver
CALL srand(0)
myRank = 0

domain%m_numReg = opts_numReg
!Index_t&  regElemSize(Index_t idx) { return m_regElemSize[idx] ; }
ALLOCATE(m_regElemSize(0:domain%m_numReg-1))
!To store the start of each chunk in regElemlist
ALLOCATE(m_regElemKeys(0:domain%m_numReg))
nextIndex = 0

! If we only have one region just fill it
! Fill out the regNumList with material numbers, which are always
! the region index plus one
IF (domain%m_numReg == 1) THEN
   DO WHILE (nextIndex < domain%m_numElem)
      domain%m_regNumList(nextIndex) = 1
      nextIndex = nextIndex + 1
   END DO
   domain%m_regElemSize(0) = 0
ELSE
   lastReg = -1
   runto = 0
   costDenominator = 0
   ALLOCATE(regBinEnd(0:domain%m_numReg-1))
   ! Determine the relative weight of all regions. This is based off
   ! the -b flag. Balance is the value passed into b.
   DO i=0, domain%m_numReg-1
      domain%m_regElemSize(i) = 0
      costDenominator = costDenominator + (i+1)**opts_balance
      regBinEnd(i) = costDenominator
   END DO

   DO WHILE (nextIndex < domain%m_numElem)
      ! Pick the region
      regionVar = MOD(rand(), costDenominator)
      i = 0
      DO WHILE (regionVar .GE. regBinEnd(i))
         i = i + 1
      END DO

      regionNum = MOD(i + myRank, domain%m_numReg) + 1
      DO WHILE (regionNum .EQ. lastReg)
         regionVar = MOD(rand(), costDenominator)
         i = 0
         DO WHILE (regionVar .GE. regBinEnd(i))
            i = i + 1
         END DO
         regionNum = MOD(i + myRank, domain%m_numReg) + 1
      END DO

      ! Pick the bin size of the region and determine the number of elements.
      binSize = MOD(rand(), 1000)
      IF (binSize .LT. 773) THEN
         elements = MOD(rand(), 15) + 1
      ELSE IF (binSize .LT. 937) THEN
         elements = MOD(rand(), 16) + 16
      ELSE IF (binSize .LT. 970) THEN
         elements = MOD(rand(), 32) + 32
      ELSE IF (binSize .LT. 974) THEN
         elements = MOD(rand(), 64) + 64
      ELSE IF (binSize .LT. 978) THEN
         elements = MOD(rand(), 128) + 128
      ELSE IF (binSize .LT. 981) THEN
         elements = MOD(rand(), 256) + 256
      ELSE
         elements = MOD(rand(), 1537) + 512
         runto = elements + nextIndex
         ! Store the elements. If we hit the end before we run out of
         ! elements then just stop.
         DO WHILE (nextIndex .LT. runto .AND. nextIndex .LT. domain%m_numElem)
            domain%m_regNumList(nextIndex) = regionNum
            nextIndex = nextIndex + 1
         END DO
         lastReg = regionNum
      END IF

      DEALLOCATE(regBinEnd)
   END DO

   ! Convert regNumList to region index sets
   ! First, count size of each region
   DO i=0, domain%m_numElem-1
      r = domain%m_regNumList(i) - 1
      m_regElemSize(r) = m_regElemSize(r) + 1
   END DO

   ! Allocate the region index sets as one big array
   ALLOCATE(m_regElemList(0:SUM(m_regElemSize)))

   ! Get the keys to each region index set
   domain%m_regElemKeys(0) = 0
   DO i=1, domain%m_numElem
      domain%m_regElemKeys(i) = domain%m_regElemKeys(i-1) + m_regElemSize(i-1)
   END DO

   ! Set the array to 0
   m_regElemSize = 0

   ! Third, fill index sets
   DO i=0, domain%m_numElem-1
      r = domain%m_regNumList(i) - 1
      regndx = m_regElemSize(r)
      m_regElemSize(r) = m_regElemSize(r) + 1
      regElemlist(domain%m_regElemKeys(r)+regndx) = i
      ! Index_t r = regNumList(i)-1;       // region index == regnum-1
      ! Index_t regndx = regElemSize(r)++; // Note increment
      ! regElemlist(r,regndx) = i;
   END DO
END IF

STOP






CALL AllocateNodeElemIndexes(domain)

!Create a material IndexSet (entire domain same material for now)
DO i=0, domElems-1
   domain%m_matElemlist(i) = i
END DO

  
! initialize material parameters
  domain%m_dtfixed         = -1.0e-7_RLK
  domain%m_deltatime       =  1.0e-7_RLK
  domain%m_deltatimemultlb =  1.1_RLK
  domain%m_deltatimemultub =  1.2_RLK
  domain%m_stoptime        =  1.0e-2_RLK
  domain%m_dtcourant       =  1.0e+20_RLK
  domain%m_dthydro         =  1.0e+20_RLK
  domain%m_dtmax           =  1.0e-2_RLK
  domain%m_time            =  0.0_RLK
  domain%m_cycle           =  0
  
  domain%m_e_cut = 1.0e-7_RLK
  domain%m_p_cut = 1.0e-7_RLK
  domain%m_q_cut = 1.0e-7_RLK
  domain%m_u_cut = 1.0e-7_RLK
  domain%m_v_cut = 1.0e-10_RLK
  
  domain%m_hgcoef  = 3.0_RLK
  domain%m_ss4o3   =(4.0_RLK)/(3.0_RLK)
  
  domain%m_qstop              = 1.0e+12_RLK
  domain%m_monoq_max_slope    = 1.0_RLK
  domain%m_monoq_limiter_mult = 2.0_RLK
  domain%m_qlc_monoq          = 0.5_RLK
  domain%m_qqc_monoq          = (2.0_RLK)/(3.0_RLK)
  domain%m_qqc                = 2.0_RLK
  
  domain%m_pmin =  0.0_RLK
  domain%m_emin = -1.0e+15_RLK
  
  domain%m_dvovmax =  0.1_RLK
  
  domain%m_eosvmax =  1.0e+9_RLK
  domain%m_eosvmin =  1.0e-9_RLK
  
  domain%m_refdens =  1.0_RLK

 ! initialize field data
 DO i=0, domElems-1
    DO lnode=0,7
       gnode = domain%m_nodelist(i+lnode)  ! The "i*8" here does seem to be wrong - nodelist(i)[lnode] in the original C++ code
       x_local(lnode) = domain%m_x(gnode)
       y_local(lnode) = domain%m_y(gnode)
       z_local(lnode) = domain%m_z(gnode)
    END DO
   
    ! volume calculations
    volume = CalcElemVolume(x_local, y_local, z_local)
    domain%m_volo(i) = volume
    domain%m_elemMass(i) = volume
    DO j=0, 7
       idx = domain%m_nodelist(i*8+j)  ! -> Is this correct, the index is very different to the official C++ index
       domain%m_nodalMass(idx) =  domain%m_nodalMass(idx) + ( volume / 8.0_RLK)
    END DO
 END DO

 ! deposit energy   - They are not applying a scaling here!!
 !domain%m_e(0) = 3.948746e+7

 ! Deposit initial energy
 ! An energy of 3.948746e+7 is correct for a problem with
 ! 45 zones along a side - we need to scale it
 scale = (edgeElems)/45.0_RLK  ! tp is used for the MPI decomp, which is just 1 here.
 einit = ebase*scale*scale*scale
 IF (edgeElems == 45) THEN  ! Copilot
    einit = einit*1.0e+7    ! Copilot
 END IF                     ! Copilot
 
 ! set up symmetry nodesets
 nidx = 0
 
 DO i=0,edgeNodes-1
    planeInc = i*edgeNodes*edgeNodes
    rowInc   = i*edgeNodes
    DO j=0,edgeNodes-1
       domain%m_symmX(nidx) = planeInc + j*edgeNodes
       domain%m_symmY(nidx) = planeInc + j
       domain%m_symmZ(nidx) = rowInc   + j
       nidx=nidx+1
       domain%m_symm_is_set = .TRUE.
    END DO
 END DO

 ! set up elemement connectivity information
 domain%m_lxim(0) = 0  ! Is the index here correct? FORTRAN starts from 1
 DO i=1,domElems-1
    domain%m_lxim(i)   = i-1
    domain%m_lxip(i-1) = i
 END DO

 domain%m_lxip(domElems-1) = domElems ! Should it not be 'domElems-1'?

 DO i=0, edgeElems-1  ! Is the indexing in this loop correct? FORTRAN starts from 1
    domain%m_letam(i)=i
    domain%m_letap(domElems-edgeElems+i) = domElems-edgeElems+i
 END DO

DO i=edgeElems,domElems-1
   domain%m_letam(i) = i-edgeElems
   domain%m_letap(i-edgeElems) = i
END DO

DO i=0,edgeElems*edgeElems-1  ! Is the indexing correct here? Starts at 1, but Fortran-indexing tends to start at 1.
   domain%m_lzetam(i) = i
   domain%m_lzetap(domElems-edgeElems*edgeElems+i) = domElems-edgeElems*edgeElems+i
END DO

DO i=(edgeElems*edgeElems), domElems-1
   domain%m_lzetam(i) = i - edgeElems*edgeElems
   domain%m_lzetap(i-edgeElems*edgeElems) = i
END DO

! set up boundary condition information
domain%m_elemBC = 0 ! clear BCs by default


! ---------------------
! Checked up to this point, but have not fixed actual errors
! ----------------------


! faces on "external" boundaries will be
! symmetry plane or free surface BCs
DO i=0, edgeElems-1
   planeInc = i*edgeElems*edgeElems
   rowInc   = i*edgeElems
   DO j=0,edgeElems-1
      domain%m_elemBC(planeInc+j*edgeElems)                     = &
         IOR(domain%m_elemBC(planeInc+(j)*edgeElems), XI_M_SYMM)
      domain%m_elemBC(planeInc+(j)*edgeElems+edgeElems-1)       = &
         IOR(domain%m_elemBC(planeInc+(j)*edgeElems+edgeElems-1), XI_P_FREE)
      domain%m_elemBC(planeInc+j)                               = &
         IOR(domain%m_elemBC(planeInc+j),ETA_M_SYMM)
      domain%m_elemBC(planeInc+j+edgeElems*edgeElems-edgeElems) = &
         IOR(domain%m_elemBC(planeInc+j+edgeElems*edgeElems-edgeElems),ETA_P_FREE) 
      domain%m_elemBC(rowInc+j)                                 = &
         IOR(domain%m_elemBC(rowInc+j),ZETA_M_SYMM) 
      domain%m_elemBC(rowInc+j+domElems-edgeElems*edgeElems)    = &
         IOR(domain%m_elemBC(rowInc+j+domElems-edgeElems*edgeElems),ZETA_P_FREE)
   END DO
END DO


! timestep to solution
!!$ timeval start, end
!!$ gettimeofday(&start, NULL)
CALL CPU_TIME(starttim)

DO
   call TimeIncrement(domain)
   CALL LagrangeLeapFrog(domain)
   ! CALL LagrangeLeapFrog(grad_domain)
   !CALL __ENZYME_AUTODIFF(LagrangeLeapFrog, domain, grad_domain)

!#ifdef LULESH_SHOW_PROGRESS
!   PRINT *,"time = ", domain%m_time, " dt=",domain%m_deltatime
!#endif

   IF(domain%m_cycle >= 231) EXIT
   !IF(domain%m_time >= domain%m_stoptime) EXIT
END DO


CALL CPU_TIME(endtim)
!!$  gettimeofday(&end, NULL);

elapsed_time = endtim - starttim
!!$  double elapsed_time = double(end.tv_sec - start.tv_sec) + double(end.tv_usec - start.tv_usec) *1e-6;
!!$  

PRINT *,""
PRINT *,""
PRINT '("Elapsed time = ", E13.6)', elapsed_time


ElemId = 0

PRINT *,"Run completed:"
PRINT '("   Problem size        = ", I8)',    edgeElems
PRINT '("   Iteration count     = ", I8)',    domain%m_cycle
PRINT '("   Final Origin Energy = ", e13.6)', domain%m_e(ElemId)
PRINT *,""

  
 MaxAbsDiff = 0.0_RLK
 TotalAbsDiff = 0.0_RLK
 MaxRelDiff = 0.0_RLK


! MIGHT WANT TO DOUBLE CHECK THESE LOOPS
 DO j=0, edgeElems-1
    DO k=j+1, edgeElems-1
       AbsDiff = ABS(domain%m_e(j*edgeElems+k) - domain%m_e(k*edgeElems+j))
       TotalAbsDiff  = TotalAbsDiff+AbsDiff

       if (MaxAbsDiff <AbsDiff) MaxAbsDiff = AbsDiff

       RelDiff = AbsDiff / domain%m_e(k*edgeElems+j)

       if (MaxRelDiff <RelDiff)  MaxRelDiff = RelDiff

    END DO
 END DO

 PRINT *,"  Testing Plane 0 of Energy Array:"
 PRINT '("        MaxAbsDiff   = ", E13.6)', MaxAbsDiff
 PRINT '("        TotalAbsDiff = ", E13.6)', TotalAbsDiff
 PRINT '("        MaxRelDiff   = ", E13.6)', MaxRelDiff
 PRINT *,""

END PROGRAM lulesh



