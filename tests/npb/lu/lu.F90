MODULE lu

      implicit none

PRIVATE

      INTEGER isiz1, isiz2, isiz3
      parameter (isiz1=33, isiz2=33, isiz3=33)

      INTEGER ipr_default
      parameter (ipr_default = 1)
      DOUBLE PRECISION omega_default
      parameter (omega_default = 1.2d0)
      DOUBLE PRECISION tolrsd1_def, tolrsd2_def, tolrsd3_def, tolrsd4_def, tolrsd5_def
      parameter (tolrsd1_def=1.0d-08, tolrsd2_def=1.0d-08, tolrsd3_def=1.0d-08, tolrsd4_def=1.0d-08, tolrsd5_def=1.0d-08)

      DOUBLE PRECISION c1, c2, c3, c4, c5
      parameter( c1 = 1.40d0, c2 = 0.40d0, c3 = 0.10d0, c4 = 1.00d0, c5 = 1.40d0 )

      INTEGER nx, ny, nz
      INTEGER nx0, ny0, nz0
      INTEGER ist, iend
      INTEGER jst, jend
      INTEGER ii1, ii2
      INTEGER ji1, ji2
      INTEGER ki1, ki2
      DOUBLE PRECISION dxi, deta, dzeta
      DOUBLE PRECISION tx1, tx2, tx3
      DOUBLE PRECISION ty1, ty2, ty3
      DOUBLE PRECISION tz1, tz2, tz3

      !common/cgcon/ dxi, deta, dzeta, tx1, tx2, tx3, ty1, ty2, ty3, tz1, tz2, tz3, nx, ny, nz, nx0, ny0, nz0, ist, iend, jst, jend, ii1, ii2, ji1, ji2, ki1, ki2

      DOUBLE PRECISION dx1, dx2, dx3, dx4, dx5
      DOUBLE PRECISION dy1, dy2, dy3, dy4, dy5
      DOUBLE PRECISION dz1, dz2, dz3, dz4, dz5
      DOUBLE PRECISION dssp

      !common/disp/ dx1,dx2,dx3,dx4,dx5,dy1,dy2,dy3,dy4,dy5,dz1,dz2,dz3,dz4,dz5,dssp

      DOUBLE PRECISION u(5,isiz1/2*2+1,isiz2/2*2+1,isiz3)
      DOUBLE PRECISION rsd(5,isiz1/2*2+1,isiz2/2*2+1,isiz3)
      DOUBLE PRECISION frct(5,isiz1/2*2+1,isiz2/2*2+1,isiz3)
      DOUBLE PRECISION flux(5,isiz1)
      DOUBLE PRECISION qs(isiz1/2*2+1,isiz2/2*2+1,isiz3)
      DOUBLE PRECISION rho_i(isiz1/2*2+1,isiz2/2*2+1,isiz3)

      !common/cvar/ u, rsd, frct, flux,qs, rho_i

      INTEGER ipr, inorm

      !common/cprcon/ ipr, inorm

      INTEGER itmax, invert
      DOUBLE PRECISION  dt, omega, tolrsd(5), rsdnm(5), errnm(5), frc, ttotal

      !common/ctscon/ dt, omega, tolrsd, rsdnm, errnm, frc, ttotal, itmax, invert

      DOUBLE PRECISION a(5,5,isiz1/2*2+1,isiz2), b(5,5,isiz1/2*2+1,isiz2),c(5,5,isiz1/2*2+1,isiz2),d(5,5,isiz1/2*2+1,isiz2)

      !common/cjac/ a, b, c, d

      DOUBLE PRECISION ce(5,13)

      !common/cexact/ ce

      PUBLIC :: dolu
      ! Expose the SSOR solver's configuration inputs and convergence
      ! state so a reference-vs-SDFG numerical test can (a) seed the
      ! solver with the same NPB-class parameters on both sides and
      ! (b) compare ``rsdnm`` (residual norms set by ``l2norm``) after
      ! ``call_dolu``.  ``dolu()`` reads ``nx0`` / ``ny0`` / ``nz0`` /
      ! ``itmax`` / ``dt`` / ``omega`` / ``tolrsd`` / ``inorm`` as
      ! pre-set module state -- the bridge surfaces these on the SDFG
      ! side via kwargs; this PUBLIC just makes them symbol-accessible
      ! on the gfortran reference side too.  Benchmark behaviour is
      ! unchanged.
      PUBLIC :: nx0, ny0, nz0, itmax, dt, omega, tolrsd, inorm
      PUBLIC :: rsdnm

CONTAINS

      SUBROUTINE dolu()
            call domain()
            call setcoeff()
            call setbv()
            call setiv()
            call erhs()
            call ssor(1)
            call setbv()
            call setiv()
            call ssor(itmax)
      END SUBROUTINE dolu

      SUBROUTINE domain()
            implicit none

            nx = nx0
            ny = ny0
            nz = nz0

            ist = 2
            iend = nx - 1

            jst = 2
            jend = ny - 1

            ii1 = 2
            ii2 = nx0 - 1
            ji1 = 2
            ji2 = ny0 - 2
            ki1 = 3
            ki2 = nz0 - 1

            return
      END SUBROUTINE domain

      SUBROUTINE setcoeff()
            implicit none

            dxi = 1.0d0 / ( nx0 - 1 )
            deta = 1.0d0 / ( ny0 - 1 )
            dzeta = 1.0d0 / ( nz0 - 1 )

            tx1 = 1.0d0 / ( dxi * dxi )
            tx2 = 1.0d0 / ( 2.0d0 * dxi )
            tx3 = 1.0d0 / dxi

            ty1 = 1.0d0 / ( deta * deta )
            ty2 = 1.0d0 / ( 2.0d0 * deta )
            ty3 = 1.0d0 / deta

            tz1 = 1.0d0 / ( dzeta * dzeta )
            tz2 = 1.0d0 / ( 2.0d0 * dzeta )
            tz3 = 1.0d0 / dzeta

            dx1 = 0.75d0
            dx2 = dx1
            dx3 = dx1
            dx4 = dx1
            dx5 = dx1

            dy1 = 0.75d0
            dy2 = dy1
            dy3 = dy1
            dy4 = dy1
            dy5 = dy1

            dz1 = 1.00d0
            dz2 = dz1
            dz3 = dz1
            dz4 = dz1
            dz5 = dz1

            dssp = ( max (dx1, dy1, dz1 ) ) / 4.0d0

            ce(1,1) = 2.0d0
            ce(1,2) = 0.0d0
            ce(1,3) = 0.0d0
            ce(1,4) = 4.0d0
            ce(1,5) = 5.0d0
            ce(1,6) = 3.0d0
            ce(1,7) = 0.50d0
            ce(1,8) = 0.02d0
            ce(1,9) = 0.01d0
            ce(1,10) = 0.03d0
            ce(1,11) = 0.50d0
            ce(1,12) = 0.40d0
            ce(1,13) = 0.30d0

            ce(2,1) = 1.0d0
            ce(2,2) = 0.0d0
            ce(2,3) = 0.0d0
            ce(2,4) = 0.0d0
            ce(2,5) = 1.0d0
            ce(2,6) = 2.0d0
            ce(2,7) = 3.0d0
            ce(2,8) = 0.01d0
            ce(2,9) = 0.03d0
            ce(2,10) = 0.02d0
            ce(2,11) = 0.40d0
            ce(2,12) = 0.30d0
            ce(2,13) = 0.50d0

            ce(3,1) = 2.0d0
            ce(3,2) = 2.0d0
            ce(3,3) = 0.0d0
            ce(3,4) = 0.0d0
            ce(3,5) = 0.0d0
            ce(3,6) = 2.0d0
            ce(3,7) = 3.0d0
            ce(3,8) = 0.04d0
            ce(3,9) = 0.03d0
            ce(3,10) = 0.05d0
            ce(3,11) = 0.30d0
            ce(3,12) = 0.50d0
            ce(3,13) = 0.40d0

            ce(4,1) = 2.0d0
            ce(4,2) = 2.0d0
            ce(4,3) = 0.0d0
            ce(4,4) = 0.0d0
            ce(4,5) = 0.0d0
            ce(4,6) = 2.0d0
            ce(4,7) = 3.0d0
            ce(4,8) = 0.03d0
            ce(4,9) = 0.05d0
            ce(4,10) = 0.04d0
            ce(4,11) = 0.20d0
            ce(4,12) = 0.10d0
            ce(4,13) = 0.30d0

            ce(5,1) = 5.0d0
            ce(5,2) = 4.0d0
            ce(5,3) = 3.0d0
            ce(5,4) = 2.0d0
            ce(5,5) = 0.10d0
            ce(5,6) = 0.40d0
            ce(5,7) = 0.30d0
            ce(5,8) = 0.05d0
            ce(5,9) = 0.04d0
            ce(5,10) = 0.03d0
            ce(5,11) = 0.10d0
            ce(5,12) = 0.30d0
            ce(5,13) = 0.20d0

            return
      END SUBROUTINE setcoeff

      SUBROUTINE setbv()

            implicit none

            INTEGER i, j, k, m
            DOUBLE PRECISION temp1(5), temp2(5)

            do j = 1, ny
                  do i = 1, nx
                        call exact( i, j, 1, temp1 )
                        call exact( i, j, nz, temp2 )
                        do m = 1, 5
                              u( m, i, j, 1 ) = temp1(m)
                              u( m, i, j, nz ) = temp2(m)
                        end do
                  end do
            end do

            do k = 1, nz
                  do i = 1, nx
                        call exact( i, 1, k, temp1 )
                        call exact( i, ny, k, temp2 )
                        do m = 1, 5
                              u( m, i, 1, k ) = temp1(m)
                              u( m, i, ny, k ) = temp2(m)
                        end do
                  end do
            end do

            do k = 1, nz
                  do j = 1, ny
                        call exact( 1, j, k, temp1 )
                        call exact( nx, j, k, temp2 )
                        do m = 1, 5
                              u( m, 1, j, k ) = temp1(m)
                              u( m, nx, j, k ) = temp2(m)
                        end do
                  end do
            end do

            return
      END SUBROUTINE setbv

      SUBROUTINE exact( i, j, k, u000ijk )

            implicit none

            INTEGER i, j, k
            DOUBLE PRECISION u000ijk(*)

            INTEGER m
            DOUBLE PRECISION xi, eta, zeta

            xi  = ( dble ( i - 1 ) ) / ( nx0 - 1 )
            eta  = ( dble ( j - 1 ) ) / ( ny0 - 1 )
            zeta = ( dble ( k - 1 ) ) / ( nz - 1 )


            do m = 1, 5
                  u000ijk(m) = ce(m,1) + (ce(m,2) + (ce(m,5) + (ce(m,8) + ce(m,11) * xi) * xi) * xi) * xi + (ce(m,3) + (ce(m,6) + (ce(m,9) + ce(m,12) * eta) * eta) * eta) * eta + (ce(m,4) + (ce(m,7) + (ce(m,10) + ce(m,13) * zeta) * zeta) * zeta) * zeta
            end do

            return
      END SUBROUTINE exact

      SUBROUTINE setiv()

            implicit none

            INTEGER i, j, k, m
            DOUBLE PRECISION  xi, eta, zeta
            DOUBLE PRECISION  pxi, peta, pzeta
            DOUBLE PRECISION  ue_1jk(5),ue_nx0jk(5),ue_i1k(5), ue_iny0k(5),ue_ij1(5),ue_ijnz(5)


            do k = 2, nz - 1
                  zeta = ( dble (k-1) ) / (nz-1)
                  do j = 2, ny - 1
                        eta = ( dble (j-1) ) / (ny0-1)
                        do i = 2, nx - 1
                              xi = ( dble (i-1) ) / (nx0-1)
                              call exact (1,j,k,ue_1jk)
                              call exact (nx0,j,k,ue_nx0jk)
                              call exact (i,1,k,ue_i1k)
                              call exact (i,ny0,k,ue_iny0k)
                              call exact (i,j,1,ue_ij1)
                              call exact (i,j,nz,ue_ijnz)
                              do m = 1, 5
                                    pxi =   ( 1.0d0 - xi ) * ue_1jk(m) + xi   * ue_nx0jk(m)
                                    peta =  ( 1.0d0 - eta ) * ue_i1k(m) + eta   * ue_iny0k(m)
                                    pzeta = ( 1.0d0 - zeta ) * ue_ij1(m) + zeta   * ue_ijnz(m)

                                    u( m, i, j, k ) = pxi + peta + pzeta - pxi * peta - peta * pzeta - pzeta * pxi + pxi * peta * pzeta
                              end do
                        end do
                  end do
            end do

            return
      END SUBROUTINE setiv
      
      SUBROUTINE erhs()

            implicit none

            INTEGER i, j, k, m
            DOUBLE PRECISION  xi, eta, zeta
            DOUBLE PRECISION  q
            DOUBLE PRECISION  u21, u31, u41
            DOUBLE PRECISION  tmp
            DOUBLE PRECISION  u21i, u31i, u41i, u51i
            DOUBLE PRECISION  u21j, u31j, u41j, u51j
            DOUBLE PRECISION  u21k, u31k, u41k, u51k
            DOUBLE PRECISION  u21im1, u31im1, u41im1, u51im1
            DOUBLE PRECISION  u21jm1, u31jm1, u41jm1, u51jm1
            DOUBLE PRECISION  u21km1, u31km1, u41km1, u51km1


            do k = 1, nz
                  do j = 1, ny
                        do i = 1, nx
                              do m = 1, 5
                                    frct( m, i, j, k ) = 0.0d0
                              end do
                        end do
                  end do
            end do

            do k = 1, nz
                  zeta = ( dble(k-1) ) / ( nz - 1 )
                  do j = 1, ny
                        eta = ( dble(j-1) ) / ( ny0 - 1 )
                        do i = 1, nx
                              xi = ( dble(i-1) ) / ( nx0 - 1 )
                              do m = 1, 5
                                    rsd(m,i,j,k) = ce(m,1) + (ce(m,2) + (ce(m,5) + (ce(m,8) + ce(m,11) * xi) * xi) * xi) * xi + (ce(m,3) + (ce(m,6) + (ce(m,9) + ce(m,12) * eta) * eta) * eta) * eta + (ce(m,4) + (ce(m,7) + (ce(m,10) + ce(m,13) * zeta) * zeta) * zeta) * zeta
                              end do
                        end do
                  end do
            end do

            do k = 2, nz - 1
                  do j = jst, jend
                        do i = 1, nx
                              flux(1,i) = rsd(2,i,j,k)
                              u21 = rsd(2,i,j,k) / rsd(1,i,j,k)
                              q = 0.50d0 * (rsd(2,i,j,k) * rsd(2,i,j,k) + rsd(3,i,j,k) * rsd(3,i,j,k) + rsd(4,i,j,k) * rsd(4,i,j,k)) / rsd(1,i,j,k)
                              flux(2,i) = rsd(2,i,j,k) * u21 + c2 * ( rsd(5,i,j,k) - q )
                              flux(3,i) = rsd(3,i,j,k) * u21
                              flux(4,i) = rsd(4,i,j,k) * u21
                              flux(5,i) = ( c1 * rsd(5,i,j,k) - c2 * q ) * u21
                        end do

                        do i = ist, iend
                              do m = 1, 5
                                    frct(m,i,j,k) =  frct(m,i,j,k) - tx2 * ( flux(m,i+1) - flux(m,i-1) )
                              end do
                        end do

                        do i = ist, nx
                              tmp = 1.0d0 / rsd(1,i,j,k)

                              u21i = tmp * rsd(2,i,j,k)
                              u31i = tmp * rsd(3,i,j,k)
                              u41i = tmp * rsd(4,i,j,k)
                              u51i = tmp * rsd(5,i,j,k)

                              tmp = 1.0d0 / rsd(1,i-1,j,k)

                              u21im1 = tmp * rsd(2,i-1,j,k)
                              u31im1 = tmp * rsd(3,i-1,j,k)
                              u41im1 = tmp * rsd(4,i-1,j,k)
                              u51im1 = tmp * rsd(5,i-1,j,k)

                              flux(2,i) = (4.0d0/3.0d0) * tx3 * ( u21i - u21im1 )
                              flux(3,i) = tx3 * ( u31i - u31im1 )
                              flux(4,i) = tx3 * ( u41i - u41im1 )
                              flux(5,i) = 0.50d0 * ( 1.0d0 - c1*c5 ) * tx3 * ( ( u21i  **2 + u31i  **2 + u41i  **2 ) - ( u21im1**2 + u31im1**2 + u41im1**2 ) ) + (1.0d0/6.0d0) * tx3 * ( u21i**2 - u21im1**2 ) + c1 * c5 * tx3 * ( u51i - u51im1 )
                        end do

                        do i = ist, iend
                              frct(1,i,j,k) = frct(1,i,j,k) + dx1 * tx1 * (rsd(1,i-1,j,k)- 2.0d0 * rsd(1,i,j,k) + rsd(1,i+1,j,k) )
                              frct(2,i,j,k) = frct(2,i,j,k) + tx3 * c3 * c4 * ( flux(2,i+1) - flux(2,i) ) + dx2 * tx1 * (rsd(2,i-1,j,k)- 2.0d0 * rsd(2,i,j,k)+ rsd(2,i+1,j,k) )
                              frct(3,i,j,k) = frct(3,i,j,k) + tx3 * c3 * c4 * ( flux(3,i+1) - flux(3,i) ) + dx3 * tx1 * (rsd(3,i-1,j,k) - 2.0d0 * rsd(3,i,j,k) +rsd(3,i+1,j,k) )
                              frct(4,i,j,k) = frct(4,i,j,k) + tx3 * c3 * c4 * ( flux(4,i+1) - flux(4,i) ) + dx4 * tx1 * ( rsd(4,i-1,j,k)  - 2.0d0 * rsd(4,i,j,k)  +rsd(4,i+1,j,k) )
                              frct(5,i,j,k) = frct(5,i,j,k) + tx3 * c3 * c4 * ( flux(5,i+1) - flux(5,i) ) + dx5 * tx1 * ( rsd(5,i-1,j,k) - 2.0d0 * rsd(5,i,j,k) + rsd(5,i+1,j,k) )
                        end do

                        do m = 1, 5
                              frct(m,2,j,k) = frct(m,2,j,k) - dssp * ( + 5.0d0 * rsd(m,2,j,k) - 4.0d0 * rsd(m,3,j,k) + rsd(m,4,j,k) )
                              frct(m,3,j,k) = frct(m,3,j,k) - dssp * ( - 4.0d0 * rsd(m,2,j,k) + 6.0d0 * rsd(m,3,j,k) - 4.0d0 * rsd(m,4,j,k) + rsd(m,5,j,k) )
                        end do

                        do i = 4, nx - 3
                              do m = 1, 5
                                    frct(m,i,j,k) = frct(m,i,j,k) - dssp * (rsd(m,i-2,j,k) - 4.0d0 * rsd(m,i-1,j,k) + 6.0d0 * rsd(m,i,j,k) - 4.0d0 * rsd(m,i+1,j,k) +rsd(m,i+2,j,k) )
                              end do
                        end do

                        do m = 1, 5
                              frct(m,nx-2,j,k) = frct(m,nx-2,j,k) - dssp * (rsd(m,nx-4,j,k) - 4.0d0 * rsd(m,nx-3,j,k) + 6.0d0 * rsd(m,nx-2,j,k) - 4.0d0 * rsd(m,nx-1,j,k)  )
                              frct(m,nx-1,j,k) = frct(m,nx-1,j,k) - dssp * (rsd(m,nx-3,j,k) - 4.0d0 * rsd(m,nx-2,j,k)  + 5.0d0 * rsd(m,nx-1,j,k) )
                        end do

                  end do
            end do

            do k = 2, nz - 1
                  do i = ist, iend
                        do j = 1, ny
                              flux(1,j) = rsd(3,i,j,k)
                              u31 = rsd(3,i,j,k) / rsd(1,i,j,k)
                              q = 0.50d0 * (  rsd(2,i,j,k) * rsd(2,i,j,k)  + rsd(3,i,j,k) * rsd(3,i,j,k) + rsd(4,i,j,k) * rsd(4,i,j,k) ) / rsd(1,i,j,k)
                              flux(2,j) = rsd(2,i,j,k) * u31 
                              flux(3,j) = rsd(3,i,j,k) * u31 + c2 * ( rsd(5,i,j,k) - q )
                              flux(4,j) = rsd(4,i,j,k) * u31
                              flux(5,j) = ( c1 * rsd(5,i,j,k) - c2 * q ) * u31
                        end do

                        do j = jst, jend
                              do m = 1, 5
                                    frct(m,i,j,k) =  frct(m,i,j,k) - ty2 * ( flux(m,j+1) - flux(m,j-1) )
                              end do
                        end do

                        do j = jst, ny
                              tmp = 1.0d0 / rsd(1,i,j,k)

                              u21j = tmp * rsd(2,i,j,k)
                              u31j = tmp * rsd(3,i,j,k)
                              u41j = tmp * rsd(4,i,j,k)
                              u51j = tmp * rsd(5,i,j,k)

                              tmp = 1.0d0 / rsd(1,i,j-1,k)

                              u21jm1 = tmp * rsd(2,i,j-1,k)
                              u31jm1 = tmp * rsd(3,i,j-1,k)
                              u41jm1 = tmp * rsd(4,i,j-1,k)
                              u51jm1 = tmp * rsd(5,i,j-1,k)

                              flux(2,j) = ty3 * ( u21j - u21jm1 )
                              flux(3,j) = (4.0d0/3.0d0) * ty3 *  ( u31j - u31jm1 )
                              flux(4,j) = ty3 * ( u41j - u41jm1 )
                              flux(5,j) = 0.50d0 * ( 1.0d0 - c1*c5 )  * ty3 * ( ( u21j  **2 + u31j  **2 + u41j  **2 ) - ( u21jm1**2 + u31jm1**2 + u41jm1**2 ) ) + (1.0d0/6.0d0)* ty3 * ( u31j**2 - u31jm1**2 ) + c1 * c5 * ty3 * ( u51j - u51jm1 )
                        end do

                        do j = jst, jend
                              frct(1,i,j,k) = frct(1,i,j,k) + dy1 * ty1 * ( rsd(1,i,j-1,k) - 2.0d0 * rsd(1,i,j,k) + rsd(1,i,j+1,k) )
                              frct(2,i,j,k) = frct(2,i,j,k) + ty3 * c3 * c4 * ( flux(2,j+1) - flux(2,j) ) + dy2 * ty1 * ( rsd(2,i,j-1,k)  - 2.0d0 * rsd(2,i,j,k) + rsd(2,i,j+1,k) )
                              frct(3,i,j,k) = frct(3,i,j,k)  + ty3 * c3 * c4 * ( flux(3,j+1) - flux(3,j) ) + dy3 * ty1 * ( rsd(3,i,j-1,k) - 2.0d0 * rsd(3,i,j,k)  + rsd(3,i,j+1,k) )
                              frct(4,i,j,k) = frct(4,i,j,k) + ty3 * c3 * c4 * ( flux(4,j+1) - flux(4,j) ) + dy4 * ty1 * ( rsd(4,i,j-1,k)  - 2.0d0 * rsd(4,i,j,k) + rsd(4,i,j+1,k) )
                              frct(5,i,j,k) = frct(5,i,j,k)  + ty3 * c3 * c4 * ( flux(5,j+1) - flux(5,j) ) + dy5 * ty1 * ( rsd(5,i,j-1,k)  - 2.0d0 * rsd(5,i,j,k) + rsd(5,i,j+1,k) )
                        end do

                        do m = 1, 5
                              frct(m,i,2,k) = frct(m,i,2,k) - dssp * ( + 5.0d0 * rsd(m,i,2,k) - 4.0d0 * rsd(m,i,3,k) + rsd(m,i,4,k) )
                              frct(m,i,3,k) = frct(m,i,3,k) - dssp * ( - 4.0d0 * rsd(m,i,2,k) + 6.0d0 * rsd(m,i,3,k)  - 4.0d0 * rsd(m,i,4,k) + rsd(m,i,5,k) )
                        end do

                        do j = 4, ny - 3
                              do m = 1, 5
                                    frct(m,i,j,k) = frct(m,i,j,k) - dssp * (rsd(m,i,j-2,k) - 4.0d0 * rsd(m,i,j-1,k) + 6.0d0 * rsd(m,i,j,k) - 4.0d0 * rsd(m,i,j+1,k) +rsd(m,i,j+2,k) )
                              end do
                        end do

                        do m = 1, 5
                              frct(m,i,ny-2,k) = frct(m,i,ny-2,k) - dssp * ( rsd(m,i,ny-4,k)  - 4.0d0 * rsd(m,i,ny-3,k) + 6.0d0 * rsd(m,i,ny-2,k) - 4.0d0 * rsd(m,i,ny-1,k)  )
                              frct(m,i,ny-1,k) = frct(m,i,ny-1,k)  - dssp * (rsd(m,i,ny-3,k) - 4.0d0 * rsd(m,i,ny-2,k) + 5.0d0 * rsd(m,i,ny-1,k)  )
                        end do

                  end do
            end do

            do j = jst, jend
                  do i = ist, iend
                        do k = 1, nz
                              flux(1,k) = rsd(4,i,j,k)
                              u41 = rsd(4,i,j,k) / rsd(1,i,j,k)
                              q = 0.50d0 * (  rsd(2,i,j,k) * rsd(2,i,j,k)  + rsd(3,i,j,k) * rsd(3,i,j,k) + rsd(4,i,j,k) * rsd(4,i,j,k) ) / rsd(1,i,j,k)
                              flux(2,k) = rsd(2,i,j,k) * u41 
                              flux(3,k) = rsd(3,i,j,k) * u41 
                              flux(4,k) = rsd(4,i,j,k) * u41 + c2 *  ( rsd(5,i,j,k) - q )
                              flux(5,k) = ( c1 * rsd(5,i,j,k) - c2 * q ) * u41
                        end do

                        do k = 2, nz - 1
                              do m = 1, 5
                                    frct(m,i,j,k) =  frct(m,i,j,k) - tz2 * ( flux(m,k+1) - flux(m,k-1) )
                              end do
                        end do

                        do k = 2, nz
                              tmp = 1.0d0 / rsd(1,i,j,k)

                              u21k = tmp * rsd(2,i,j,k)
                              u31k = tmp * rsd(3,i,j,k)
                              u41k = tmp * rsd(4,i,j,k)
                              u51k = tmp * rsd(5,i,j,k)

                              tmp = 1.0d0 / rsd(1,i,j,k-1)

                              u21km1 = tmp * rsd(2,i,j,k-1)
                              u31km1 = tmp * rsd(3,i,j,k-1)
                              u41km1 = tmp * rsd(4,i,j,k-1)
                              u51km1 = tmp * rsd(5,i,j,k-1)

                              flux(2,k) = tz3 * ( u21k - u21km1 )
                              flux(3,k) = tz3 * ( u31k - u31km1 )
                              flux(4,k) = (4.0d0/3.0d0) * tz3 * ( u41k   - u41km1 )
                              flux(5,k) = 0.50d0 * ( 1.0d0 - c1*c5 ) * tz3 * ( ( u21k  **2 + u31k  **2 + u41k  **2 )  - ( u21km1**2 + u31km1**2 + u41km1**2 ) ) + (1.0d0/6.0d0) * tz3 * ( u41k**2 - u41km1**2 ) + c1 * c5 * tz3 * ( u51k - u51km1 )
                        end do

                        do k = 2, nz - 1
                              frct(1,i,j,k) = frct(1,i,j,k) + dz1 * tz1 * ( rsd(1,i,j,k+1) - 2.0d0 * rsd(1,i,j,k) +rsd(1,i,j,k-1) )
                              frct(2,i,j,k) = frct(2,i,j,k) + tz3 * c3 * c4 * ( flux(2,k+1) - flux(2,k) ) + dz2 * tz1 * ( rsd(2,i,j,k+1)  - 2.0d0 * rsd(2,i,j,k)  +rsd(2,i,j,k-1) )
                              frct(3,i,j,k) = frct(3,i,j,k) + tz3 * c3 * c4 * ( flux(3,k+1) - flux(3,k) )  + dz3 * tz1 * ( rsd(3,i,j,k+1) - 2.0d0 * rsd(3,i,j,k)  +rsd(3,i,j,k-1) )
                              frct(4,i,j,k) = frct(4,i,j,k) + tz3 * c3 * c4 * ( flux(4,k+1) - flux(4,k) ) + dz4 * tz1 * ( rsd(4,i,j,k+1) - 2.0d0 * rsd(4,i,j,k) +rsd(4,i,j,k-1) )
                              frct(5,i,j,k) = frct(5,i,j,k) + tz3 * c3 * c4 * ( flux(5,k+1) - flux(5,k) ) + dz5 * tz1 * (rsd(5,i,j,k+1)  - 2.0d0 * rsd(5,i,j,k) + rsd(5,i,j,k-1) )
                        end do

                        do m = 1, 5
                              frct(m,i,j,2) = frct(m,i,j,2) - dssp * ( + 5.0d0 * rsd(m,i,j,2) - 4.0d0 * rsd(m,i,j,3) + rsd(m,i,j,4) )
                              frct(m,i,j,3) = frct(m,i,j,3) - dssp * (- 4.0d0 * rsd(m,i,j,2)+ 6.0d0 * rsd(m,i,j,3) - 4.0d0 * rsd(m,i,j,4) + rsd(m,i,j,5) )
                        end do

                        do k = 4, nz - 3
                              do m = 1, 5
                                    frct(m,i,j,k) = frct(m,i,j,k)- dssp * (rsd(m,i,j,k-2)- 4.0d0 * rsd(m,i,j,k-1) + 6.0d0 * rsd(m,i,j,k) - 4.0d0 * rsd(m,i,j,k+1)+rsd(m,i,j,k+2) )
                              end do
                        end do

                        do m = 1, 5
                              frct(m,i,j,nz-2) = frct(m,i,j,nz-2) - dssp * ( rsd(m,i,j,nz-4)- 4.0d0 * rsd(m,i,j,nz-3)+ 6.0d0 * rsd(m,i,j,nz-2)- 4.0d0 * rsd(m,i,j,nz-1)  )
                              frct(m,i,j,nz-1) = frct(m,i,j,nz-1) - dssp * (rsd(m,i,j,nz-3)- 4.0d0 * rsd(m,i,j,nz-2) + 5.0d0 * rsd(m,i,j,nz-1)  )
                        end do
                  end do
            end do

            return
      END SUBROUTINE erhs
      
      SUBROUTINE ssor(niter)

            implicit none
            INTEGER niter

            INTEGER i, j, k, m, n
            INTEGER istep
            DOUBLE PRECISION  tmp, tv(5*isiz1*isiz2)
            DOUBLE PRECISION  delunm(5)

            tmp = 1.0d0 / ( omega * ( 2.0d0 - omega ) ) 

            do j=1,isiz2
                  do i=1,isiz1
                        do n=1,5
                              do m=1,5
                                    a(m,n,i,j) = 0.0d0
                                    b(m,n,i,j) = 0.0d0
                                    c(m,n,i,j) = 0.0d0
                                    d(m,n,i,j) = 0.0d0
                              enddo
                        enddo
                  enddo
            enddo

            call rhs()
      
            call l2norm( isiz1, isiz2, isiz3, rsd, rsdnm )

            do istep = 1, niter
                  if (mod ( istep, 20) .eq. 0 .or. istep .eq. itmax .or. istep .eq. 1) then
                  endif
            
                  do k = 2, nz - 1
                        do j = jst, jend
                              do i = ist, iend
                                    do m = 1, 5
                                          rsd(m,i,j,k) = dt * rsd(m,i,j,k)
                                    end do
                              end do
                        end do
                  end do
      
                  do k = 2, nz -1 
                        call jacld(k)
            
                        call blts( isiz1, isiz2, isiz3, k, rsd, a, b, c)
                  end do
      
                  do k = nz - 1, 2, -1
                        call jacu(k)

                        call buts( isiz1, isiz2, isiz3, k, rsd, tv, a, b, c)
                  end do
      

                  do k = 2, nz-1
                        do j = jst, jend
                              do i = ist, iend
                                    do m = 1, 5
                                          u( m, i, j, k ) = u(m, i, j, k ) + tmp * rsd(m, i, j, k )
                                    end do
                              end do
                        end do
                  end do
      
                  if ( mod ( istep, inorm ) .eq. 0 ) then
                        call l2norm( isiz1, isiz2, isiz3, rsd, delunm )
                  end if
      
                  call rhs
            
                  if ( ( mod ( istep, inorm ) .eq. 0 ) .or. ( istep .eq. itmax ) ) then
                        call l2norm( isiz1, isiz2, isiz3, rsd, rsdnm )
                  end if

                  if ( ( rsdnm(1) .lt. tolrsd(1) ) .and. ( rsdnm(2) .lt. tolrsd(2) ) .and. ( rsdnm(3) .lt. tolrsd(3) ) .and. ( rsdnm(4) .lt. tolrsd(4) ) .and. ( rsdnm(5) .lt. tolrsd(5) ) ) then
                        return
                  end if
      
            end do
            return
      END SUBROUTINE ssor

      SUBROUTINE rhs()

            implicit none

            INTEGER i, j, k, m
            DOUBLE PRECISION  q
            DOUBLE PRECISION  tmp, utmp(6,isiz3), rtmp(5,isiz3)
            DOUBLE PRECISION  u21, u31, u41
            DOUBLE PRECISION  u21i, u31i, u41i, u51i
            DOUBLE PRECISION  u21j, u31j, u41j, u51j
            DOUBLE PRECISION  u21k, u31k, u41k, u51k
            DOUBLE PRECISION  u21im1, u31im1, u41im1, u51im1
            DOUBLE PRECISION  u21jm1, u31jm1, u41jm1, u51jm1
            DOUBLE PRECISION  u21km1, u31km1, u41km1, u51km1


            do k = 1, nz
                  do j = 1, ny
                        do i = 1, nx
                              do m = 1, 5
                                    rsd(m,i,j,k) = - frct(m,i,j,k)
                              end do
                              tmp = 1.0d0 / u(1,i,j,k)
                              rho_i(i,j,k) = tmp
                              qs(i,j,k) = 0.50d0 * (  u(2,i,j,k) * u(2,i,j,k)  + u(3,i,j,k) * u(3,i,j,k) + u(4,i,j,k) * u(4,i,j,k) ) * tmp
                        end do
                  end do
            end do

            do k = 2, nz - 1
                  do j = jst, jend
                        do i = 1, nx
                              flux(1,i) = u(2,i,j,k)
                              u21 = u(2,i,j,k) * rho_i(i,j,k)

                              q = qs(i,j,k)

                              flux(2,i) = u(2,i,j,k) * u21 + c2 *  ( u(5,i,j,k) - q )
                              flux(3,i) = u(3,i,j,k) * u21
                              flux(4,i) = u(4,i,j,k) * u21
                              flux(5,i) = ( c1 * u(5,i,j,k) - c2 * q ) * u21
                        end do

                        do i = ist, iend
                              do m = 1, 5
                                    rsd(m,i,j,k) =  rsd(m,i,j,k) - tx2 * ( flux(m,i+1) - flux(m,i-1) )
                              end do
                        end do

                        do i = ist, nx
                              tmp = rho_i(i,j,k)

                              u21i = tmp * u(2,i,j,k)
                              u31i = tmp * u(3,i,j,k)
                              u41i = tmp * u(4,i,j,k)
                              u51i = tmp * u(5,i,j,k)

                              tmp = rho_i(i-1,j,k)

                              u21im1 = tmp * u(2,i-1,j,k)
                              u31im1 = tmp * u(3,i-1,j,k)
                              u41im1 = tmp * u(4,i-1,j,k)
                              u51im1 = tmp * u(5,i-1,j,k)

                              flux(2,i) = (4.0d0/3.0d0) * tx3 * (u21i-u21im1)
                              flux(3,i) = tx3 * ( u31i - u31im1 )
                              flux(4,i) = tx3 * ( u41i - u41im1 )
                              flux(5,i) = 0.50d0 * ( 1.0d0 - c1*c5 )  * tx3 * ( ( u21i  **2 + u31i  **2 + u41i  **2 ) - ( u21im1**2 + u31im1**2 + u41im1**2 ) ) + (1.0d0/6.0d0)  * tx3 * ( u21i**2 - u21im1**2 ) + c1 * c5 * tx3 * ( u51i - u51im1 )
                        end do

                        do i = ist, iend
                              rsd(1,i,j,k) = rsd(1,i,j,k) + dx1 * tx1 * ( u(1,i-1,j,k) - 2.0d0 * u(1,i,j,k) +  u(1,i+1,j,k) )
                              rsd(2,i,j,k) = rsd(2,i,j,k) + tx3 * c3 * c4 * ( flux(2,i+1) - flux(2,i) ) + dx2 * tx1 * (  u(2,i-1,j,k) - 2.0d0 * u(2,i,j,k) +  u(2,i+1,j,k) )
                              rsd(3,i,j,k) = rsd(3,i,j,k) + tx3 * c3 * c4 * ( flux(3,i+1) - flux(3,i) )  + dx3 * tx1 * (  u(3,i-1,j,k) - 2.0d0 * u(3,i,j,k)  +  u(3,i+1,j,k) )
                              rsd(4,i,j,k) = rsd(4,i,j,k) + tx3 * c3 * c4 * ( flux(4,i+1) - flux(4,i) )  + dx4 * tx1 * ( u(4,i-1,j,k) - 2.0d0 * u(4,i,j,k) + u(4,i+1,j,k) )
                              rsd(5,i,j,k) = rsd(5,i,j,k) + tx3 * c3 * c4 * ( flux(5,i+1) - flux(5,i) ) + dx5 * tx1 * ( u(5,i-1,j,k) - 2.0d0 * u(5,i,j,k) + u(5,i+1,j,k) )
                        end do

                        do m = 1, 5
                              rsd(m,2,j,k) = rsd(m,2,j,k) - dssp * ( + 5.0d0 * u(m,2,j,k) - 4.0d0 * u(m,3,j,k) +  u(m,4,j,k) )
                              rsd(m,3,j,k) = rsd(m,3,j,k) - dssp * ( - 4.0d0 * u(m,2,j,k) + 6.0d0 * u(m,3,j,k) - 4.0d0 * u(m,4,j,k) + u(m,5,j,k) )
                        end do

                        do i = 4, nx - 3
                              do m = 1, 5
                                    rsd(m,i,j,k) = rsd(m,i,j,k) - dssp * (  u(m,i-2,j,k) - 4.0d0 * u(m,i-1,j,k) + 6.0d0 * u(m,i,j,k) - 4.0d0 * u(m,i+1,j,k) + u(m,i+2,j,k) )
                              end do
                        end do


                        do m = 1, 5
                              rsd(m,nx-2,j,k) = rsd(m,nx-2,j,k) - dssp * ( u(m,nx-4,j,k) - 4.0d0 * u(m,nx-3,j,k) + 6.0d0 * u(m,nx-2,j,k) - 4.0d0 * u(m,nx-1,j,k)  )
                              rsd(m,nx-1,j,k) = rsd(m,nx-1,j,k) - dssp * ( u(m,nx-3,j,k) - 4.0d0 * u(m,nx-2,j,k)  + 5.0d0 * u(m,nx-1,j,k) )
                        end do

                  end do
            end do

            do k = 2, nz - 1
                  do i = ist, iend
                        do j = 1, ny
                              flux(1,j) = u(3,i,j,k)
                              u31 = u(3,i,j,k) * rho_i(i,j,k)

                              q = qs(i,j,k)

                              flux(2,j) = u(2,i,j,k) * u31 
                              flux(3,j) = u(3,i,j,k) * u31 + c2 * (u(5,i,j,k)-q)
                              flux(4,j) = u(4,i,j,k) * u31
                              flux(5,j) = ( c1 * u(5,i,j,k) - c2 * q ) * u31
                        end do

                        do j = jst, jend
                              do m = 1, 5
                                    rsd(m,i,j,k) =  rsd(m,i,j,k) - ty2 * ( flux(m,j+1) - flux(m,j-1) )
                              end do
                        end do

                        do j = jst, ny
                              tmp = rho_i(i,j,k)

                              u21j = tmp * u(2,i,j,k)
                              u31j = tmp * u(3,i,j,k)
                              u41j = tmp * u(4,i,j,k)
                              u51j = tmp * u(5,i,j,k)

                              tmp = rho_i(i,j-1,k)
                              u21jm1 = tmp * u(2,i,j-1,k)
                              u31jm1 = tmp * u(3,i,j-1,k)
                              u41jm1 = tmp * u(4,i,j-1,k)
                              u51jm1 = tmp * u(5,i,j-1,k)

                              flux(2,j) = ty3 * ( u21j - u21jm1 )
                              flux(3,j) = (4.0d0/3.0d0) * ty3 * (u31j-u31jm1)
                              flux(4,j) = ty3 * ( u41j - u41jm1 )
                              flux(5,j) = 0.50d0 * ( 1.0d0 - c1*c5 ) * ty3 * ( ( u21j  **2 + u31j  **2 + u41j  **2 ) - ( u21jm1**2 + u31jm1**2 + u41jm1**2 ) ) + (1.0d0/6.0d0)  * ty3 * ( u31j**2 - u31jm1**2 ) + c1 * c5 * ty3 * ( u51j - u51jm1 )
                        end do

                        do j = jst, jend
                              rsd(1,i,j,k) = rsd(1,i,j,k) + dy1 * ty1 * ( u(1,i,j-1,k) - 2.0d0 * u(1,i,j,k) + u(1,i,j+1,k) )
                              rsd(2,i,j,k) = rsd(2,i,j,k)  + ty3 * c3 * c4 * ( flux(2,j+1) - flux(2,j) ) + dy2 * ty1 * (  u(2,i,j-1,k) - 2.0d0 * u(2,i,j,k) +  u(2,i,j+1,k) )
                              rsd(3,i,j,k) = rsd(3,i,j,k) + ty3 * c3 * c4 * ( flux(3,j+1) - flux(3,j) )  + dy3 * ty1 * ( u(3,i,j-1,k) - 2.0d0 * u(3,i,j,k) + u(3,i,j+1,k) )
                              rsd(4,i,j,k) = rsd(4,i,j,k) + ty3 * c3 * c4 * ( flux(4,j+1) - flux(4,j) ) + dy4 * ty1 * ( u(4,i,j-1,k) - 2.0d0 * u(4,i,j,k) + u(4,i,j+1,k) )
                              rsd(5,i,j,k) = rsd(5,i,j,k) + ty3 * c3 * c4 * ( flux(5,j+1) - flux(5,j) ) + dy5 * ty1 * (u(5,i,j-1,k) - 2.0d0 * u(5,i,j,k)  + u(5,i,j+1,k) )
                        end do
                  end do

                  do i = ist, iend
                        do m = 1, 5
                              rsd(m,i,2,k) = rsd(m,i,2,k) - dssp * ( + 5.0d0 * u(m,i,2,k) - 4.0d0 * u(m,i,3,k)  + u(m,i,4,k) )
                              rsd(m,i,3,k) = rsd(m,i,3,k) - dssp * ( - 4.0d0 * u(m,i,2,k)  + 6.0d0 * u(m,i,3,k)  - 4.0d0 * u(m,i,4,k) + u(m,i,5,k) )
                        end do
                  end do

                  do j = 4, ny - 3
                        do i = ist, iend
                              do m = 1, 5
                                    rsd(m,i,j,k) = rsd(m,i,j,k)  - dssp * ( u(m,i,j-2,k) - 4.0d0 * u(m,i,j-1,k) + 6.0d0 * u(m,i,j,k) - 4.0d0 * u(m,i,j+1,k) + u(m,i,j+2,k) )
                              end do
                        end do
                  end do

                  do i = ist, iend
                        do m = 1, 5
                              rsd(m,i,ny-2,k) = rsd(m,i,ny-2,k) - dssp * ( u(m,i,ny-4,k) - 4.0d0 * u(m,i,ny-3,k)  + 6.0d0 * u(m,i,ny-2,k)  - 4.0d0 * u(m,i,ny-1,k)  )
                              rsd(m,i,ny-1,k) = rsd(m,i,ny-1,k)  - dssp * ( u(m,i,ny-3,k)  - 4.0d0 * u(m,i,ny-2,k) + 5.0d0 * u(m,i,ny-1,k) )
                        end do
                  end do

            end do

            do j = jst, jend
                  do i = ist, iend
                        do k = 1, nz
                              utmp(1,k) = u(1,i,j,k)
                              utmp(2,k) = u(2,i,j,k)
                              utmp(3,k) = u(3,i,j,k)
                              utmp(4,k) = u(4,i,j,k)
                              utmp(5,k) = u(5,i,j,k)
                              utmp(6,k) = rho_i(i,j,k)
                        end do
                        do k = 1, nz
                              flux(1,k) = utmp(4,k)
                              u41 = utmp(4,k) * utmp(6,k)

                              q = qs(i,j,k)

                              flux(2,k) = utmp(2,k) * u41 
                              flux(3,k) = utmp(3,k) * u41 
                              flux(4,k) = utmp(4,k) * u41 + c2 * (utmp(5,k)-q)
                              flux(5,k) = ( c1 * utmp(5,k) - c2 * q ) * u41
                        end do

                        do k = 2, nz - 1
                              do m = 1, 5
                                    rtmp(m,k) =  rsd(m,i,j,k) - tz2 * ( flux(m,k+1) - flux(m,k-1) )
                              end do
                        end do

                        do k = 2, nz
                              tmp = utmp(6,k)

                              u21k = tmp * utmp(2,k)
                              u31k = tmp * utmp(3,k)
                              u41k = tmp * utmp(4,k)
                              u51k = tmp * utmp(5,k)

                              tmp = utmp(6,k-1)

                              u21km1 = tmp * utmp(2,k-1)
                              u31km1 = tmp * utmp(3,k-1)
                              u41km1 = tmp * utmp(4,k-1)
                              u51km1 = tmp * utmp(5,k-1)

                              flux(2,k) = tz3 * ( u21k - u21km1 )
                              flux(3,k) = tz3 * ( u31k - u31km1 )
                              flux(4,k) = (4.0d0/3.0d0) * tz3 * (u41k-u41km1)
                              flux(5,k) = 0.50d0 * ( 1.0d0 - c1*c5 )  * tz3 * ( ( u21k  **2 + u31k  **2 + u41k  **2 ) - ( u21km1**2 + u31km1**2 + u41km1**2 ) )  + (1.0d0/6.0d0) * tz3 * ( u41k**2 - u41km1**2 ) + c1 * c5 * tz3 * ( u51k - u51km1 )
                        end do

                        do k = 2, nz - 1
                              rtmp(1,k) = rtmp(1,k) + dz1 * tz1 * (utmp(1,k-1) - 2.0d0 * utmp(1,k) +  utmp(1,k+1) )
                              rtmp(2,k) = rtmp(2,k) + tz3 * c3 * c4 * ( flux(2,k+1) - flux(2,k) ) + dz2 * tz1 * ( utmp(2,k-1)  - 2.0d0 * utmp(2,k) + utmp(2,k+1) )
                              rtmp(3,k) = rtmp(3,k) + tz3 * c3 * c4 * ( flux(3,k+1) - flux(3,k) ) + dz3 * tz1 * (utmp(3,k-1) - 2.0d0 * utmp(3,k) + utmp(3,k+1) )
                              rtmp(4,k) = rtmp(4,k) + tz3 * c3 * c4 * ( flux(4,k+1) - flux(4,k) ) + dz4 * tz1 * ( utmp(4,k-1) - 2.0d0 * utmp(4,k) + utmp(4,k+1) )
                              rtmp(5,k) = rtmp(5,k) + tz3 * c3 * c4 * ( flux(5,k+1) - flux(5,k) ) + dz5 * tz1 * ( utmp(5,k-1) - 2.0d0 * utmp(5,k) +utmp(5,k+1) )
                        end do

                        do m = 1, 5
                              rsd(m,i,j,2) = rtmp(m,2) - dssp * ( + 5.0d0 * utmp(m,2)- 4.0d0 * utmp(m,3) + utmp(m,4) )
                              rsd(m,i,j,3) = rtmp(m,3) - dssp * ( - 4.0d0 * utmp(m,2) + 6.0d0 * utmp(m,3) - 4.0d0 * utmp(m,4) + utmp(m,5) )
                        end do

                        do k = 4, nz - 3
                              do m = 1, 5
                                    rsd(m,i,j,k) = rtmp(m,k) - dssp * (utmp(m,k-2) - 4.0d0 * utmp(m,k-1) + 6.0d0 * utmp(m,k) - 4.0d0 * utmp(m,k+1) + utmp(m,k+2) )
                              end do
                        end do

                        do m = 1, 5
                              rsd(m,i,j,nz-2) = rtmp(m,nz-2) - dssp * ( utmp(m,nz-4)  - 4.0d0 * utmp(m,nz-3) + 6.0d0 * utmp(m,nz-2) - 4.0d0 * utmp(m,nz-1)  )
                              rsd(m,i,j,nz-1) = rtmp(m,nz-1) - dssp * ( utmp(m,nz-3) - 4.0d0 * utmp(m,nz-2) + 5.0d0 * utmp(m,nz-1) )
                        end do
                  end do
            end do

            return
      END SUBROUTINE rhs

      SUBROUTINE l2norm (ldx, ldy, ldz, v, sum)

            implicit none

            INTEGER ldx, ldy, ldz
            DOUBLE PRECISION  v(5,ldx/2*2+1,ldy/2*2+1,*), sum(5)

            INTEGER i, j, k, m


            do m = 1, 5
                  sum(m) = 0.0d0
            end do

            do k = 2, nz0-1
                  do j = jst, jend
                        do i = ist, iend
                              do m = 1, 5
                                    sum(m) = sum(m) + v(m,i,j,k) * v(m,i,j,k)
                              end do
                        end do
                  end do
            end do

            do m = 1, 5
                  sum(m) = sqrt ( sum(m) / ( (nx0-2)*(ny0-2)*(nz0-2) ) )
            end do

            return
      END SUBROUTINE l2norm

      SUBROUTINE jacld(k)
            implicit none

            INTEGER k
            INTEGER i, j
            DOUBLE PRECISION  r43
            DOUBLE PRECISION  c1345
            DOUBLE PRECISION  c34
            DOUBLE PRECISION  tmp1, tmp2, tmp3



            r43 = ( 4.0d0 / 3.0d0 )
            c1345 = c1 * c3 * c4 * c5
            c34 = c3 * c4

            do j = jst, jend
                  do i = ist, iend
                        tmp1 = rho_i(i,j,k)
                        tmp2 = tmp1 * tmp1
                        tmp3 = tmp1 * tmp2

                        d(1,1,i,j) =  1.0d0 + dt * 2.0d0 * (   tx1 * dx1 + ty1 * dy1 + tz1 * dz1 )
                        d(1,2,i,j) =  0.0d0
                        d(1,3,i,j) =  0.0d0
                        d(1,4,i,j) =  0.0d0
                        d(1,5,i,j) =  0.0d0

                        d(2,1,i,j) = -dt * 2.0d0 * (  tx1 * r43 + ty1 + tz1  ) * c34 * tmp2 * u(2,i,j,k)
                        d(2,2,i,j) =  1.0d0 + dt * 2.0d0 * c34 * tmp1  * (  tx1 * r43 + ty1 + tz1 ) + dt * 2.0d0 * (   tx1 * dx2 + ty1 * dy2  + tz1 * dz2  )
                        d(2,3,i,j) = 0.0d0
                        d(2,4,i,j) = 0.0d0
                        d(2,5,i,j) = 0.0d0

                        d(3,1,i,j) = -dt * 2.0d0 * (  tx1 + ty1 * r43 + tz1  ) * c34 * tmp2 * u(3,i,j,k)
                        d(3,2,i,j) = 0.0d0
                        d(3,3,i,j) = 1.0d0 + dt * 2.0d0 * c34 * tmp1 * (  tx1 + ty1 * r43 + tz1 ) + dt * 2.0d0 * (  tx1 * dx3 + ty1 * dy3 + tz1 * dz3 )
                        d(3,4,i,j) = 0.0d0
                        d(3,5,i,j) = 0.0d0

                        d(4,1,i,j) = -dt * 2.0d0 * (  tx1 + ty1 + tz1 * r43  ) * c34 * tmp2 * u(4,i,j,k)
                        d(4,2,i,j) = 0.0d0
                        d(4,3,i,j) = 0.0d0
                        d(4,4,i,j) = 1.0d0 + dt * 2.0d0 * c34 * tmp1 * (  tx1 + ty1 + tz1 * r43 ) + dt * 2.0d0 * (  tx1 * dx4 + ty1 * dy4 + tz1 * dz4 )
                        d(4,5,i,j) = 0.0d0

                        d(5,1,i,j) = -dt * 2.0d0 * ( ( ( tx1 * ( r43*c34 - c1345 ) + ty1 * ( c34 - c1345 ) + tz1 * ( c34 - c1345 ) ) * ( u(2,i,j,k) ** 2 ) + ( tx1 * ( c34 - c1345 ) + ty1 * ( r43*c34 - c1345 ) + tz1 * ( c34 - c1345 ) ) * ( u(3,i,j,k) ** 2 ) + ( tx1 * ( c34 - c1345 )+ ty1 * ( c34 - c1345 )+ tz1 * ( r43*c34 - c1345 ) ) * ( u(4,i,j,k) ** 2 )) * tmp3 + ( tx1 + ty1 + tz1 ) * c1345 * tmp2 * u(5,i,j,k) )

                        d(5,2,i,j) = dt * 2.0d0 * tmp2 * u(2,i,j,k) * ( tx1 * ( r43*c34 - c1345 ) + ty1 * (     c34 - c1345 ) + tz1 * (     c34 - c1345 ) )
                        d(5,3,i,j) = dt * 2.0d0 * tmp2 * u(3,i,j,k) * ( tx1 * ( c34 - c1345 ) + ty1 * ( r43*c34 -c1345 ) + tz1 * ( c34 - c1345 ) )
                        d(5,4,i,j) = dt * 2.0d0 * tmp2 * u(4,i,j,k) * ( tx1 * ( c34 - c1345 ) + ty1 * ( c34 - c1345 ) + tz1 * ( r43*c34 - c1345 ) )
                        d(5,5,i,j) = 1.0d0 + dt * 2.0d0 * ( tx1  + ty1 + tz1 ) * c1345 * tmp1 + dt * 2.0d0 * (  tx1 * dx5 +  ty1 * dy5 +  tz1 * dz5 )

                        tmp1 = rho_i(i,j,k-1)
                        tmp2 = tmp1 * tmp1
                        tmp3 = tmp1 * tmp2

                        a(1,1,i,j) = - dt * tz1 * dz1
                        a(1,2,i,j) =   0.0d0
                        a(1,3,i,j) =   0.0d0
                        a(1,4,i,j) = - dt * tz2
                        a(1,5,i,j) =   0.0d0

                        a(2,1,i,j) = - dt * tz2 * ( - ( u(2,i,j,k-1)*u(4,i,j,k-1) ) * tmp2 ) - dt * tz1 * ( - c34 * tmp2 * u(2,i,j,k-1) )
                        a(2,2,i,j) = - dt * tz2 * ( u(4,i,j,k-1) * tmp1 ) - dt * tz1 * c34 * tmp1- dt * tz1 * dz2 
                        a(2,3,i,j) = 0.0d0
                        a(2,4,i,j) = - dt * tz2 * ( u(2,i,j,k-1) * tmp1 )
                        a(2,5,i,j) = 0.0d0

                        a(3,1,i,j) = - dt * tz2 * ( - ( u(3,i,j,k-1)*u(4,i,j,k-1) ) * tmp2 ) - dt * tz1 * ( - c34 * tmp2 * u(3,i,j,k-1) )
                        a(3,2,i,j) = 0.0d0
                        a(3,3,i,j) = - dt * tz2 * ( u(4,i,j,k-1) * tmp1 ) - dt * tz1 * ( c34 * tmp1 ) - dt * tz1 * dz3
                        a(3,4,i,j) = - dt * tz2 * ( u(3,i,j,k-1) * tmp1 )
                        a(3,5,i,j) = 0.0d0

                        a(4,1,i,j) = - dt * tz2 * ( - ( u(4,i,j,k-1) * tmp1 ) ** 2 + c2 * qs(i,j,k-1) * tmp1 ) - dt * tz1 * ( - r43 * c34 * tmp2 * u(4,i,j,k-1) )
                        a(4,2,i,j) = - dt * tz2 * ( - c2 * ( u(2,i,j,k-1) * tmp1 ) )
                        a(4,3,i,j) = - dt * tz2 * ( - c2 * ( u(3,i,j,k-1) * tmp1 ) )
                        a(4,4,i,j) = - dt * tz2 * ( 2.0d0 - c2 ) * ( u(4,i,j,k-1) * tmp1 ) - dt * tz1 * ( r43 * c34 * tmp1 ) - dt * tz1 * dz4
                        a(4,5,i,j) = - dt * tz2 * c2

                        a(5,1,i,j) = - dt * tz2 * ( ( c2 * 2.0d0 * qs(i,j,k-1) - c1 * u(5,i,j,k-1) ) * u(4,i,j,k-1) * tmp2 ) - dt * tz1 * ( - ( c34 - c1345 ) * tmp3 * (u(2,i,j,k-1)**2) - ( c34 - c1345 ) * tmp3 * (u(3,i,j,k-1)**2) - ( r43*c34 - c1345 )* tmp3 * (u(4,i,j,k-1)**2) - c1345 * tmp2 * u(5,i,j,k-1) )
                        a(5,2,i,j) = - dt * tz2 * ( - c2 * ( u(2,i,j,k-1)*u(4,i,j,k-1) ) * tmp2 ) - dt * tz1 * ( c34 - c1345 ) * tmp2 * u(2,i,j,k-1)
                        a(5,3,i,j) = - dt * tz2 * ( - c2 * ( u(3,i,j,k-1)*u(4,i,j,k-1) ) * tmp2 ) - dt * tz1 * ( c34 - c1345 ) * tmp2 * u(3,i,j,k-1)
                        a(5,4,i,j) = - dt * tz2 * ( c1 * ( u(5,i,j,k-1) * tmp1 ) - c2 * ( qs(i,j,k-1) * tmp1 + u(4,i,j,k-1)*u(4,i,j,k-1) * tmp2 ) )- dt * tz1 * ( r43*c34 - c1345 ) * tmp2 * u(4,i,j,k-1)
                        a(5,5,i,j) = - dt * tz2 * ( c1 * ( u(4,i,j,k-1) * tmp1 ) ) - dt * tz1 * c1345 * tmp1 - dt * tz1 * dz5

                        tmp1 = rho_i(i,j-1,k)
                        tmp2 = tmp1 * tmp1
                        tmp3 = tmp1 * tmp2

                        b(1,1,i,j) = - dt * ty1 * dy1
                        b(1,2,i,j) =   0.0d0
                        b(1,3,i,j) = - dt * ty2
                        b(1,4,i,j) =   0.0d0
                        b(1,5,i,j) =   0.0d0

                        b(2,1,i,j) = - dt * ty2 * ( - ( u(2,i,j-1,k)*u(3,i,j-1,k) ) * tmp2 ) - dt * ty1 * ( - c34 * tmp2 * u(2,i,j-1,k) )
                        b(2,2,i,j) = - dt * ty2 * ( u(3,i,j-1,k) * tmp1 ) - dt * ty1 * ( c34 * tmp1 ) - dt * ty1 * dy2
                        b(2,3,i,j) = - dt * ty2 * ( u(2,i,j-1,k) * tmp1 )
                        b(2,4,i,j) = 0.0d0
                        b(2,5,i,j) = 0.0d0

                        b(3,1,i,j) = - dt * ty2 * ( - ( u(3,i,j-1,k) * tmp1 ) ** 2 + c2 * ( qs(i,j-1,k) * tmp1 ) ) - dt * ty1 * ( - r43 * c34 * tmp2 * u(3,i,j-1,k) )
                        b(3,2,i,j) = - dt * ty2 * ( - c2 * ( u(2,i,j-1,k) * tmp1 ) )
                        b(3,3,i,j) = - dt * ty2 * ( ( 2.0d0 - c2 ) * ( u(3,i,j-1,k) * tmp1 ) ) - dt * ty1 * ( r43 * c34 * tmp1 ) - dt * ty1 * dy3
                        b(3,4,i,j) = - dt * ty2 * ( - c2 * ( u(4,i,j-1,k) * tmp1 ) )
                        b(3,5,i,j) = - dt * ty2 * c2

                        b(4,1,i,j) = - dt * ty2 * ( - ( u(3,i,j-1,k)*u(4,i,j-1,k) ) * tmp2 ) - dt * ty1 * ( - c34 * tmp2 * u(4,i,j-1,k) )
                        b(4,2,i,j) = 0.0d0
                        b(4,3,i,j) = - dt * ty2 * ( u(4,i,j-1,k) * tmp1 )
                        b(4,4,i,j) = - dt * ty2 * ( u(3,i,j-1,k) * tmp1 ) - dt * ty1 * ( c34 * tmp1 ) - dt * ty1 * dy4
                        b(4,5,i,j) = 0.0d0

                        b(5,1,i,j) = - dt * ty2 * ( ( c2 * 2.0d0 * qs(i,j-1,k) - c1 * u(5,i,j-1,k) ) * ( u(3,i,j-1,k) * tmp2 ) ) - dt * ty1 * ( - (     c34 - c1345 )*tmp3*(u(2,i,j-1,k)**2) - ( r43*c34 - c1345 )*tmp3*(u(3,i,j-1,k)**2) - (     c34 - c1345 )*tmp3*(u(4,i,j-1,k)**2) - c1345*tmp2*u(5,i,j-1,k) )
                        b(5,2,i,j) = - dt * ty2 * ( - c2 * ( u(2,i,j-1,k)*u(3,i,j-1,k) ) * tmp2 ) - dt * ty1 * ( c34 - c1345 ) * tmp2 * u(2,i,j-1,k)
                        b(5,3,i,j) = - dt * ty2 * ( c1 * ( u(5,i,j-1,k) * tmp1 ) - c2 * ( qs(i,j-1,k) * tmp1 + u(3,i,j-1,k)*u(3,i,j-1,k) * tmp2 ) ) - dt * ty1 * ( r43*c34 - c1345 ) * tmp2 * u(3,i,j-1,k)
                        b(5,4,i,j) = - dt * ty2 * ( - c2 * ( u(3,i,j-1,k)*u(4,i,j-1,k) ) * tmp2 ) - dt * ty1 * ( c34 - c1345 ) * tmp2 * u(4,i,j-1,k)
                        b(5,5,i,j) = - dt * ty2 * ( c1 * ( u(3,i,j-1,k) * tmp1 ) ) - dt * ty1 * c1345 * tmp1 - dt * ty1 * dy5

                        tmp1 = rho_i(i-1,j,k)
                        tmp2 = tmp1 * tmp1
                        tmp3 = tmp1 * tmp2

                        c(1,1,i,j) = - dt * tx1 * dx1
                        c(1,2,i,j) = - dt * tx2
                        c(1,3,i,j) =   0.0d0
                        c(1,4,i,j) =   0.0d0
                        c(1,5,i,j) =   0.0d0

                        c(2,1,i,j) = - dt * tx2 * ( - ( u(2,i-1,j,k) * tmp1 ) ** 2 + c2 * qs(i-1,j,k) * tmp1 ) - dt * tx1 * ( - r43 * c34 * tmp2 * u(2,i-1,j,k) )
                        c(2,2,i,j) = - dt * tx2 * ( ( 2.0d0 - c2 ) * ( u(2,i-1,j,k) * tmp1 ) ) - dt * tx1 * ( r43 * c34 * tmp1 ) - dt * tx1 * dx2
                        c(2,3,i,j) = - dt * tx2 * ( - c2 * ( u(3,i-1,j,k) * tmp1 ) )
                        c(2,4,i,j) = - dt * tx2 * ( - c2 * ( u(4,i-1,j,k) * tmp1 ) )
                        c(2,5,i,j) = - dt * tx2 * c2 

                        c(3,1,i,j) = - dt * tx2 * ( - ( u(2,i-1,j,k) * u(3,i-1,j,k) ) * tmp2 ) - dt * tx1 * ( - c34 * tmp2 * u(3,i-1,j,k) )
                        c(3,2,i,j) = - dt * tx2 * ( u(3,i-1,j,k) * tmp1 )
                        c(3,3,i,j) = - dt * tx2 * ( u(2,i-1,j,k) * tmp1 ) - dt * tx1 * ( c34 * tmp1 ) - dt * tx1 * dx3
                        c(3,4,i,j) = 0.0d0
                        c(3,5,i,j) = 0.0d0

                        c(4,1,i,j) = - dt * tx2 * ( - ( u(2,i-1,j,k)*u(4,i-1,j,k) ) * tmp2 ) - dt * tx1 * ( - c34 * tmp2 * u(4,i-1,j,k) )
                        c(4,2,i,j) = - dt * tx2 * ( u(4,i-1,j,k) * tmp1 )
                        c(4,3,i,j) = 0.0d0
                        c(4,4,i,j) = - dt * tx2 * ( u(2,i-1,j,k) * tmp1 ) - dt * tx1 * ( c34 * tmp1 ) - dt * tx1 * dx4
                        c(4,5,i,j) = 0.0d0

                        c(5,1,i,j) = - dt * tx2 * ( ( c2 * 2.0d0 * qs(i-1,j,k) - c1 * u(5,i-1,j,k) ) * u(2,i-1,j,k) * tmp2 ) - dt * tx1 * ( - ( r43*c34 - c1345 ) * tmp3 * ( u(2,i-1,j,k)**2 ) - (     c34 - c1345 ) * tmp3 * ( u(3,i-1,j,k)**2 ) - (     c34 - c1345 ) * tmp3 * ( u(4,i-1,j,k)**2 )- c1345 * tmp2 * u(5,i-1,j,k) )
                        c(5,2,i,j) = - dt * tx2 * ( c1 * ( u(5,i-1,j,k) * tmp1 ) - c2 * ( u(2,i-1,j,k)*u(2,i-1,j,k) * tmp2 + qs(i-1,j,k) * tmp1 ) ) - dt * tx1 * ( r43*c34 - c1345 ) * tmp2 * u(2,i-1,j,k)
                        c(5,3,i,j) = - dt * tx2 * ( - c2 * ( u(3,i-1,j,k)*u(2,i-1,j,k) ) * tmp2 ) - dt * tx1 * (  c34 - c1345 ) * tmp2 * u(3,i-1,j,k)
                        c(5,4,i,j) = - dt * tx2 * ( - c2 * ( u(4,i-1,j,k)*u(2,i-1,j,k) ) * tmp2 ) - dt * tx1 * (  c34 - c1345 ) * tmp2 * u(4,i-1,j,k)
                        c(5,5,i,j) = - dt * tx2 * ( c1 * ( u(2,i-1,j,k) * tmp1 ) ) - dt * tx1 * c1345 * tmp1 - dt * tx1 * dx5
                  end do
            end do

            return
      END SUBROUTINE jacld

      SUBROUTINE blts ( ldmx, ldmy, ldmz, k, v, ldz, ldy, ldx)
            implicit none

            INTEGER ldmx, ldmy, ldmz
            INTEGER k
            DOUBLE PRECISION  v( 5, ldmx/2*2+1, ldmy/2*2+1, *), ldz( 5, 5, ldmx/2*2+1, ldmy), ldy( 5, 5, ldmx/2*2+1, ldmy), ldx( 5, 5, ldmx/2*2+1, ldmy)
            INTEGER i, j, m
            DOUBLE PRECISION  tmp, tmp1
            DOUBLE PRECISION  tmat(5,5), tv(5)



            do j = jst, jend
                  do i = ist, iend
                        do m = 1, 5
                              v( m, i, j, k ) =  v( m, i, j, k ) - omega * (  ldz( m, 1, i, j ) * v( 1, i, j, k-1 ) + ldz( m, 2, i, j ) * v( 2, i, j, k-1 ) + ldz( m, 3, i, j ) * v( 3, i, j, k-1 ) + ldz( m, 4, i, j ) * v( 4, i, j, k-1 ) + ldz( m, 5, i, j ) * v( 5, i, j, k-1 )  )
                        end do
                  end do
            end do


            do j = jst, jend
                  do i = ist, iend
                        do m = 1, 5
                              tv( m ) =  v( m, i, j, k )- omega * ( ldy( m, 1, i, j ) * v( 1, i, j-1, k )+ ldx( m, 1, i, j ) * v( 1, i-1, j, k )+ ldy( m, 2, i, j ) * v( 2, i, j-1, k )+ ldx( m, 2, i, j ) * v( 2, i-1, j, k )+ ldy( m, 3, i, j ) * v( 3, i, j-1, k )+ ldx( m, 3, i, j ) * v( 3, i-1, j, k )+ ldy( m, 4, i, j ) * v( 4, i, j-1, k )+ ldx( m, 4, i, j ) * v( 4, i-1, j, k )+ ldy( m, 5, i, j ) * v( 5, i, j-1, k )+ ldx( m, 5, i, j ) * v( 5, i-1, j, k ) )
                        end do
            
                        do m = 1, 5
                              tmat( m, 1 ) = d( m, 1, i, j )
                              tmat( m, 2 ) = d( m, 2, i, j )
                              tmat( m, 3 ) = d( m, 3, i, j )
                              tmat( m, 4 ) = d( m, 4, i, j )
                              tmat( m, 5 ) = d( m, 5, i, j )
                        end do

                        tmp1 = 1.0d0 / tmat( 1, 1 )
                        tmp = tmp1 * tmat( 2, 1 )
                        tmat( 2, 2 ) =  tmat( 2, 2 )- tmp * tmat( 1, 2 )
                        tmat( 2, 3 ) =  tmat( 2, 3 )- tmp * tmat( 1, 3 )
                        tmat( 2, 4 ) =  tmat( 2, 4 )- tmp * tmat( 1, 4 )
                        tmat( 2, 5 ) =  tmat( 2, 5 )- tmp * tmat( 1, 5 )
                        tv( 2 ) = tv( 2 )- tv( 1 ) * tmp

                        tmp = tmp1 * tmat( 3, 1 )
                        tmat( 3, 2 ) =  tmat( 3, 2 )- tmp * tmat( 1, 2 )
                        tmat( 3, 3 ) =  tmat( 3, 3 )- tmp * tmat( 1, 3 )
                        tmat( 3, 4 ) =  tmat( 3, 4 )- tmp * tmat( 1, 4 )
                        tmat( 3, 5 ) =  tmat( 3, 5 )- tmp * tmat( 1, 5 )
                        tv( 3 ) = tv( 3 )- tv( 1 ) * tmp

                        tmp = tmp1 * tmat( 4, 1 )
                        tmat( 4, 2 ) =  tmat( 4, 2 )- tmp * tmat( 1, 2 )
                        tmat( 4, 3 ) =  tmat( 4, 3 )- tmp * tmat( 1, 3 )
                        tmat( 4, 4 ) =  tmat( 4, 4 )- tmp * tmat( 1, 4 )
                        tmat( 4, 5 ) =  tmat( 4, 5 )- tmp * tmat( 1, 5 )
                        tv( 4 ) = tv( 4 )- tv( 1 ) * tmp

                        tmp = tmp1 * tmat( 5, 1 )
                        tmat( 5, 2 ) =  tmat( 5, 2 )- tmp * tmat( 1, 2 )
                        tmat( 5, 3 ) =  tmat( 5, 3 )- tmp * tmat( 1, 3 )
                        tmat( 5, 4 ) =  tmat( 5, 4 )- tmp * tmat( 1, 4 )
                        tmat( 5, 5 ) =  tmat( 5, 5 )- tmp * tmat( 1, 5 )
                        tv( 5 ) = tv( 5 )- tv( 1 ) * tmp



                        tmp1 = 1.0d0 / tmat( 2, 2 )
                        tmp = tmp1 * tmat( 3, 2 )
                        tmat( 3, 3 ) =  tmat( 3, 3 )- tmp * tmat( 2, 3 )
                        tmat( 3, 4 ) =  tmat( 3, 4 )- tmp * tmat( 2, 4 )
                        tmat( 3, 5 ) =  tmat( 3, 5 )- tmp * tmat( 2, 5 )
                        tv( 3 ) = tv( 3 )- tv( 2 ) * tmp

                        tmp = tmp1 * tmat( 4, 2 )
                        tmat( 4, 3 ) =  tmat( 4, 3 )- tmp * tmat( 2, 3 )
                        tmat( 4, 4 ) =  tmat( 4, 4 )- tmp * tmat( 2, 4 )
                        tmat( 4, 5 ) =  tmat( 4, 5 )- tmp * tmat( 2, 5 )
                        tv( 4 ) = tv( 4 )- tv( 2 ) * tmp

                        tmp = tmp1 * tmat( 5, 2 )
                        tmat( 5, 3 ) =  tmat( 5, 3 )- tmp * tmat( 2, 3 )
                        tmat( 5, 4 ) =  tmat( 5, 4 )- tmp * tmat( 2, 4 )
                        tmat( 5, 5 ) =  tmat( 5, 5 )- tmp * tmat( 2, 5 )
                        tv( 5 ) = tv( 5 )- tv( 2 ) * tmp



                        tmp1 = 1.0d0 / tmat( 3, 3 )
                        tmp = tmp1 * tmat( 4, 3 )
                        tmat( 4, 4 ) =  tmat( 4, 4 )- tmp * tmat( 3, 4 )
                        tmat( 4, 5 ) =  tmat( 4, 5 )- tmp * tmat( 3, 5 )
                        tv( 4 ) = tv( 4 )- tv( 3 ) * tmp

                        tmp = tmp1 * tmat( 5, 3 )
                        tmat( 5, 4 ) =  tmat( 5, 4 )- tmp * tmat( 3, 4 )
                        tmat( 5, 5 ) =  tmat( 5, 5 )- tmp * tmat( 3, 5 )
                        tv( 5 ) = tv( 5 )- tv( 3 ) * tmp



                        tmp1 = 1.0d0 / tmat( 4, 4 )
                        tmp = tmp1 * tmat( 5, 4 )
                        tmat( 5, 5 ) =  tmat( 5, 5 ) - tmp * tmat( 4, 5 )
                        tv( 5 ) = tv( 5 ) - tv( 4 ) * tmp

                        v( 5, i, j, k ) = tv( 5 ) / tmat( 5, 5 )

                        tv( 4 ) = tv( 4 ) - tmat( 4, 5 ) * v( 5, i, j, k )
                        v( 4, i, j, k ) = tv( 4 ) / tmat( 4, 4 )

                        tv( 3 ) = tv( 3 ) - tmat( 3, 4 ) * v( 4, i, j, k )  - tmat( 3, 5 ) * v( 5, i, j, k )
                        v( 3, i, j, k ) = tv( 3 ) / tmat( 3, 3 )

                        tv( 2 ) = tv( 2 )  - tmat( 2, 3 ) * v( 3, i, j, k ) - tmat( 2, 4 ) * v( 4, i, j, k ) - tmat( 2, 5 ) * v( 5, i, j, k )
                        v( 2, i, j, k ) = tv( 2 ) / tmat( 2, 2 )

                        tv( 1 ) = tv( 1 ) - tmat( 1, 2 ) * v( 2, i, j, k ) - tmat( 1, 3 ) * v( 3, i, j, k ) - tmat( 1, 4 ) * v( 4, i, j, k ) - tmat( 1, 5 ) * v( 5, i, j, k )
                        v( 1, i, j, k ) = tv( 1 ) / tmat( 1, 1 )
                  enddo
            enddo

            return
      END SUBROUTINE blts

      SUBROUTINE buts(ldmx, ldmy, ldmz, k, v, tv, udx, udy, udz)
            implicit none

            INTEGER ldmx, ldmy, ldmz
            INTEGER k
            DOUBLE PRECISION v( 5,ldmx/2*2+1, ldmy/2*2+1, *), tv( 5, ldmx/2*2+1, ldmy), udx( 5, 5, ldmx/2*2+1, ldmy), udy( 5, 5, ldmx/2*2+1, ldmy), udz( 5, 5, ldmx/2*2+1, ldmy )

            INTEGER i, j, m
            DOUBLE PRECISION tmp, tmp1
            DOUBLE PRECISION tmat(5,5)

            do j = jend, jst, -1
                  do i = iend, ist, -1
                        do m = 1, 5
                              tv( m, i, j ) =   omega * (  udz( m, 1, i, j ) * v( 1, i, j, k+1 ) + udz( m, 2, i, j ) * v( 2, i, j, k+1 )  + udz( m, 3, i, j ) * v( 3, i, j, k+1 ) + udz( m, 4, i, j ) * v( 4, i, j, k+1 )  + udz( m, 5, i, j ) * v( 5, i, j, k+1 ) )
                        end do
                  end do
            end do


            do j = jend, jst, -1
                  do i = iend, ist, -1
                        do m = 1, 5
                              tv( m, i, j ) = tv( m, i, j ) + omega * ( udy( m, 1, i, j ) * v( 1, i, j+1, k ) + udx( m, 1, i, j ) * v( 1, i+1, j, k )  + udy( m, 2, i, j ) * v( 2, i, j+1, k )  + udx( m, 2, i, j ) * v( 2, i+1, j, k )  + udy( m, 3, i, j ) * v( 3, i, j+1, k ) + udx( m, 3, i, j ) * v( 3, i+1, j, k ) + udy( m, 4, i, j ) * v( 4, i, j+1, k ) + udx( m, 4, i, j ) * v( 4, i+1, j, k ) + udy( m, 5, i, j ) * v( 5, i, j+1, k ) + udx( m, 5, i, j ) * v( 5, i+1, j, k ) )
                        end do

                        do m = 1, 5
                              tmat( m, 1 ) = d( m, 1, i, j )
                              tmat( m, 2 ) = d( m, 2, i, j )
                              tmat( m, 3 ) = d( m, 3, i, j )
                              tmat( m, 4 ) = d( m, 4, i, j )
                              tmat( m, 5 ) = d( m, 5, i, j )
                        end do

                        tmp1 = 1.0d0 / tmat( 1, 1 )
                        tmp = tmp1 * tmat( 2, 1 )
                        tmat( 2, 2 ) =  tmat( 2, 2 ) - tmp * tmat( 1, 2 )
                        tmat( 2, 3 ) =  tmat( 2, 3 )  - tmp * tmat( 1, 3 )
                        tmat( 2, 4 ) =  tmat( 2, 4 ) - tmp * tmat( 1, 4 )
                        tmat( 2, 5 ) =  tmat( 2, 5 ) - tmp * tmat( 1, 5 )
                        tv( 2, i, j ) = tv( 2, i, j ) - tv( 1, i, j ) * tmp

                        tmp = tmp1 * tmat( 3, 1 )
                        tmat( 3, 2 ) =  tmat( 3, 2 ) - tmp * tmat( 1, 2 )
                        tmat( 3, 3 ) =  tmat( 3, 3 ) - tmp * tmat( 1, 3 )
                        tmat( 3, 4 ) =  tmat( 3, 4 ) - tmp * tmat( 1, 4 )
                        tmat( 3, 5 ) =  tmat( 3, 5 ) - tmp * tmat( 1, 5 )
                        tv( 3, i, j ) = tv( 3, i, j )  - tv( 1, i, j ) * tmp

                        tmp = tmp1 * tmat( 4, 1 )
                        tmat( 4, 2 ) =  tmat( 4, 2 ) - tmp * tmat( 1, 2 )
                        tmat( 4, 3 ) =  tmat( 4, 3 ) - tmp * tmat( 1, 3 )
                        tmat( 4, 4 ) =  tmat( 4, 4 ) - tmp * tmat( 1, 4 )
                        tmat( 4, 5 ) =  tmat( 4, 5 ) - tmp * tmat( 1, 5 )
                        tv( 4, i, j ) = tv( 4, i, j ) - tv( 1, i, j ) * tmp

                        tmp = tmp1 * tmat( 5, 1 )
                        tmat( 5, 2 ) =  tmat( 5, 2 )  - tmp * tmat( 1, 2 )
                        tmat( 5, 3 ) =  tmat( 5, 3 ) - tmp * tmat( 1, 3 )
                        tmat( 5, 4 ) =  tmat( 5, 4 ) - tmp * tmat( 1, 4 )
                        tmat( 5, 5 ) =  tmat( 5, 5 ) - tmp * tmat( 1, 5 )
                        tv( 5, i, j ) = tv( 5, i, j ) - tv( 1, i, j ) * tmp



                        tmp1 = 1.0d0 / tmat( 2, 2 )
                        tmp = tmp1 * tmat( 3, 2 )
                        tmat( 3, 3 ) =  tmat( 3, 3 ) - tmp * tmat( 2, 3 )
                        tmat( 3, 4 ) =  tmat( 3, 4 )- tmp * tmat( 2, 4 )
                        tmat( 3, 5 ) =  tmat( 3, 5 ) - tmp * tmat( 2, 5 )
                        tv( 3, i, j ) = tv( 3, i, j ) - tv( 2, i, j ) * tmp

                        tmp = tmp1 * tmat( 4, 2 )
                        tmat( 4, 3 ) =  tmat( 4, 3 ) - tmp * tmat( 2, 3 )
                        tmat( 4, 4 ) =  tmat( 4, 4 ) - tmp * tmat( 2, 4 )
                        tmat( 4, 5 ) =  tmat( 4, 5 ) - tmp * tmat( 2, 5 )
                        tv( 4, i, j ) = tv( 4, i, j ) - tv( 2, i, j ) * tmp

                        tmp = tmp1 * tmat( 5, 2 )
                        tmat( 5, 3 ) =  tmat( 5, 3 )  - tmp * tmat( 2, 3 )
                        tmat( 5, 4 ) =  tmat( 5, 4 ) - tmp * tmat( 2, 4 )
                        tmat( 5, 5 ) =  tmat( 5, 5 ) - tmp * tmat( 2, 5 )
                        tv( 5, i, j ) = tv( 5, i, j ) - tv( 2, i, j ) * tmp



                        tmp1 = 1.0d0 / tmat( 3, 3 )
                        tmp = tmp1 * tmat( 4, 3 )
                        tmat( 4, 4 ) =  tmat( 4, 4 )- tmp * tmat( 3, 4 )
                        tmat( 4, 5 ) =  tmat( 4, 5 )- tmp * tmat( 3, 5 )
                        tv( 4, i, j ) = tv( 4, i, j )- tv( 3, i, j ) * tmp

                        tmp = tmp1 * tmat( 5, 3 )
                        tmat( 5, 4 ) =  tmat( 5, 4 )- tmp * tmat( 3, 4 )
                        tmat( 5, 5 ) =  tmat( 5, 5 )- tmp * tmat( 3, 5 )
                        tv( 5, i, j ) = tv( 5, i, j )- tv( 3, i, j ) * tmp



                        tmp1 = 1.0d0 / tmat( 4, 4 )
                        tmp = tmp1 * tmat( 5, 4 )
                        tmat( 5, 5 ) =  tmat( 5, 5 ) - tmp * tmat( 4, 5 )
                        tv( 5, i, j ) = tv( 5, i, j )- tv( 4, i, j ) * tmp

                        tv( 5, i, j ) = tv( 5, i, j )/ tmat( 5, 5 )

                        tv( 4, i, j ) = tv( 4, i, j )- tmat( 4, 5 ) * tv( 5, i, j )
                        tv( 4, i, j ) = tv( 4, i, j )/ tmat( 4, 4 )

                        tv( 3, i, j ) = tv( 3, i, j )- tmat( 3, 4 ) * tv( 4, i, j )- tmat( 3, 5 ) * tv( 5, i, j )
                        tv( 3, i, j ) = tv( 3, i, j )/ tmat( 3, 3 )

                        tv( 2, i, j ) = tv( 2, i, j )- tmat( 2, 3 ) * tv( 3, i, j )- tmat( 2, 4 ) * tv( 4, i, j )- tmat( 2, 5 ) * tv( 5, i, j )
                        tv( 2, i, j ) = tv( 2, i, j )/ tmat( 2, 2 )

                        tv( 1, i, j ) = tv( 1, i, j )- tmat( 1, 2 ) * tv( 2, i, j ) - tmat( 1, 3 ) * tv( 3, i, j )- tmat( 1, 4 ) * tv( 4, i, j )- tmat( 1, 5 ) * tv( 5, i, j )
                        tv( 1, i, j ) = tv( 1, i, j )/ tmat( 1, 1 )

                        v( 1, i, j, k ) = v( 1, i, j, k ) - tv( 1, i, j )
                        v( 2, i, j, k ) = v( 2, i, j, k ) - tv( 2, i, j )
                        v( 3, i, j, k ) = v( 3, i, j, k ) - tv( 3, i, j )
                        v( 4, i, j, k ) = v( 4, i, j, k ) - tv( 4, i, j )
                        v( 5, i, j, k ) = v( 5, i, j, k ) - tv( 5, i, j )
                  enddo
            end do
      
            return
      END SUBROUTINE buts

      SUBROUTINE jacu(k)
            implicit none

            INTEGER k
            INTEGER i, j
            DOUBLE PRECISION r43
            DOUBLE PRECISION c1345
            DOUBLE PRECISION c34
            DOUBLE PRECISION tmp1, tmp2, tmp3

            r43 = ( 4.0d0 / 3.0d0 )
            c1345 = c1 * c3 * c4 * c5
            c34 = c3 * c4

            do j = jst, jend
                  do i = ist, iend
                        tmp1 = rho_i(i,j,k)
                        tmp2 = tmp1 * tmp1
                        tmp3 = tmp1 * tmp2

                        d(1,1,i,j) =  1.0d0+ dt * 2.0d0 * (   tx1 * dx1+ ty1 * dy1+ tz1 * dz1 )
                        d(1,2,i,j) =  0.0d0
                        d(1,3,i,j) =  0.0d0
                        d(1,4,i,j) =  0.0d0
                        d(1,5,i,j) =  0.0d0

                        d(2,1,i,j) =  dt * 2.0d0* ( - tx1 * r43 - ty1 - tz1 )* ( c34 * tmp2 * u(2,i,j,k) )
                        d(2,2,i,j) =  1.0d0+ dt * 2.0d0 * c34 * tmp1 * (  tx1 * r43 + ty1 + tz1 )+ dt * 2.0d0 * (   tx1 * dx2+ ty1 * dy2+ tz1 * dz2  )
                        d(2,3,i,j) = 0.0d0
                        d(2,4,i,j) = 0.0d0
                        d(2,5,i,j) = 0.0d0

                        d(3,1,i,j) = dt * 2.0d0* ( - tx1 - ty1 * r43 - tz1 )* ( c34 * tmp2 * u(3,i,j,k) )
                        d(3,2,i,j) = 0.0d0
                        d(3,3,i,j) = 1.0d0+ dt * 2.0d0 * c34 * tmp1* (  tx1 + ty1 * r43 + tz1 )+ dt * 2.0d0 * (  tx1 * dx3+ ty1 * dy3+ tz1 * dz3 )
                        d(3,4,i,j) = 0.0d0
                        d(3,5,i,j) = 0.0d0

                        d(4,1,i,j) = dt * 2.0d0* ( - tx1 - ty1 - tz1 * r43 )* ( c34 * tmp2 * u(4,i,j,k) )
                        d(4,2,i,j) = 0.0d0
                        d(4,3,i,j) = 0.0d0
                        d(4,4,i,j) = 1.0d0+ dt * 2.0d0 * c34 * tmp1* (  tx1 + ty1 + tz1 * r43 )+ dt * 2.0d0 * (  tx1 * dx4+ ty1 * dy4+ tz1 * dz4 )
                        d(4,5,i,j) = 0.0d0

                        d(5,1,i,j) = -dt * 2.0d0* ( ( ( tx1 * ( r43*c34 - c1345 ) + ty1 * ( c34 - c1345 ) + tz1 * ( c34 - c1345 ) ) * ( u(2,i,j,k) ** 2 ) + ( tx1 * ( c34 - c1345 ) + ty1 * ( r43*c34 - c1345 ) + tz1 * ( c34 - c1345 ) ) * ( u(3,i,j,k) ** 2 ) + ( tx1 * ( c34 - c1345 ) + ty1 * ( c34 - c1345 ) + tz1 * ( r43*c34 - c1345 ) ) * ( u(4,i,j,k) ** 2 ) ) * tmp3 + ( tx1 + ty1 + tz1 ) * c1345 * tmp2 * u(5,i,j,k) )

                        d(5,2,i,j) = dt * 2.0d0 * ( tx1 * ( r43*c34 - c1345 ) + ty1 * (     c34 - c1345 ) + tz1 * (     c34 - c1345 ) ) * tmp2 * u(2,i,j,k)
                        d(5,3,i,j) = dt * 2.0d0 * ( tx1 * ( c34 - c1345 ) + ty1 * ( r43*c34 -c1345 ) + tz1 * ( c34 - c1345 ) ) * tmp2 * u(3,i,j,k)
                        d(5,4,i,j) = dt * 2.0d0 * ( tx1 * ( c34 - c1345 ) + ty1 * ( c34 - c1345 ) + tz1 * ( r43*c34 - c1345 ) ) * tmp2 * u(4,i,j,k)
                        d(5,5,i,j) = 1.0d0 + dt * 2.0d0 * ( tx1 + ty1 + tz1 ) * c1345 * tmp1 + dt * 2.0d0 * (  tx1 * dx5 +  ty1 * dy5 +  tz1 * dz5 )

                        tmp1 = rho_i(i+1,j,k)
                        tmp2 = tmp1 * tmp1
                        tmp3 = tmp1 * tmp2

                        a(1,1,i,j) = - dt * tx1 * dx1
                        a(1,2,i,j) =   dt * tx2
                        a(1,3,i,j) =   0.0d0
                        a(1,4,i,j) =   0.0d0
                        a(1,5,i,j) =   0.0d0

                        a(2,1,i,j) =  dt * tx2 * ( - ( u(2,i+1,j,k) * tmp1 ) ** 2 + c2 * qs(i+1,j,k) * tmp1 ) - dt * tx1 * ( - r43 * c34 * tmp2 * u(2,i+1,j,k) )
                        a(2,2,i,j) =  dt * tx2 * ( ( 2.0d0 - c2 ) * ( u(2,i+1,j,k) * tmp1 ) ) - dt * tx1 * ( r43 * c34 * tmp1 ) - dt * tx1 * dx2
                        a(2,3,i,j) =  dt * tx2 * ( - c2 * ( u(3,i+1,j,k) * tmp1 ) )
                        a(2,4,i,j) =  dt * tx2 * ( - c2 * ( u(4,i+1,j,k) * tmp1 ) )
                        a(2,5,i,j) =  dt * tx2 * c2 

                        a(3,1,i,j) =  dt * tx2 * ( - ( u(2,i+1,j,k) * u(3,i+1,j,k) ) * tmp2 ) - dt * tx1 * ( - c34 * tmp2 * u(3,i+1,j,k) )
                        a(3,2,i,j) =  dt * tx2 * ( u(3,i+1,j,k) * tmp1 )
                        a(3,3,i,j) =  dt * tx2 * ( u(2,i+1,j,k) * tmp1 ) - dt * tx1 * ( c34 * tmp1 ) - dt * tx1 * dx3
                        a(3,4,i,j) = 0.0d0
                        a(3,5,i,j) = 0.0d0

                        a(4,1,i,j) = dt * tx2 * ( - ( u(2,i+1,j,k)*u(4,i+1,j,k) ) * tmp2 ) - dt * tx1 * ( - c34 * tmp2 * u(4,i+1,j,k) )
                        a(4,2,i,j) = dt * tx2 * ( u(4,i+1,j,k) * tmp1 )
                        a(4,3,i,j) = 0.0d0
                        a(4,4,i,j) = dt * tx2 * ( u(2,i+1,j,k) * tmp1 ) - dt * tx1 * ( c34 * tmp1 ) - dt * tx1 * dx4
                        a(4,5,i,j) = 0.0d0

                        a(5,1,i,j) = dt * tx2 * ( ( c2 * 2.0d0 * qs(i+1,j,k) - c1 * u(5,i+1,j,k) ) * ( u(2,i+1,j,k) * tmp2 ) ) - dt * tx1 * ( - ( r43*c34 - c1345 ) * tmp3 * ( u(2,i+1,j,k)**2 )- (     c34 - c1345 ) * tmp3 * ( u(3,i+1,j,k)**2 ) - (     c34 - c1345 ) * tmp3 * ( u(4,i+1,j,k)**2 ) - c1345 * tmp2 * u(5,i+1,j,k) )
                        a(5,2,i,j) = dt * tx2 * ( c1 * ( u(5,i+1,j,k) * tmp1 ) - c2 * (  u(2,i+1,j,k)*u(2,i+1,j,k) * tmp2 + qs(i+1,j,k) * tmp1 ) ) - dt * tx1 * ( r43*c34 - c1345 ) * tmp2 * u(2,i+1,j,k)
                        a(5,3,i,j) = dt * tx2 * ( - c2 * ( u(3,i+1,j,k)*u(2,i+1,j,k) ) * tmp2 ) - dt * tx1 * (  c34 - c1345 ) * tmp2 * u(3,i+1,j,k)
                        a(5,4,i,j) = dt * tx2 * ( - c2 * ( u(4,i+1,j,k)*u(2,i+1,j,k) ) * tmp2 ) - dt * tx1 * (  c34 - c1345 ) * tmp2 * u(4,i+1,j,k)
                        a(5,5,i,j) = dt * tx2 * ( c1 * ( u(2,i+1,j,k) * tmp1 ) ) - dt * tx1 * c1345 * tmp1 - dt * tx1 * dx5

                        tmp1 = rho_i(i,j+1,k)
                        tmp2 = tmp1 * tmp1
                        tmp3 = tmp1 * tmp2

                        b(1,1,i,j) = - dt * ty1 * dy1
                        b(1,2,i,j) =   0.0d0
                        b(1,3,i,j) =  dt * ty2
                        b(1,4,i,j) =   0.0d0
                        b(1,5,i,j) =   0.0d0

                        b(2,1,i,j) =  dt * ty2 * ( - ( u(2,i,j+1,k)*u(3,i,j+1,k) ) * tmp2 ) - dt * ty1 * ( - c34 * tmp2 * u(2,i,j+1,k) )
                        b(2,2,i,j) =  dt * ty2 * ( u(3,i,j+1,k) * tmp1 ) - dt * ty1 * ( c34 * tmp1 ) - dt * ty1 * dy2
                        b(2,3,i,j) =  dt * ty2 * ( u(2,i,j+1,k) * tmp1 )
                        b(2,4,i,j) = 0.0d0
                        b(2,5,i,j) = 0.0d0

                        b(3,1,i,j) =  dt * ty2 * ( - ( u(3,i,j+1,k) * tmp1 ) ** 2 + c2 * ( qs(i,j+1,k) * tmp1 ) ) - dt * ty1 * ( - r43 * c34 * tmp2 * u(3,i,j+1,k) )
                        b(3,2,i,j) =  dt * ty2 * ( - c2 * ( u(2,i,j+1,k) * tmp1 ) )
                        b(3,3,i,j) =  dt * ty2 * ( ( 2.0d0 - c2 ) * ( u(3,i,j+1,k) * tmp1 ) ) - dt * ty1 * ( r43 * c34 * tmp1 ) - dt * ty1 * dy3
                        b(3,4,i,j) =  dt * ty2 * ( - c2 * ( u(4,i,j+1,k) * tmp1 ) )
                        b(3,5,i,j) =  dt * ty2 * c2

                        b(4,1,i,j) =  dt * ty2 * ( - ( u(3,i,j+1,k)*u(4,i,j+1,k) ) * tmp2 ) - dt * ty1 * ( - c34 * tmp2 * u(4,i,j+1,k) )
                        b(4,2,i,j) = 0.0d0
                        b(4,3,i,j) =  dt * ty2 * ( u(4,i,j+1,k) * tmp1 )
                        b(4,4,i,j) =  dt * ty2 * ( u(3,i,j+1,k) * tmp1 ) - dt * ty1 * ( c34 * tmp1 ) - dt * ty1 * dy4
                        b(4,5,i,j) = 0.0d0

                        b(5,1,i,j) =  dt * ty2 * ( ( c2 * 2.0d0 * qs(i,j+1,k) - c1 * u(5,i,j+1,k) ) * ( u(3,i,j+1,k) * tmp2 ) ) - dt * ty1 * ( - (     c34 - c1345 )*tmp3*(u(2,i,j+1,k)**2) - ( r43*c34 - c1345 )*tmp3*(u(3,i,j+1,k)**2) - (     c34 - c1345 )*tmp3*(u(4,i,j+1,k)**2) - c1345*tmp2*u(5,i,j+1,k) )
                        b(5,2,i,j) =  dt * ty2 * ( - c2 * ( u(2,i,j+1,k)*u(3,i,j+1,k) ) * tmp2 ) - dt * ty1 * ( c34 - c1345 ) * tmp2 * u(2,i,j+1,k)
                        b(5,3,i,j) =  dt * ty2 * ( c1 * ( u(5,i,j+1,k) * tmp1 ) - c2  * ( qs(i,j+1,k) * tmp1 + u(3,i,j+1,k)*u(3,i,j+1,k) * tmp2 ) ) - dt * ty1 * ( r43*c34 - c1345 ) * tmp2 * u(3,i,j+1,k)
                        b(5,4,i,j) =  dt * ty2 * ( - c2 * ( u(3,i,j+1,k)*u(4,i,j+1,k) ) * tmp2 ) - dt * ty1 * ( c34 - c1345 ) * tmp2 * u(4,i,j+1,k)
                        b(5,5,i,j) =  dt * ty2 * ( c1 * ( u(3,i,j+1,k) * tmp1 ) ) - dt * ty1 * c1345 * tmp1 - dt * ty1 * dy5

                        tmp1 = rho_i(i,j,k+1)
                        tmp2 = tmp1 * tmp1
                        tmp3 = tmp1 * tmp2

                        c(1,1,i,j) = - dt * tz1 * dz1
                        c(1,2,i,j) =   0.0d0
                        c(1,3,i,j) =   0.0d0
                        c(1,4,i,j) = dt * tz2
                        c(1,5,i,j) =   0.0d0

                        c(2,1,i,j) = dt * tz2 * ( - ( u(2,i,j,k+1)*u(4,i,j,k+1) ) * tmp2 ) - dt * tz1 * ( - c34 * tmp2 * u(2,i,j,k+1) )
                        c(2,2,i,j) = dt * tz2 * ( u(4,i,j,k+1) * tmp1 ) - dt * tz1 * c34 * tmp1 - dt * tz1 * dz2 
                        c(2,3,i,j) = 0.0d0
                        c(2,4,i,j) = dt * tz2 * ( u(2,i,j,k+1) * tmp1 )
                        c(2,5,i,j) = 0.0d0

                        c(3,1,i,j) = dt * tz2 * ( - ( u(3,i,j,k+1)*u(4,i,j,k+1) ) * tmp2 ) - dt * tz1 * ( - c34 * tmp2 * u(3,i,j,k+1) )
                        c(3,2,i,j) = 0.0d0
                        c(3,3,i,j) = dt * tz2 * ( u(4,i,j,k+1) * tmp1 ) - dt * tz1 * ( c34 * tmp1 ) - dt * tz1 * dz3
                        c(3,4,i,j) = dt * tz2 * ( u(3,i,j,k+1) * tmp1 )
                        c(3,5,i,j) = 0.0d0

                        c(4,1,i,j) = dt * tz2 * ( - ( u(4,i,j,k+1) * tmp1 ) ** 2 + c2 * ( qs(i,j,k+1) * tmp1 ) ) - dt * tz1 * ( - r43 * c34 * tmp2 * u(4,i,j,k+1) )
                        c(4,2,i,j) = dt * tz2 * ( - c2 * ( u(2,i,j,k+1) * tmp1 ) )
                        c(4,3,i,j) = dt * tz2 * ( - c2 * ( u(3,i,j,k+1) * tmp1 ) )
                        c(4,4,i,j) = dt * tz2 * ( 2.0d0 - c2 ) * ( u(4,i,j,k+1) * tmp1 ) - dt * tz1 * ( r43 * c34 * tmp1 ) - dt * tz1 * dz4
                        c(4,5,i,j) = dt * tz2 * c2

                        c(5,1,i,j) = dt * tz2 * ( ( c2 * 2.0d0 * qs(i,j,k+1) - c1 * u(5,i,j,k+1) ) * ( u(4,i,j,k+1) * tmp2 ) ) - dt * tz1 * ( - ( c34 - c1345 ) * tmp3 * (u(2,i,j,k+1)**2) - ( c34 - c1345 ) * tmp3 * (u(3,i,j,k+1)**2) - ( r43*c34 - c1345 )* tmp3 * (u(4,i,j,k+1)**2) - c1345 * tmp2 * u(5,i,j,k+1) )
                        c(5,2,i,j) = dt * tz2 * ( - c2 * ( u(2,i,j,k+1)*u(4,i,j,k+1) ) * tmp2 ) - dt * tz1 * ( c34 - c1345 ) * tmp2 * u(2,i,j,k+1)
                        c(5,3,i,j) = dt * tz2 * ( - c2 * ( u(3,i,j,k+1)*u(4,i,j,k+1) ) * tmp2 ) - dt * tz1 * ( c34 - c1345 ) * tmp2 * u(3,i,j,k+1)
                        c(5,4,i,j) = dt * tz2 * ( c1 * ( u(5,i,j,k+1) * tmp1 ) - c2 * ( qs(i,j,k+1) * tmp1 + u(4,i,j,k+1)*u(4,i,j,k+1) * tmp2 ) ) - dt * tz1 * ( r43*c34 - c1345 ) * tmp2 * u(4,i,j,k+1)
                        c(5,5,i,j) = dt * tz2 * ( c1 * ( u(4,i,j,k+1) * tmp1 ) ) - dt * tz1 * c1345 * tmp1 - dt * tz1 * dz5
                  end do
            end do

            return
      END SUBROUTINE jacu

END MODULE lu
