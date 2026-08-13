! Standalone vexx_bp_k verification driver.
!
! Populates the QE module globals of the OMP-preserving inlined TU
! (out_regex_omp/vexx_bp_k_merged_omp.f90) from a QE dump slot
! (experiments/BaTiO3_nat005_hse/dump_vexx, slots 0..3 = exx iters 1..2 x
! k-points 3,6), calls the REAL inlined vexx_bp_k on the dumped psi with
! hpsi=0, and verifies hpsi == big_result (Vx*psi) against the dumped ground
! truth: big_result_full (US+PAW operator, mode 'full') or big_result_nc
! (augmentation off, mode 'nc').
!
! The TU recomputes qgm / ylm / vkb / coulomb_fac internally via its inlined
! qvan_init, qvan2, ylmr2, init_us_2 and g2_convolution_all -- so this driver
! loads only the static tables those consume (tab_qrad, tab_beta, uspp index
! tables, structure-factor inputs, PAW Fock kernel ke) plus the per-slot
! runtime inputs (psi, becpsi, exxbuff, becxx, occupations, index tables).
! FFT is provided by fft_shim.f90 (FFTW3, cached threaded plans); BLAS/LAPACK
! are linked; MPI runs as a singleton with COMM_SELF communicators.
module verify_vexx
  implicit none
  private
  public :: run_case
  integer, parameter :: dp = 8
  ! parsed meta key-value store
  integer, parameter :: MAXMETA = 512
  character(len=64) :: mkey(MAXMETA)
  real(dp) :: mval(MAXMETA)
  integer :: nmeta = 0
contains

  subroutine load_meta(fname, append)
    character(len=*), intent(in) :: fname
    logical, intent(in) :: append
    character(len=256) :: line
    integer :: u, ios
    if (.not. append) nmeta = 0
    open(newunit=u, file=fname, status='old', action='read', iostat=ios)
    if (ios /= 0) then
       write(*,*) 'cannot open meta file: ', trim(fname)
       error stop 2
    end if
    do
       read(u,'(A)', iostat=ios) line
       if (ios /= 0) exit
       if (len_trim(line) == 0) cycle
       nmeta = nmeta + 1
       if (nmeta > MAXMETA) error stop 'meta store full'
       read(line,*) mkey(nmeta), mval(nmeta)
    end do
    close(u)
  end subroutine load_meta

  function getr(key) result(v)
    character(len=*), intent(in) :: key
    real(dp) :: v
    integer :: i
    do i = 1, nmeta
       if (trim(mkey(i)) == trim(key)) then
          v = mval(i)
          return
       end if
    end do
    write(*,*) 'meta key not found: ', trim(key)
    error stop 3
  end function getr

  function geti(key) result(v)
    character(len=*), intent(in) :: key
    integer :: v
    v = nint(getr(key))
  end function geti

  function haskey(key) result(t)
    character(len=*), intent(in) :: key
    logical :: t
    integer :: i
    t = .false.
    do i = 1, nmeta
       if (trim(mkey(i)) == trim(key)) then
          t = .true.
          return
       end if
    end do
  end function haskey

  subroutine rd_c(fname, a, n)
    character(len=*), intent(in) :: fname
    integer, intent(in) :: n
    complex(dp), intent(out) :: a(*)
    integer :: u, ios
    open(newunit=u, file=fname, form='unformatted', access='stream', &
         status='old', action='read', iostat=ios)
    if (ios /= 0) then
       write(*,*) 'cannot open ', trim(fname)
       error stop 2
    end if
    read(u) a(1:n)
    close(u)
  end subroutine rd_c

  subroutine rd_r(fname, a, n)
    character(len=*), intent(in) :: fname
    integer, intent(in) :: n
    real(dp), intent(out) :: a(*)
    integer :: u, ios
    open(newunit=u, file=fname, form='unformatted', access='stream', &
         status='old', action='read', iostat=ios)
    if (ios /= 0) then
       write(*,*) 'cannot open ', trim(fname)
       error stop 2
    end if
    read(u) a(1:n)
    close(u)
  end subroutine rd_r

  subroutine rd_i(fname, a, n)
    character(len=*), intent(in) :: fname
    integer, intent(in) :: n
    integer, intent(out) :: a(*)
    integer :: u, ios
    open(newunit=u, file=fname, form='unformatted', access='stream', &
         status='old', action='read', iostat=ios)
    if (ios /= 0) then
       write(*,*) 'cannot open ', trim(fname)
       error stop 2
    end if
    read(u) a(1:n)
    close(u)
  end subroutine rd_i

  subroutine run_case(dumpdir, slot, mode)
    use mpi,               only: MPI_COMM_SELF
    use omp_lib,           only: omp_get_wtime, omp_get_max_threads
    ! --- TU modules whose state we initialize ---------------------------
    use wvfct,             only: current_k, npwx, nbnd
    use klist,             only: xk, nks, nkstot
    use noncollin_module,  only: noncolin, npol, nspin_mag
    use control_flags,     only: gamma_only, many_fft, tqr, use_gpu
    use cell_base,         only: at, bg, omega, tpiba, tpiba2, alat
    use mp_pools,          only: npool, my_pool_id, kunit
    use mp_exx,            only: negrp, my_egrp_id, me_egrp, nproc_egrp, &
                                 jblock, max_pairs, max_ibands, iexx_start, &
                                 nibands, ibands, egrp_pairs, all_start, all_end, &
                                 iexx_istart, iexx_iend, inter_egrp_comm, intra_egrp_comm
    use exx_base,          only: dfftt, exxbuff, x_occupation, x_nbnd_occ, &
                                 xkq_collect, index_xk, index_xkq, nqs, gt, &
                                 exxalfa, exxdiv, eps, eps_qdiv, erfc_scrlen, &
                                 erf_scrlen, gau_scrlen, yukawa, grid_factor, &
                                 x_gamma_extrapolation, use_coulomb_vcut_ws, &
                                 use_coulomb_vcut_spheric, nq1, nq2, nq3
    use gvect,             only: ngm, gstart, g, gg, mill, eigts1, eigts2, eigts3
    use ions_base,         only: nat, ityp, tau
    use ions_base,         only: nsp_ions => nsp
    use uspp,              only: okvan, nkb, nhtol, nhtolm, indv, ijtoh, ofsbeta, &
                                 ap, lpx, lpl
    use uspp_param,        only: nh, nhm, nbetam, lmaxkb, lmaxq, upf
    use uspp_param,        only: nsp_param => nsp
    use beta_mod,          only: nqx, qmax, tab_beta
    use qrad_mod,          only: dq, tab_qrad
    use paw_variables,     only: okpaw
    use paw_exx,           only: ke, paw_has_init_paw_fockrnl
    use us_exx,            only: becxx
    use exx_bp_utils,      only: igk_exx
    use becmod,            only: bec_type
    use fft_base,          only: dfftp
    use exx_bp,            only: vexx_bp_k, coulomb_fac
    use us_exx,            only: qvan_init, qvan_clean, qgm
    use uspp_init,         only: init_us_2
    implicit none
    character(len=*), intent(in) :: dumpdir, mode
    integer, intent(in) :: slot
    !
    character(len=512) :: pfx, spfx, f
    type(bec_type) :: becpsi
    complex(dp), allocatable :: psi(:,:), hpsi(:,:), ref(:,:)
    real(dp), allocatable :: cfac_ref(:)
    integer, allocatable :: idxkq(:)
    integer :: n, m, lda, iter, ngmt, nkqs_, nib_len, iq, ikq, isp, d1
    integer :: becxx_nbnd, i, irep, nx2
    real(dp) :: maxref, maxdiff, d
    real(dp), allocatable :: tbuf(:)
    !
    write(pfx,'(A,A,I0,A)') trim(dumpdir), '/vexx_', slot, '_'
    spfx = trim(dumpdir)//'/vexx_static_'
    call load_meta(trim(pfx)//'meta.txt', .false.)
    call load_meta(trim(spfx)//'meta.txt', .true.)
    !
    ! ---- scalars ------------------------------------------------------
    n = geti('n'); m = geti('m'); lda = geti('lda'); iter = geti('iter')
    npwx = geti('npwx'); npol = geti('npol'); nbnd = geti('nbnd')
    noncolin = geti('noncolin') /= 0
    nspin_mag = 1
    current_k = geti('current_k')
    nks = geti('nks'); nkstot = geti('nkstot')
    ngmt = geti('ngmt'); nqs = geti('nqs'); nkqs_ = geti('nkqs')
    gamma_only = .false.; use_gpu = .false.
    many_fft = geti('many_fft'); tqr = geti('tqr') /= 0
    okvan = geti('okvan') /= 0; okpaw = geti('okpaw') /= 0
    nkb = geti('nkb'); nat = geti('nat')
    x_nbnd_occ = geti('x_nbnd_occ')
    exxalfa = getr('exxalfa'); omega = getr('omega'); tpiba2 = getr('tpiba2')
    exxdiv = getr('exxdiv'); eps = getr('eps'); eps_qdiv = getr('eps_qdiv')
    gau_scrlen = getr('gau_scrlen'); erf_scrlen = getr('erf_scrlen')
    erfc_scrlen = getr('erfc_scrlen'); yukawa = getr('yukawa')
    grid_factor = getr('grid_factor')
    x_gamma_extrapolation = geti('x_gamma_extrapolation') /= 0
    use_coulomb_vcut_ws = geti('use_coulomb_vcut_ws') /= 0
    use_coulomb_vcut_spheric = geti('use_coulomb_vcut_spheric') /= 0
    nq1 = geti('nq1'); nq2 = geti('nq2'); nq3 = geti('nq3')
    alat = getr('alat'); tpiba = getr('tpiba')
    nsp_param = geti('nsp'); nsp_ions = nsp_param
    nhm = geti('nhm'); nbetam = geti('nbetam')
    lmaxkb = geti('lmaxkb'); lmaxq = geti('lmaxq')
    nqx = geti('nqx'); qmax = getr('qmax')
    ngm = geti('ngm'); gstart = geti('gstart')
    dfftp%nr1 = geti('nr1p'); dfftp%nr2 = geti('nr2p'); dfftp%nr3 = geti('nr3p')
    if (abs(getr('dq') - dq) > 1d-14) error stop 'dq mismatch vs qrad_mod parameter'
    !
    ! ---- mp / pools (serial semantics, MPI singleton) ------------------
    npool = 1; my_pool_id = 0; kunit = 1
    negrp = 1; my_egrp_id = 0; me_egrp = 0; nproc_egrp = 1
    inter_egrp_comm = MPI_COMM_SELF; intra_egrp_comm = MPI_COMM_SELF
    jblock = geti('jblock'); max_pairs = geti('max_pairs')
    max_ibands = geti('max_ibands'); iexx_start = geti('iexx_start')
    allocate( nibands(1), all_start(1), all_end(1), iexx_istart(1), iexx_iend(1) )
    nibands(1) = geti('nibands')
    all_start(1) = geti('all_start'); all_end(1) = geti('all_end')
    iexx_istart(1) = geti('iexx_istart'); iexx_iend(1) = geti('iexx_iend')
    nib_len = nibands(1)   ! ibands bin length written as SIZE(ibands,1)
    call bin_len(trim(pfx)//'ibands.bin', 4, i)
    allocate( ibands(i,1) )
    call rd_i(trim(pfx)//'ibands.bin', ibands(:,1), i)
    allocate( egrp_pairs(2, max_pairs, 1) )
    call rd_i(trim(pfx)//'egrp_pairs.bin', egrp_pairs(:,:,1), 2*max_pairs)
    !
    ! ---- cell / k geometry ---------------------------------------------
    call rd_r(trim(spfx)//'at.bin', at, 9)
    call rd_r(trim(spfx)//'bg.bin', bg, 9)
    call rd_r(trim(spfx)//'xk.bin', xk(:,1:nks), 3*nks)
    allocate( xkq_collect(3, nkqs_), index_xk(nkqs_) )
    call rd_r(trim(spfx)//'xkq_collect.bin', xkq_collect, 3*nkqs_)
    call rd_i(trim(spfx)//'index_xk.bin', index_xk, nkqs_)
    allocate( index_xkq(nkstot, nqs), idxkq(nqs) )
    index_xkq = 0
    call rd_i(trim(pfx)//'index_xkq.bin', idxkq, nqs)
    index_xkq(geti('current_ik'), 1:nqs) = idxkq
    !
    ! ---- EXX grid descriptor + G-vectors --------------------------------
    dfftt%nr1 = geti('nr1t'); dfftt%nr2 = geti('nr2t'); dfftt%nr3 = geti('nr3t')
    dfftt%nr1x = geti('nr1xt'); dfftt%nr2x = geti('nr2xt'); dfftt%nr3x = geti('nr3xt')
    dfftt%nnr = geti('nrxxs'); dfftt%ngm = ngmt
    allocate( dfftt%nl(ngmt) )
    call rd_i(trim(spfx)//'dfftt_nl.bin', dfftt%nl, ngmt)
    allocate( gt(3, ngmt) )
    call rd_r(trim(spfx)//'gt.bin', gt, 3*ngmt)
    !
    ! ---- dense-grid G data for init_us_2 / structure factors ------------
    allocate( g(3,ngm), gg(ngm), mill(3,ngm) )
    call rd_r(trim(spfx)//'g.bin', g, 3*ngm)
    call rd_r(trim(spfx)//'gg.bin', gg, ngm)
    call rd_i(trim(spfx)//'mill.bin', mill, 3*ngm)
    allocate( eigts1(-dfftp%nr1:dfftp%nr1, nat), &
              eigts2(-dfftp%nr2:dfftp%nr2, nat), &
              eigts3(-dfftp%nr3:dfftp%nr3, nat) )
    call rd_c(trim(spfx)//'eigts1.bin', eigts1, (2*dfftp%nr1+1)*nat)
    call rd_c(trim(spfx)//'eigts2.bin', eigts2, (2*dfftp%nr2+1)*nat)
    call rd_c(trim(spfx)//'eigts3.bin', eigts3, (2*dfftp%nr3+1)*nat)
    allocate( tau(3,nat), ityp(nat) )
    call rd_r(trim(spfx)//'tau.bin', tau, 3*nat)
    call rd_i(trim(spfx)//'ityp.bin', ityp, nat)
    !
    ! ---- US/PAW tables ---------------------------------------------------
    allocate( nh(nsp_param), upf(nsp_param) )
    do isp = 1, nsp_param
       write(f,'(A,I0)') 'nh_', isp
       nh(isp) = geti(trim(f))
       write(f,'(A,I0)') 'tvanp_', isp
       upf(isp)%tvanp = geti(trim(f)) /= 0
       write(f,'(A,I0)') 'tpawp_', isp
       upf(isp)%tpawp = geti(trim(f)) /= 0
    end do
    allocate( nhtol(nhm,nsp_param), nhtolm(nhm,nsp_param), indv(nhm,nsp_param) )
    allocate( ijtoh(nhm,nhm,nsp_param), ofsbeta(nat) )
    call rd_i(trim(spfx)//'nhtol.bin', nhtol, nhm*nsp_param)
    call rd_i(trim(spfx)//'nhtolm.bin', nhtolm, nhm*nsp_param)
    call rd_i(trim(spfx)//'indv.bin', indv, nhm*nsp_param)
    ! interp_beta reads upf(nt)%nbeta; recover it from the indv map
    ! (indv(ih,nt) enumerates each projector's beta index 1..nbeta(nt))
    do isp = 1, nsp_param
       upf(isp)%nbeta = maxval(indv(1:nh(isp),isp))
    end do
    call rd_i(trim(spfx)//'ijtoh.bin', ijtoh, nhm*nhm*nsp_param)
    call rd_i(trim(spfx)//'ofsbeta.bin', ofsbeta, nat)
    if (size(ap) /= geti('ap_d1')*geti('ap_d2')*geti('ap_d3')) &
       error stop 'ap size mismatch (lqmax/nlx differ between TU and dump)'
    call rd_r(trim(spfx)//'ap.bin', ap, size(ap))
    call rd_i(trim(spfx)//'lpx.bin', lpx, size(lpx))
    call rd_i(trim(spfx)//'lpl.bin', lpl, size(lpl))
    allocate( tab_qrad(geti('tabqrad_d1'), geti('tabqrad_d2'), &
                       geti('tabqrad_d3'), geti('tabqrad_d4')) )
    call rd_r(trim(spfx)//'tab_qrad.bin', tab_qrad, size(tab_qrad))
    allocate( tab_beta(geti('tabbeta_d1'), geti('tabbeta_d2'), geti('tabbeta_d3')) )
    call rd_r(trim(spfx)//'tab_beta.bin', tab_beta, size(tab_beta))
    !
    ! ---- PAW Fock kernel -------------------------------------------------
    if (okpaw) then
       allocate( ke(geti('ke_n')) )
       do isp = 1, geti('ke_n')
          write(f,'(A,I0)') 'ke_sz_', isp
          if (geti(trim(f)) == 0) cycle
          write(f,'(A,I0)') 'ke_d1_', isp
          d1 = geti(trim(f))
          allocate( ke(isp)%k(d1,d1,d1,d1) )
          write(f,'(A,A,I0,A)') trim(spfx), 'ke_', isp, '.bin'
          call rd_r(trim(f), ke(isp)%k, d1**4)
       end do
       paw_has_init_paw_fockrnl = .true.
    end if
    !
    ! ---- per-slot runtime inputs -----------------------------------------
    allocate( exxbuff(geti('exxbuff_nr'), geti('nbnd_buff'), geti('nkq_buff')) )
    write(f,'(A,A,I0,A)') trim(dumpdir), '/vexx_it', iter, '_exxbuff.bin'
    call rd_c(trim(f), exxbuff, size(exxbuff))
    call bin_len(trim(pfx)//'x_occupation.bin', 8, i)
    nx2 = i/nbnd
    allocate( x_occupation(nbnd, nx2) )
    call rd_r(trim(pfx)//'x_occupation.bin', x_occupation, nbnd*nx2)
    allocate( igk_exx(npwx, nks) )
    igk_exx = 0
    call rd_i(trim(pfx)//'igk_exx.bin', igk_exx(:,current_k), npwx)
    if (okvan) then
       becxx_nbnd = geti('becxx_nbnd')
       allocate( becxx(nkqs_) )
       do iq = 1, nqs
          ikq = idxkq(iq)
          if (.not. allocated(becxx(ikq)%k)) then
             allocate( becxx(ikq)%k(nkb, becxx_nbnd) )
             write(f,'(A,A,I0,A)') trim(pfx), 'becxx_', iq, '.bin'
             call rd_c(trim(f), becxx(ikq)%k, nkb*becxx_nbnd)
          end if
       end do
       allocate( becpsi%k(nkb, m) )
       call rd_c(trim(pfx)//'becpsi.bin', becpsi%k, nkb*m)
    end if
    allocate( psi(npwx*npol, m), hpsi(npwx*npol, m) )
    call rd_c(trim(pfx)//'psi.bin', psi, npwx*npol*m)
    !
    ! ---- 'aug' probe mode: cross-check the TU's internally computed
    ! augmentation ingredients against the QE-dumped references (old in-kernel
    ! dump schema provides qgm_<iq> and vkb), then stop.  Splits the fault
    ! domain: qgm -> qvan tables, vkb -> beta/structure-factor tables.
    if (mode == 'aug') then
       block
         complex(dp), allocatable :: qgm_ref(:), vkb_ref(:,:), vkbp(:,:), qgm_flat(:)
         real(dp) :: xkq_(3), xkp_(3), mr
         integer :: nref, ncmp, ikq1
         ikq1 = idxkq(1)
         xkq_ = xkq_collect(:,ikq1)
         xkp_ = xk(:,current_k)
         call qvan_init(ngmt, xkq_, xkp_)
         call bin_len(trim(pfx)//'qgm_1.bin', 16, nref)
         allocate( qgm_ref(nref), qgm_flat(size(qgm)) )
         call rd_c(trim(pfx)//'qgm_1.bin', qgm_ref, nref)
         qgm_flat = reshape(qgm, [size(qgm)])
         ncmp = min(size(qgm_flat), nref)
         write(*,'(A,I0,A,I0)') 'aug probe qgm: TU size ', size(qgm_flat), ' ref size ', nref
         mr = maxval(abs(qgm_ref(1:ncmp)))
         write(*,'(A,ES12.4,A,ES12.4)') 'aug probe qgm: max|diff| = ', &
              maxval(abs(qgm_flat(1:ncmp)-qgm_ref(1:ncmp))), '  max|ref| = ', mr
         call qvan_clean()
         allocate( vkbp(npwx,nkb), vkb_ref(npwx,nkb) )
         vkbp = (0.0_dp, 0.0_dp)
         call rd_c(trim(pfx)//'vkb.bin', vkb_ref, npwx*nkb)
         call init_us_2(n, igk_exx(1:n,current_k), xkp_, vkbp)
         mr = maxval(abs(vkb_ref(1:n,:)))
         write(*,'(A,ES12.4,A,ES12.4)') 'aug probe vkb: max|diff| = ', &
              maxval(abs(vkbp(1:n,:)-vkb_ref(1:n,:))), '  max|ref| = ', mr
         write(*,'(A,ES12.4,A,I0,A,ES12.4,A,ES12.4)') 'aug probe vkb: max|tu| = ', &
              maxval(abs(vkbp(1:n,:))), '  nqx=', nqx, ' qmax=', qmax, &
              ' max|tab_beta|=', maxval(abs(tab_beta))
         write(*,'(A,8I8)') 'aug probe igk_exx(1:8): ', igk_exx(1:8,current_k)
       end block
       return
    end if
    !
    ! ---- mode: 'nc' switches augmentation off (matches the dumped probe) --
    if (mode == 'nc') then
       okvan = .false.
       okpaw = .false.
    end if
    !
    ! ---- reference -------------------------------------------------------
    allocate( ref(n*npol, m) )
    if (mode == 'nc') then
       call rd_c(trim(pfx)//'big_result_nc.bin', ref, n*npol*m)
    else
       call rd_c(trim(pfx)//'big_result_full.bin', ref, n*npol*m)
    end if
    !
    ! ---- run + time --------------------------------------------------------
    write(*,'(A,I0,A,A,A,I0,A,I0,A,I0,A,I0)') 'slot ', slot, ' mode ', trim(mode), &
         '  iter=', iter, ' k=', current_k, '  n=', n, ' m=', m
    write(*,'(A,I0)') 'omp_max_threads = ', omp_get_max_threads()
    !
    ! boundary-dump schema: hpsi_in.bin (if present) seeds the accumulator and
    ! the ground truth is hpsi_out; older in-kernel dumps have no hpsi_in and
    ! their big_result excludes the accumulator, so start from zero there.
    ! ONE kernel call; hpsi is then compared against the dumped ground truth.
    block
      logical :: has_hin
      complex(dp), allocatable :: hin(:,:)
      real(dp) :: tk0, tk1
      allocate( hin(npwx*npol, m) )
      hin = (0.0_dp, 0.0_dp)
      inquire(file=trim(pfx)//'hpsi_in.bin', exist=has_hin)
      if (has_hin) call rd_c(trim(pfx)//'hpsi_in.bin', hin(1:n,1:m), n*m)
      hpsi = hin
      tk0 = omp_get_wtime()
      call vexx_bp_k(lda, n, m, psi, hpsi, becpsi)
      tk1 = omp_get_wtime()
      ! single-call wall time, one line per rank (parsed by measure_sweep.sh)
      write(*,'(A,F12.4)') 'kernel_time_s ', tk1 - tk0
    end block
    !
    ! ---- verify ------------------------------------------------------------
    maxref = 0.0_dp; maxdiff = 0.0_dp
    do i = 1, m
       maxref = max(maxref, maxval(abs(ref(:,i))))
       maxdiff = max(maxdiff, maxval(abs(hpsi(1:n*npol,i) - ref(:,i))))
    end do
    write(*,'(A,ES12.4,A,ES12.4,A,ES12.4)') 'verify: max|diff| = ', maxdiff, &
         '  max|ref| = ', maxref, '  rel = ', maxdiff/maxref
    !
    ! coulomb_fac cross-check (TU-computed vs QE-dumped)
    allocate( cfac_ref(ngmt*nqs), tbuf(ngmt*nqs) )
    call rd_r(trim(pfx)//'coulomb_fac.bin', cfac_ref, ngmt*nqs)
    tbuf = reshape(coulomb_fac(:,:,current_k), [ngmt*nqs])
    write(*,'(A,ES12.4)') 'coulomb_fac cross-check: max|diff| = ', &
         maxval(abs(tbuf - cfac_ref))
    !
    !
    if (maxdiff/maxref > 1.0d-11) then
       write(*,'(A)') 'VERIFY: FAIL'
       error stop 1
    else
       write(*,'(A)') 'VERIFY: PASS'
    end if
  end subroutine run_case

  subroutine bin_len(fname, elem_bytes, n)
    character(len=*), intent(in) :: fname
    integer, intent(in) :: elem_bytes
    integer, intent(out) :: n
    integer(8) :: sz
    integer :: u, ios
    open(newunit=u, file=fname, form='unformatted', access='stream', &
         status='old', action='read', iostat=ios)
    if (ios /= 0) then
       write(*,*) 'cannot open ', trim(fname)
       error stop 2
    end if
    inquire(unit=u, size=sz)
    close(u)
    n = int(sz/elem_bytes)
  end subroutine bin_len

end module verify_vexx

program verify_vexx_main
  use mpi
  use verify_vexx, only: run_case
  implicit none
  character(len=512) :: dumpdir, arg
  integer :: slot, ierr
  character(len=8) :: mode
  if (command_argument_count() < 2) then
     write(*,*) 'usage: verify_vexx <dumpdir> <slot 0-3> [full|nc|aug]'
     stop 1
  end if
  call get_command_argument(1, dumpdir)
  call get_command_argument(2, arg); read(arg,*) slot
  mode = 'full'
  if (command_argument_count() >= 3) call get_command_argument(3, mode)
  call MPI_Init(ierr)
  call run_case(trim(dumpdir), slot, trim(mode))
  call MPI_Finalize(ierr)
end program verify_vexx_main
