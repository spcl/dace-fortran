! Baseline (CPU/OpenMP) driver for the standalone us_exx::newdxx_g kernel:
! loads one dumped set (static tables + per-set transient state) into the
! TU's module variables, calls the REAL newdxx_g -- flag='c' k-point arm,
! OMP pragmas intact -- and verifies the accumulated deexx against the
! dumped pw.x ground truth (ndx_<set>_deexx_out.bin).
!
!   usage: ./verify_newdxx <dumpdir> <set> [reps] [warmup]
!     reps=0 (default): ONE verified call, prints kernel_time_s  (gate mode)
!     reps>0: warmup discarded calls + reps timed calls, EVERY call verified
!             (deexx reset to the dumped input before each), prints
!             kernel_time_s per call and a min/mean summary     (bench mode)
!
! Gate: max|out - ref| <= 1e-11 * max(1, max|ref|), the same bound as the
! SDFG lane.  newdxx_g's accumulation order is thread-count independent
! (static-schedule atom ownership; serial block loop), so observed error is
! ~1e-16.  Exits nonzero on any FAIL.
!
! DECK REPLICATION (env DECK_REP=R, default 1) grows the problem R-fold along
! the ATOM axis -- not the G axis used by addusxx_g, because newdxx_g's output
! is a reduction into deexx(ofsbeta(na)+ih): the output indirection here is
! ofsbeta, so that is the array that gets shifted.  nat -> R*nat, nkb -> R*nkb,
! ityp/tau/eigts1-3/becphi_c/deexx tiled R times, and replica k's ofsbeta is
! SHIFTED by +k*nkb so the beta manifolds -- hence the deexx target ranges --
! stay disjoint.  qgm/mill/dfftt/vc are species- or G-indexed and are NOT
! replicated.  Each replica must reproduce the pw.x reference on its own deexx
! slice, so the tiled-ref gate below is a per-replica check.  This axis is also
! the OpenMP `do` loop (nat = 2..5 in the shipped decks), so R is what gives
! the kernel enough iterations to fill a socket.
! DECK_REP_NOOFFSET=1 is the NEGATIVE CONTROL: it drops the +k*nkb shift, all
! replicas accumulate into replica 0's deexx range, and verification must FAIL.
program verify_newdxx_main
  use, intrinsic :: iso_fortran_env, only: error_unit
  use omp_lib,        only: omp_get_wtime, omp_get_max_threads
  use fft_types,      only: fft_type_descriptor
  use control_flags,  only: gamma_only
  use cell_base,      only: omega
  use gvect,          only: eigts1, eigts2, eigts3, gstart, mill
  use ions_base,      only: ityp, nat, tau
  use us_exx,         only: qgm, nij_type, newdxx_g
  use uspp,           only: ijtoh, nkb, ofsbeta, okvan
  use uspp_param,     only: lmaxq, nh, nhm, nsp, upf
  implicit none
  integer, parameter :: dp = 8
  real(dp), parameter :: tol = 1.0e-11_dp
  character(len=512) :: ddir
  character(len=64)  :: arg
  character(len=600) :: f
  type(fft_type_descriptor) :: dfftt
  complex(dp), allocatable :: vc(:), deexx(:), deexx_in(:), ref(:), becphi_c(:)
  real(dp) :: xkq(3), xk(3), maxref, maxdiff, t0, t1, tmin, tsum
  integer :: iset, nnr, ngmt, d1, d2, reps, warmup, irep, nfail
  integer :: rep, shift, nat0, nkb0
  logical :: ok

  if (command_argument_count() < 2) then
    write(error_unit,'(A)') 'usage: verify_newdxx <dumpdir> <set> [reps] [warmup]'
    error stop 1
  end if
  call get_command_argument(1, ddir)
  call get_command_argument(2, arg); read(arg,*) iset
  reps = 0; warmup = 0
  if (command_argument_count() >= 3) then
    call get_command_argument(3, arg); read(arg,*) reps
  end if
  if (command_argument_count() >= 4) then
    call get_command_argument(4, arg); read(arg,*) warmup
  end if

  write(*,'(A,I0)') 'threads ', omp_get_max_threads()

  rep   = env_i('DECK_REP', 1)
  shift = 1 - env_i('DECK_REP_NOOFFSET', 0)
  if (rep < 1) then
    write(error_unit,'(A)') 'DECK_REP must be >= 1'; error stop 1
  end if
  write(*,'(A,I0,A,I0)') 'replicate ', rep, ' ofsbeta_offset ', shift

  ! ---------------- static module state (adxndx_static_*) ----------------
  call load_static(trim(ddir), nnr, ngmt)

  ! ---------------- per-set transient state (ndx_<set>_*) ----------------
  write(f,'(A,A,I0,A)') trim(ddir), '/ndx_', iset, '_meta.txt'
  d1 = meta_i(trim(f), 'qgm_d1'); d2 = meta_i(trim(f), 'qgm_d2')
  allocate( qgm(d1, d2) )
  write(f,'(A,A,I0,A)') trim(ddir), '/ndx_', iset, '_qgm.bin'
  call rd_c(trim(f), qgm, d1*d2)
  write(f,'(A,A,I0,A)') trim(ddir), '/ndx_', iset, '_nij_type.bin'
  call rd_i(trim(f), nij_type, nsp)
  allocate( vc(nnr), deexx(nkb), deexx_in(nkb), ref(nkb), becphi_c(nkb) )
  write(f,'(A,A,I0,A)') trim(ddir), '/ndx_', iset, '_vc_in.bin';     call rd_c(trim(f), vc, nnr)
  write(f,'(A,A,I0,A)') trim(ddir), '/ndx_', iset, '_deexx_in.bin';  call rd_c_rep1(trim(f), deexx_in, nkb0, rep)
  write(f,'(A,A,I0,A)') trim(ddir), '/ndx_', iset, '_deexx_out.bin'; call rd_c_rep1(trim(f), ref, nkb0, rep)
  write(f,'(A,A,I0,A)') trim(ddir), '/ndx_', iset, '_xkq.bin';       call rd_r(trim(f), xkq, 3)
  write(f,'(A,A,I0,A)') trim(ddir), '/ndx_', iset, '_xk.bin';        call rd_r(trim(f), xk, 3)
  write(f,'(A,A,I0,A)') trim(ddir), '/ndx_', iset, '_becphi_c.bin';  call rd_c_rep1(trim(f), becphi_c, nkb0, rep)

  nfail = 0
  do irep = 1, warmup
    deexx = deexx_in
    call newdxx_g( dfftt, vc, xkq, xk, 'c', deexx, becphi_c=becphi_c )
  end do
  tmin = huge(tmin); tsum = 0.0_dp
  do irep = 1, max(reps, 1)
    deexx = deexx_in
    t0 = omp_get_wtime()
    call newdxx_g( dfftt, vc, xkq, xk, 'c', deexx, becphi_c=becphi_c )
    t1 = omp_get_wtime()
    write(*,'(A,F12.6)') 'kernel_time_s ', t1 - t0
    tmin = min(tmin, t1 - t0); tsum = tsum + (t1 - t0)
    call check(deexx, ref, ok)
    if (.not. ok) nfail = nfail + 1
  end do
  if (reps > 1) then
    write(*,'(A,F12.6)') 'kernel_time_min_s  ', tmin
    write(*,'(A,F12.6)') 'kernel_time_mean_s ', tsum / reps
  end if
  if (nfail > 0) then
    write(*,'(A)') 'VERIFY: FAIL'
    error stop 2
  end if
  write(*,'(A)') 'VERIFY: PASS'

contains

  subroutine check(out, refv, pass)
    complex(dp), intent(in) :: out(:), refv(:)
    logical, intent(out) :: pass
    integer :: ir, lo
    maxref  = max(1.0_dp, maxval(abs(refv)))
    maxdiff = maxval(abs(out - refv))
    write(*,'(A,ES12.4,A,ES12.4,A,ES12.4)') 'max|diff| = ', maxdiff, &
         '  max|ref| = ', maxval(abs(refv)), '  rel = ', maxdiff/maxref
    do ir = 0, min(rep, 4) - 1
      if (rep == 1) exit
      lo = ir*nkb0 + 1
      write(*,'(A,I0,A,ES12.4)') '  replica ', ir, ' max|diff| = ', &
           maxval(abs(out(lo:lo+nkb0-1) - refv(lo:lo+nkb0-1)))
    end do
    pass = (maxdiff/maxref <= tol)
  end subroutine check

  ! Static tables shared by the addusxx_g/newdxx_g decks (adxndx_static_*).
  ! Loads exactly the state the flag='c' arm reads: sizes + omega + tvanp/nh
  ! + ityp/tau/ofsbeta/ijtoh + eigts (with their negative lbounds) + mill +
  ! the dfftt fields (nnr/ngm/nl).  g/gg/vkb/indv exist in the deck but are
  ! not read by these kernels and are not loaded.
  subroutine load_static(dir, nnr_out, ngmt_out)
    character(len=*), intent(in) :: dir
    integer, intent(out) :: nnr_out, ngmt_out
    character(len=600) :: m
    integer :: nr1p, nr2p, nr3p, i, ir
    character(len=32) :: k
    m = trim(dir)//'/adxndx_static_meta.txt'
    nnr_out  = meta_i(trim(m), 'nnr')
    ngmt_out = meta_i(trim(m), 'ngmt')
    nkb0   = meta_i(trim(m), 'nkb')
    nat0   = meta_i(trim(m), 'nat')
    nkb    = rep*nkb0
    nat    = rep*nat0
    nsp    = meta_i(trim(m), 'nsp')
    nhm    = meta_i(trim(m), 'nhm')
    lmaxq  = meta_i(trim(m), 'lmaxq')
    gstart = meta_i(trim(m), 'gstart')
    omega  = meta_r(trim(m), 'omega')
    nr1p = meta_i(trim(m), 'nr1p'); nr2p = meta_i(trim(m), 'nr2p'); nr3p = meta_i(trim(m), 'nr3p')
    okvan = .true.;  gamma_only = .false.
    allocate( nh(nsp), upf(nsp) )
    do i = 1, nsp
      write(k,'(A,I0)') 'nh_', i;    nh(i) = meta_i(trim(m), trim(k))
      write(k,'(A,I0)') 'tvanp_', i; upf(i)%tvanp = meta_i(trim(m), trim(k)) /= 0
      write(k,'(A,I0)') 'tpawp_', i; upf(i)%tpawp = meta_i(trim(m), trim(k)) /= 0
      write(k,'(A,I0)') 'nbeta_', i; upf(i)%nbeta = meta_i(trim(m), trim(k))
    end do
    block
      integer :: ngm_full
      ngm_full = meta_i(trim(m), 'ngm')
      allocate( mill(3,ngm_full) )
      call rd_i(trim(dir)//'/adxndx_static_mill.bin', mill, 3*ngm_full)
    end block
    allocate( eigts1(-nr1p:nr1p, nat), eigts2(-nr2p:nr2p, nat), eigts3(-nr3p:nr3p, nat) )
    call rd_c(trim(dir)//'/adxndx_static_eigts1.bin', eigts1, (2*nr1p+1)*nat0)
    call rd_c(trim(dir)//'/adxndx_static_eigts2.bin', eigts2, (2*nr2p+1)*nat0)
    call rd_c(trim(dir)//'/adxndx_static_eigts3.bin', eigts3, (2*nr3p+1)*nat0)
    allocate( tau(3,nat), ityp(nat), ofsbeta(nat) )
    call rd_r(trim(dir)//'/adxndx_static_tau.bin', tau, 3*nat0)
    call rd_i(trim(dir)//'/adxndx_static_ityp.bin', ityp, nat0)
    call rd_i(trim(dir)//'/adxndx_static_ofsbeta.bin', ofsbeta, nat0)
    ! ofsbeta is the ONLY table that shifts: it is the output indirection.
    do ir = 1, rep - 1
      eigts1(:, ir*nat0+1:(ir+1)*nat0) = eigts1(:, 1:nat0)
      eigts2(:, ir*nat0+1:(ir+1)*nat0) = eigts2(:, 1:nat0)
      eigts3(:, ir*nat0+1:(ir+1)*nat0) = eigts3(:, 1:nat0)
      tau(:, ir*nat0+1:(ir+1)*nat0)    = tau(:, 1:nat0)
      ityp(ir*nat0+1:(ir+1)*nat0)      = ityp(1:nat0)
      ofsbeta(ir*nat0+1:(ir+1)*nat0)   = ofsbeta(1:nat0) + shift*ir*nkb0
    end do
    allocate( ijtoh(nhm,nhm,nsp) )
    call rd_i(trim(dir)//'/adxndx_static_ijtoh.bin', ijtoh, nhm*nhm*nsp)
    allocate( nij_type(nsp) )
    nij_type = 0   ! loaded per set (CUMULATIVE species offsets, qvan_init
                   ! semantics: qgm column = nij_type(nt)+ijtoh(ih,jh,nt))
    dfftt%nr1  = meta_i(trim(m), 'nr1t');  dfftt%nr2  = meta_i(trim(m), 'nr2t');  dfftt%nr3  = meta_i(trim(m), 'nr3t')
    dfftt%nr1x = dfftt%nr1; dfftt%nr2x = dfftt%nr2; dfftt%nr3x = dfftt%nr3
    dfftt%nnr  = nnr_out
    dfftt%ngm  = ngmt_out
    allocate( dfftt%nl(ngmt_out) )
    call rd_i(trim(dir)//'/adxndx_static_dfftt_nl.bin', dfftt%nl, ngmt_out)
  end subroutine load_static

  function env_i(name, dflt) result(v)
    character(len=*), intent(in) :: name
    integer, intent(in) :: dflt
    integer :: v, ln, st
    character(len=32) :: s
    call get_environment_variable(name, s, ln, st)
    v = dflt
    if (st == 0 .and. ln > 0) read(s,*) v
  end function env_i

  ! Reads the deck's single copy into replica 0's slice, then broadcasts it to
  ! the remaining rep-1 slices in place (no extra buffer).
  subroutine rd_c_rep1(fname, a, n, r)
    character(len=*), intent(in) :: fname
    integer, intent(in) :: n, r
    complex(dp), intent(out) :: a(n, r)
    integer :: u, ios, ir
    open(newunit=u, file=fname, form='unformatted', access='stream', status='old', action='read', iostat=ios)
    if (ios /= 0) then
      write(error_unit,*) 'cannot open ', fname; error stop 2
    end if
    read(u) a(1:n, 1); close(u)
    do ir = 2, r
      a(:, ir) = a(:, 1)
    end do
  end subroutine rd_c_rep1

  ! -------- tiny meta/bin readers ("key value" text; raw stream bins) ----
  function meta_i(fname, key) result(v)
    character(len=*), intent(in) :: fname, key
    integer :: v
    v = nint(meta_r(fname, key))
  end function meta_i

  function meta_r(fname, key) result(v)
    character(len=*), intent(in) :: fname, key
    real(dp) :: v
    character(len=256) :: line, k
    integer :: u, ios
    open(newunit=u, file=fname, status='old', action='read', iostat=ios)
    if (ios /= 0) then
      write(error_unit,*) 'cannot open ', fname; error stop 2
    end if
    do
      read(u,'(A)', iostat=ios) line
      if (ios /= 0) exit
      read(line,*,iostat=ios) k, v
      if (ios == 0 .and. trim(k) == trim(key)) then
        close(u); return
      end if
    end do
    close(u)
    write(error_unit,*) 'meta key not found: ', trim(key), ' in ', fname
    error stop 3
  end function meta_r

  subroutine rd_c(fname, a, n)
    character(len=*), intent(in) :: fname
    integer, intent(in) :: n
    complex(dp), intent(out) :: a(*)
    integer :: u, ios
    open(newunit=u, file=fname, form='unformatted', access='stream', status='old', action='read', iostat=ios)
    if (ios /= 0) then
      write(error_unit,*) 'cannot open ', fname; error stop 2
    end if
    read(u) a(1:n); close(u)
  end subroutine rd_c

  subroutine rd_r(fname, a, n)
    character(len=*), intent(in) :: fname
    integer, intent(in) :: n
    real(dp), intent(out) :: a(*)
    integer :: u, ios
    open(newunit=u, file=fname, form='unformatted', access='stream', status='old', action='read', iostat=ios)
    if (ios /= 0) then
      write(error_unit,*) 'cannot open ', fname; error stop 2
    end if
    read(u) a(1:n); close(u)
  end subroutine rd_r

  subroutine rd_i(fname, a, n)
    character(len=*), intent(in) :: fname
    integer, intent(in) :: n
    integer, intent(out) :: a(*)
    integer :: u, ios
    open(newunit=u, file=fname, form='unformatted', access='stream', status='old', action='read', iostat=ios)
    if (ios /= 0) then
      write(error_unit,*) 'cannot open ', fname; error stop 2
    end if
    read(u) a(1:n); close(u)
  end subroutine rd_i

end program verify_newdxx_main
