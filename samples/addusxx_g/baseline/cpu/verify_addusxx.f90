! Baseline (CPU/OpenMP) driver for the standalone us_exx::addusxx_g kernel:
! loads one dumped set (static tables + per-set transient state) into the
! TU's module variables, calls the REAL addusxx_g -- flag='c' k-point arm,
! OMP pragmas intact -- and verifies rhoc against the dumped pw.x ground
! truth (adx_<set>_rhoc_out.bin).
!
!   usage: ./verify_addusxx <dumpdir> <set> [reps] [warmup]
!     reps=0 (default): ONE verified call, prints kernel_time_s  (gate mode)
!     reps>0: warmup discarded calls + reps timed calls, EVERY call verified
!             (rhoc reset to the dumped input before each), prints
!             kernel_time_s per call and a min/mean summary     (bench mode)
!
! Gate: max|out - ref| <= 1e-11 * max(1, max|ref|), the same bound as the
! SDFG lane.  addusxx_g's accumulation order is thread-count independent
! (static-schedule block ownership), so observed error is ~1e-16.
! Exits nonzero on any FAIL.
!
! DECK REPLICATION (env DECK_REP=R, default 1) grows the problem R-fold along
! the G-vector axis: qgm/mill/dfftt%nl are tiled R times, dfftt%ngm -> R*ngmt,
! dfftt%nnr -> R*nnr, rhoc/ref tiled R times, and replica k's nl entries are
! SHIFTED by +k*nnr so the scatter target sets stay disjoint (dfftt%nl is an
! injective G -> FFT-grid map; the shift is what keeps the union injective).
! Every replica must then reproduce the pw.x reference on its own rhoc slice,
! so the tiled-ref gate below is a per-replica check.
! DECK_REP_NOOFFSET=1 is the NEGATIVE CONTROL: it drops the +k*nnr shift, all
! replicas alias replica 0's slice, and verification must FAIL.
program verify_addusxx_main
  use, intrinsic :: iso_fortran_env, only: error_unit
  use omp_lib,        only: omp_get_wtime, omp_get_max_threads
  use fft_types,      only: fft_type_descriptor
  use control_flags,  only: gamma_only
  use gvect,          only: eigts1, eigts2, eigts3, gstart, mill
  use ions_base,      only: ityp, nat, tau
  use us_exx,         only: qgm, nij_type, addusxx_g
  use uspp,           only: ijtoh, nkb, ofsbeta, okvan
  use uspp_param,     only: lmaxq, nh, nhm, nsp, upf
  implicit none
  integer, parameter :: dp = 8
  real(dp), parameter :: tol = 1.0e-11_dp
  character(len=512) :: ddir
  character(len=64)  :: arg
  character(len=600) :: f
  type(fft_type_descriptor) :: dfftt
  complex(dp), allocatable :: rhoc(:), rhoc_in(:), ref(:), becphi_c(:), becpsi_c(:)
  real(dp) :: xkq(3), xk(3), maxref, maxdiff, t0, t1, tmin, tsum
  integer :: iset, nnr, ngmt, d1, d2, reps, warmup, irep, nfail
  integer :: rep, shift, nnr_rep
  logical :: ok

  if (command_argument_count() < 2) then
    write(error_unit,'(A)') 'usage: verify_addusxx <dumpdir> <set> [reps] [warmup]'
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
  write(*,'(A,I0,A,I0)') 'replicate ', rep, ' nl_offset ', shift

  ! ---------------- static module state (adxndx_static_*) ----------------
  call load_static(trim(ddir), nnr, ngmt)
  nnr_rep = rep*nnr

  ! ---------------- per-set transient state (adx_<set>_*) ----------------
  write(f,'(A,A,I0,A)') trim(ddir), '/adx_', iset, '_meta.txt'
  d1 = meta_i(trim(f), 'qgm_d1'); d2 = meta_i(trim(f), 'qgm_d2')
  if (d1 /= ngmt) then
    write(error_unit,'(A)') 'qgm_d1 /= ngmt: G-axis replication assumes they match'; error stop 1
  end if
  allocate( qgm(rep*d1, d2) )
  write(f,'(A,A,I0,A)') trim(ddir), '/adx_', iset, '_qgm.bin'
  call rd_c_rep2(trim(f), qgm, d1, d2, rep)
  write(f,'(A,A,I0,A)') trim(ddir), '/adx_', iset, '_nij_type.bin'
  call rd_i(trim(f), nij_type, nsp)
  allocate( rhoc(nnr_rep), rhoc_in(nnr_rep), ref(nnr_rep), becphi_c(nkb), becpsi_c(nkb) )
  write(f,'(A,A,I0,A)') trim(ddir), '/adx_', iset, '_rhoc_in.bin';  call rd_c_rep1(trim(f), rhoc_in, nnr, rep)
  write(f,'(A,A,I0,A)') trim(ddir), '/adx_', iset, '_rhoc_out.bin'; call rd_c_rep1(trim(f), ref, nnr, rep)
  write(f,'(A,A,I0,A)') trim(ddir), '/adx_', iset, '_xkq.bin';      call rd_r(trim(f), xkq, 3)
  write(f,'(A,A,I0,A)') trim(ddir), '/adx_', iset, '_xk.bin';       call rd_r(trim(f), xk, 3)
  write(f,'(A,A,I0,A)') trim(ddir), '/adx_', iset, '_becphi_c.bin'; call rd_c(trim(f), becphi_c, nkb)
  write(f,'(A,A,I0,A)') trim(ddir), '/adx_', iset, '_becpsi_c.bin'; call rd_c(trim(f), becpsi_c, nkb)

  nfail = 0
  do irep = 1, warmup
    rhoc = rhoc_in
    call addusxx_g( dfftt, rhoc, xkq, xk, 'c', becphi_c=becphi_c, becpsi_c=becpsi_c )
  end do
  tmin = huge(tmin); tsum = 0.0_dp
  do irep = 1, max(reps, 1)
    rhoc = rhoc_in
    t0 = omp_get_wtime()
    call addusxx_g( dfftt, rhoc, xkq, xk, 'c', becphi_c=becphi_c, becpsi_c=becpsi_c )
    t1 = omp_get_wtime()
    write(*,'(A,F12.6)') 'kernel_time_s ', t1 - t0
    tmin = min(tmin, t1 - t0); tsum = tsum + (t1 - t0)
    call check(rhoc, ref, ok)
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
      lo = ir*nnr + 1
      write(*,'(A,I0,A,ES12.4)') '  replica ', ir, ' max|diff| = ', &
           maxval(abs(out(lo:lo+nnr-1) - refv(lo:lo+nnr-1)))
    end do
    pass = (maxdiff/maxref <= tol)
  end subroutine check

  ! Static tables shared by the addusxx_g/newdxx_g decks (adxndx_static_*).
  ! Loads exactly the state the flag='c' arm reads: sizes + tvanp/nh +
  ! ityp/tau/ofsbeta/ijtoh + eigts (with their negative lbounds) + mill +
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
    nkb    = meta_i(trim(m), 'nkb')
    nat    = meta_i(trim(m), 'nat')
    nsp    = meta_i(trim(m), 'nsp')
    nhm    = meta_i(trim(m), 'nhm')
    lmaxq  = meta_i(trim(m), 'lmaxq')
    gstart = meta_i(trim(m), 'gstart')
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
      allocate( mill(3, max(ngm_full, rep*ngmt_out)) )
      call rd_i(trim(dir)//'/adxndx_static_mill.bin', mill, 3*ngm_full)
      ! Miller indices are NOT shifted: they index the shared eigts tables.
      do ir = 1, rep - 1
        mill(:, ir*ngmt_out+1:(ir+1)*ngmt_out) = mill(:, 1:ngmt_out)
      end do
    end block
    allocate( eigts1(-nr1p:nr1p, nat), eigts2(-nr2p:nr2p, nat), eigts3(-nr3p:nr3p, nat) )
    call rd_c(trim(dir)//'/adxndx_static_eigts1.bin', eigts1, (2*nr1p+1)*nat)
    call rd_c(trim(dir)//'/adxndx_static_eigts2.bin', eigts2, (2*nr2p+1)*nat)
    call rd_c(trim(dir)//'/adxndx_static_eigts3.bin', eigts3, (2*nr3p+1)*nat)
    allocate( tau(3,nat), ityp(nat), ofsbeta(nat) )
    call rd_r(trim(dir)//'/adxndx_static_tau.bin', tau, 3*nat)
    call rd_i(trim(dir)//'/adxndx_static_ityp.bin', ityp, nat)
    call rd_i(trim(dir)//'/adxndx_static_ofsbeta.bin', ofsbeta, nat)
    allocate( ijtoh(nhm,nhm,nsp) )
    call rd_i(trim(dir)//'/adxndx_static_ijtoh.bin', ijtoh, nhm*nhm*nsp)
    allocate( nij_type(nsp) )
    nij_type = 0   ! loaded per set (CUMULATIVE species offsets, qvan_init
                   ! semantics: qgm column = nij_type(nt)+ijtoh(ih,jh,nt))
    dfftt%nr1  = meta_i(trim(m), 'nr1t');  dfftt%nr2  = meta_i(trim(m), 'nr2t');  dfftt%nr3  = meta_i(trim(m), 'nr3t')
    dfftt%nr1x = dfftt%nr1; dfftt%nr2x = dfftt%nr2; dfftt%nr3x = dfftt%nr3
    dfftt%nnr  = rep*nnr_out
    dfftt%ngm  = rep*ngmt_out
    allocate( dfftt%nl(rep*ngmt_out) )
    call rd_i(trim(dir)//'/adxndx_static_dfftt_nl.bin', dfftt%nl, ngmt_out)
    do ir = 1, rep - 1
      dfftt%nl(ir*ngmt_out+1:(ir+1)*ngmt_out) = dfftt%nl(1:ngmt_out) + shift*ir*nnr_out
    end do
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

  ! Tiled readers: read the deck's single copy into replica 0's slice, then
  ! broadcast it to the remaining rep-1 slices in place (no extra buffer).
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

  subroutine rd_c_rep2(fname, a, d1, d2, r)
    character(len=*), intent(in) :: fname
    integer, intent(in) :: d1, d2, r
    complex(dp), intent(out) :: a(r*d1, d2)
    integer :: u, ios, j, ir
    open(newunit=u, file=fname, form='unformatted', access='stream', status='old', action='read', iostat=ios)
    if (ios /= 0) then
      write(error_unit,*) 'cannot open ', fname; error stop 2
    end if
    do j = 1, d2
      read(u) a(1:d1, j)
    end do
    close(u)
    do j = 1, d2
      do ir = 1, r - 1
        a(ir*d1+1:(ir+1)*d1, j) = a(1:d1, j)
      end do
    end do
  end subroutine rd_c_rep2

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

end program verify_addusxx_main
