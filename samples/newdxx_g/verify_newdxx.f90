! Numerical verification of the newdxx_g SDFG binding (outputs/lib/
! libnewdxx_g.so) against QE-dumped ground truth (data/BaO_nat002).
!
! Per set: load the marshalled module state (static tables + per-set
! qgm/nij_type), build the dfftt descriptor, call newdxx_g_dace on the
! dumped vc_in + deexx_in, and gate against deexx_out:  max|out - ref| <= 1e-11 *
! max(1, max|ref|).  Scope: flag='c' arm (hard-baked in the SDFG), okvan=T,
! gamma_only=F, single rank.
!
!   usage: verify_newdxx [datadir]     (default data/BaO_nat002; sets 0..1)
program verify_newdxx
  use, intrinsic :: iso_fortran_env, only: error_unit
  use fft_types,       only: fft_type_descriptor
  use control_flags,   only: gamma_only
  use gvect,           only: eigts1, eigts2, eigts3, g, gg, gstart, mill, ngm
  use ions_base,       only: ityp, nat, tau
  use us_exx,          only: qgm, nij_type
  use uspp,            only: ijtoh, nkb, ofsbeta, okvan, vkb, indv
  use uspp_param,      only: lmaxq, nh, nhm, nsp, upf
  use cell_base,       only: omega
  use newdxx_g_dace_bindings, only: newdxx_g_dace, newdxx_g_dace_finalize
  implicit none
  integer, parameter :: dp = 8
  character(len=512) :: ddir
  type(fft_type_descriptor) :: dfftt
  complex(dp), allocatable :: vc(:), deexx(:), ref(:), becphi_c(:)
  real(dp), allocatable :: bec_r(:)
  real(dp) :: xkq(3), xk(3), maxref, maxdiff
  integer :: iset, nnr, ngmt, isp, nfail, d1, d2
  character(len=600) :: f

  ddir = 'data/BaO_nat002'
  if (command_argument_count() >= 1) call get_command_argument(1, ddir)

  ! ---------------- static state (adxndx_static_*) ----------------------
  call load_static(trim(ddir), nnr, ngmt)

  nfail = 0
  do iset = 0, 1
    write(f,'(A,A,I0,A)') trim(ddir), '/ndx_', iset, '_meta.txt'
    if (.not. exists(trim(f))) then
      write(*,'(A,I0,A)') 'set ', iset, ': no dump, skip'
      cycle
    end if
    ! per-set qgm / nij_type (transient module state at the dumped call)
    d1 = meta_i(trim(f), 'qgm_d1'); d2 = meta_i(trim(f), 'qgm_d2')
    if (allocated(qgm)) deallocate(qgm)
    allocate( qgm(d1, d2) )
    write(f,'(A,A,I0,A)') trim(ddir), '/ndx_', iset, '_qgm.bin'
    call rd_c(trim(f), qgm, d1*d2)
    write(f,'(A,A,I0,A)') trim(ddir), '/ndx_', iset, '_nij_type.bin'
    call rd_i(trim(f), nij_type, nsp)
    ! per-set args
    if (.not. allocated(vc)) then
      allocate( vc(nnr), deexx(nkb), ref(nkb), becphi_c(nkb), bec_r(nkb) )
      bec_r = 0.0_dp
    end if
    write(f,'(A,A,I0,A)') trim(ddir), '/ndx_', iset, '_vc_in.bin';     call rd_c(trim(f), vc, nnr)
    write(f,'(A,A,I0,A)') trim(ddir), '/ndx_', iset, '_deexx_in.bin';  call rd_c(trim(f), deexx, nkb)
    write(f,'(A,A,I0,A)') trim(ddir), '/ndx_', iset, '_deexx_out.bin'; call rd_c(trim(f), ref, nkb)
    write(f,'(A,A,I0,A)') trim(ddir), '/ndx_', iset, '_xkq.bin';       call rd_r(trim(f), xkq, 3)
    write(f,'(A,A,I0,A)') trim(ddir), '/ndx_', iset, '_xk.bin';        call rd_r(trim(f), xk, 3)
    write(f,'(A,A,I0,A)') trim(ddir), '/ndx_', iset, '_becphi_c.bin';  call rd_c(trim(f), becphi_c, nkb)
    !
    call newdxx_g_dace(dfftt, vc, xkq, xk, deexx, becphi_c=becphi_c, becphi_r=bec_r)
    !
    write(f,'(A,A,I0,A)') trim(ddir), '/ndx_', iset, '_out_sdfg.bin'
    call wr_c(trim(f), deexx, nkb)
    maxref  = max(1.0_dp, maxval(abs(ref)))
    maxdiff = maxval(abs(deexx - ref))
    write(*,'(A,I0,A,ES12.4,A,ES12.4,A,ES12.4)') 'set ', iset, ': max|diff| = ', maxdiff, &
         '  max|ref| = ', maxval(abs(ref)), '  rel = ', maxdiff/maxref
    if (maxdiff/maxref > 1.0d-11) then
      write(*,'(A,I0,A)') 'set ', iset, ': VERIFY: FAIL'
      nfail = nfail + 1
    else
      write(*,'(A,I0,A)') 'set ', iset, ': VERIFY: PASS'
    end if
  end do

  call newdxx_g_dace_finalize()
  if (nfail > 0) then
    write(*,'(A,I0,A)') 'newdxx_g: ', nfail, ' set(s) FAILED'
    error stop 1
  end if
  write(*,'(A)') 'newdxx_g: ALL SETS PASSED'

contains

  subroutine load_static(dir, nnr_out, ngmt_out)
    character(len=*), intent(in) :: dir
    integer, intent(out) :: nnr_out, ngmt_out
    character(len=600) :: m
    integer :: nr1p, nr2p, nr3p
    m = trim(dir)//'/adxndx_static_meta.txt'
    nnr_out  = meta_i(trim(m), 'nnr')
    ngmt_out = meta_i(trim(m), 'ngmt')
    nkb   = meta_i(trim(m), 'nkb')
    nat   = meta_i(trim(m), 'nat')
    nsp   = meta_i(trim(m), 'nsp')
    nhm   = meta_i(trim(m), 'nhm')
    lmaxq = meta_i(trim(m), 'lmaxq')
    ngm   = meta_i(trim(m), 'ngm')
    gstart = meta_i(trim(m), 'gstart')
    nr1p = meta_i(trim(m), 'nr1p'); nr2p = meta_i(trim(m), 'nr2p'); nr3p = meta_i(trim(m), 'nr3p')
    omega = meta_r(trim(m), 'omega')
    okvan = .true.;  gamma_only = .false.
    allocate( nh(nsp), upf(nsp) )
    block
      integer :: i
      character(len=32) :: k
      do i = 1, nsp
        write(k,'(A,I0)') 'nh_', i;    nh(i) = meta_i(trim(m), trim(k))
        write(k,'(A,I0)') 'tvanp_', i; upf(i)%tvanp = meta_i(trim(m), trim(k)) /= 0
        write(k,'(A,I0)') 'tpawp_', i; upf(i)%tpawp = meta_i(trim(m), trim(k)) /= 0
        write(k,'(A,I0)') 'nbeta_', i; upf(i)%nbeta = meta_i(trim(m), trim(k))
      end do
    end block
    allocate( mill(3,ngm), g(3,ngm), gg(ngm) )
    call rd_i(trim(dir)//'/adxndx_static_mill.bin', mill, 3*ngm)
    call rd_r(trim(dir)//'/adxndx_static_g.bin',    g,    3*ngm)
    call rd_r(trim(dir)//'/adxndx_static_gg.bin',   gg,   ngm)
    allocate( eigts1(-nr1p:nr1p, nat), eigts2(-nr2p:nr2p, nat), eigts3(-nr3p:nr3p, nat) )
    call rd_c(trim(dir)//'/adxndx_static_eigts1.bin', eigts1, (2*nr1p+1)*nat)
    call rd_c(trim(dir)//'/adxndx_static_eigts2.bin', eigts2, (2*nr2p+1)*nat)
    call rd_c(trim(dir)//'/adxndx_static_eigts3.bin', eigts3, (2*nr3p+1)*nat)
    allocate( tau(3,nat), ityp(nat), ofsbeta(nat) )
    call rd_r(trim(dir)//'/adxndx_static_tau.bin', tau, 3*nat)
    call rd_i(trim(dir)//'/adxndx_static_ityp.bin', ityp, nat)
    call rd_i(trim(dir)//'/adxndx_static_ofsbeta.bin', ofsbeta, nat)
    allocate( ijtoh(nhm,nhm,nsp), indv(nhm,nsp) )
    call rd_i(trim(dir)//'/adxndx_static_ijtoh.bin', ijtoh, nhm*nhm*nsp)
    call rd_i(trim(dir)//'/adxndx_static_indv.bin',  indv,  nhm*nsp)
    allocate( nij_type(nsp) )
    nij_type = 0   ! loaded per set from the dump (CUMULATIVE species offsets,
                   ! qvan_init semantics: qgm column = nij_type(nt)+ijtoh(ih,jh,nt))
    block
      integer :: npwx
      npwx = meta_i(trim(m), 'npwx')
      allocate( vkb(npwx, nkb) )
      call rd_c(trim(dir)//'/adxndx_static_vkb.bin', vkb, npwx*nkb)
    end block
    ! dfftt descriptor: the fields the wrapper marshals
    dfftt%nr1  = meta_i(trim(m), 'nr1t');  dfftt%nr2  = meta_i(trim(m), 'nr2t');  dfftt%nr3  = meta_i(trim(m), 'nr3t')
    dfftt%nr1x = dfftt%nr1; dfftt%nr2x = dfftt%nr2; dfftt%nr3x = dfftt%nr3
    dfftt%nnr  = nnr_out
    dfftt%ngm  = ngmt_out
    allocate( dfftt%nl(ngmt_out) )
    call rd_i(trim(dir)//'/adxndx_static_dfftt_nl.bin', dfftt%nl, ngmt_out)
  end subroutine load_static

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

  logical function exists(fname)
    character(len=*), intent(in) :: fname
    inquire(file=fname, exist=exists)
  end function exists

  subroutine wr_c(fname, a, n)
    character(len=*), intent(in) :: fname
    integer, intent(in) :: n
    complex(dp), intent(in) :: a(*)
    integer :: u
    open(newunit=u, file=fname, form='unformatted', access='stream', status='replace')
    write(u) a(1:n)
    close(u)
  end subroutine wr_c

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

end program verify_newdxx
