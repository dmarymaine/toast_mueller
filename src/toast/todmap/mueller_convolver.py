# Copyright (C) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved. Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from configparser import InterpolationError
import warnings

from ..mpi import use_mpi

import numpy as np
import healpy as hp

from .. import qarray as qa

from ..op import Operator

from ..timing import function_timer, Timer

import ducc0

conviqt = None

if use_mpi:
    try:
        import libconviqt_wrapper as conviqt
    except ImportError:
        pass

#__all__ = ["MuellerConvolver"]

def nalm(lmax, mmax):
    return ( (mmax+1) * (mmax+2)) //2 + (mmax+1)*(lmax - mmax)

# Adri 2020 A25/A35
def mueller_to_C(mueller):
    T = np.zeros((4,4),dtype=np.complex128)
    T[0,0] = T[3,3] = 1.0
    T[1,1] = T[2,1] = 1.0/np.sqrt(2.0)
    T[1,2] = 1j / np.sqrt(2.0)
    T[2,2] = -1j / np.sqrt(2.0)
    C = T.dot(mueller.dot(np.conj(T.T)))
    return C

def truncate_blm(inp, lmax, kmax, epsilon=0.0):
    limit = epsilon * np.max(np.abs(inp))
    out =[]
    for i in range(len(inp)):
        maxk = -1
        idx = 0
        for k in range(kmax+1):
            if np.max(np.abs(inp[i,:,idx:idx+lmax+1-k]))> limit:
                maxk = k
            idx += lmax+1 -k
        if maxk == -1:
            out.append(None)
        else:
            out.append((inp[i,:,: nalm(lmax,maxk)].copy(),maxk))
        return out

class MuellerConvoler:

    class AlmPM:
     def __init__(self,lmax,mmax):
        if lmax < 0 or mmax < 0 or lmax < mmax:
            raise ValueError("bad parameter")
        self._lmax,self._mmax = lmax,mmax
        self._data = np.zeros((2*mmax+1,lmax+1),dtype=np.complex128)
    
     def __getitem__(self,lm):
        l,m = lm
        if isinstance(l,slice):
            if l.step is not None or l.start < 0 or l.stop -1 > self._lmax:
                print (l,m)
                raise ValueError("out of bounds read access")
        else:
            if l < 0 or l > self._lmax:
                print (l,m)
                raise ValueError("out of bounds read access")
        if m < -self._mmax or m > self._mmax:
            return 0.0 + 0j
        return self._data[m+self._mmax,l]
    
     def __setitem__(self,lm,val):
        l,m = lm
        if isinstance(l,slice):
            if l.step is not None or l.start < 0 or l.stop -1 > self._lmax:
                print (l,m)
                raise ValueError("out of bounds write access")
        else:
            if (
                l < 0
                or l> self._lmax
                or abs(m) > l
                or m < -self._mmax
                or m > self._mmax
            ):
              print(l,m)
              raise ValueError("out of bounds write access")
        self._data[m+self._mmax,l] = val

    def mueller_tc_prep(self,blm,mueller, lmax,mmax):
      """ convert input beam alm to T/P/P*/V blm """
      ncomp = blm.shape[0]
      blm2 = [self.AlmPM(lmax,mmax+4) for _ in range(4)]
      idx = 0
      for m in range(mmax+1):
          sign = (-1)**m
          lrange = slice(m,lmax+1)
          idxrange = slice(idx,idx+lmax+1-m)
          # T component
          blm2[0][lrange,m] = blm[0,idxrange]
          blm2[0][lrange,-m] = np.conj(blm[0,idxrange]) * sign
          # V component
          if ncomp > 3:
              blm2[3][lrange,m] = blm[3,idxrange]
              blm2[3][lrange,-m] = np.conj(blm[3,idxrange]) * sign
          # E/B components
          if ncomp > 2:
              # Adri's notes [10]
              blm2[1][lrange,m] = - (
                  blm[1,idxrange]+1j*blm[2,idxrange]
              )
              # Adri's notes [9]
              blm2[2][lrange,m] = - (
                  blm[1,idxrange] -1j*blm[2,idxrange]
              )
              # negative m
              # Adri's notes [2]
              blm2[1][lrange,-m] = np.conj(blm2[2][lrange,m]) * sign
              blm2[2][lrange,-m] = np.conj(blm2[1][lrange,m]) * sign
          idx += lmax+1-m

      C = mueller_to_C(mueller)

      # compute the blm for the full beam + Mueller matrix system at
      # angles n*pi/5 for n in [0;5[
      sqrt2 = np.sqrt(2.0)
      nbeam = 5
      inc = 4
      res = np.zeros((nbeam,ncomp,nalm(lmax,mmax+inc)),dtype=self._ctype)
      blm_eff = [self.AlmPM(lmax,mmax+4) for _ in range(4)]

      for ibeam in range(nbeam):
          alpha = ibeam * np.pi/ nbeam
          e2ia = np.exp(2*1j*alpha)
          e2iac = np.exp(-2*1j*alpha)
          e4ia = np.exp(4*1j*alpha)
          e4iac = np.exp(-4*1j * alpha)

          for m in range(-mmax-4,mmax+4+1):
              lrange = slice(abs(m),lmax+1)
              # T component, Marta's note [4a]
              blm_eff[0][lrange,m] = (
                 C[0,0] * blm2[0][lrange,m]
                 + C[3,0] * blm2[3][lrange,m]
                 +1.0/sqrt2
                 * (
                     C[1,0] * blm2[2][lrange,m+2]*e2ia
                     +C[2,0] * blm2[1][lrange,m-2]*e2iac
                 )
              )

              # V component, Marta's note [4d]
              blm_eff[3][lrange,m] = (
                  C[0,3] * blm2[0][lrange,m]
                  + C[3,3] * blm2[3][lrange,m]
                  +1.0/sqrt2
                  * (
                      C[1,3]*blm2[2][lrange,m+2]*e2ia
                      +C[2,3]*blm2[1][lrange,m-2]*e2iac
                  )
              )

              # E/B component, Marta's note [4b,4c]
              blm_eff[1][lrange,m] = (
                  sqrt2 * e2iac * (
                      C[0,1]*blm2[0][lrange,m+2]
                      +C[3,1]*blm2[3][lrange,m+2]
                  )
                  + C[2,1]*e4iac * blm2[2][lrange,m+4]
                  + C[1,1]*blm2[1][lrange,m]
              )
              blm_eff[2][lrange,m] = (
                  sqrt2*e2ia *(
                      C[0,2] * blm2[0][lrange,m-2]
                      +C[3,2] * blm2[3][lrange,m-2]
                  )
                  + C[1,2] * e4ia * blm2[1][lrange,m-4]
                  +C[2,2]*blm2[2][lrange,m]
              )
        
          for m in range(0,mmax+4+1):
              sign = (-1)**m
              lrange = slice(abs(m),lmax+1)
              if (np.max(
                      np.abs(
                          blm_eff[0][lrange,m] - sign*np.conj(blm_eff[0][lrange,-m])
                            )
                    ) > 1e-4
                 ):
                    raise RuntimeError("error T")
              if (np.max(np.abs(blm_eff[1][lrange,m]-sign*np.conj(blm_eff[2][lrange,-m])))>1e-4):
                  raise RuntimeError("error l2")
              if (np.max(np.abs(blm_eff[2][lrange,m]-sign*np.conj(blm_eff[1][lrange,-m])))>1e-4):
                  raise RuntimeError("error 2l")
              if (np.max(np.abs(blm_eff[3][lrange,m]-sign*np.conj(blm_eff[3][lrange,-m])))>1e-4):
                  raise RuntimeError("error V")
                    
              # back to original TEBV blm format
          idx = 0
          for m in range(mmax + inc+1):
              lrange = slice(m,lmax+1)
              idxrange = slice(idx,idx+lmax+1-m)
              # T component
              res[ibeam,0,idxrange] = blm_eff[0][lrange,m]
              # V component
              if ncomp > 3:
                  res[ibeam,3,idxrange] = blm_eff[3][lrange,m]
              # E/B components
              if ncomp > 2:
                  res[ibeam,1,idxrange] = -0.5*(
                      blm_eff[1][lrange,m]+blm_eff[2][lrange,m]
                  )
                  res[ibeam,2,idxrange] = 0.5j * (
                      blm_eff[1][lrange,m] - blm_eff[2][lrange,m]
                  )
              idx +=lmax + 1 -m
      return res

      # pseudo FFT (done with sin and cos)
    def pseudo_fft(self,inp):
      out = np.zeros((5,inp.shape[1],inp.shape[2]),dtype=self._ctype)
      out[0] = 0.2*(inp[0]+inp[1]+inp[2]+inp[3]+inp[4])
      c1,s1 = np.cos(2*np.pi/5),np.sin(2*np.pi/5)
      c2,s2 = np.cos(4*np.pi/5),np.sin(4*np.pi/5)
      out[1] = 0.4*(inp[0]+c1*(inp[1]+inp[4])+c2*(inp[2]+inp[3]))
      out[2] = 0.4*(s1*(inp[1]-inp[4])+s2*(inp[2]-inp[3]))
      out[3] = 0.4*(inp[0]+c2*(inp[1]+inp[4])+c1*(inp[2]+inp[3]))
      out[4] = 0.4*(s2*(inp[1]-inp[4])-s1*(inp[2]-inp[3]))

      return out
    
    def __init__(
        self,
        lmax,
        kmax,
        slm,
        blm,
        mueller,
        single_precision=True,
        epsilon=1e-4,
        ofactor=1.5,
        nthreads=1,
    ):

        self._ftype = np.float32 if single_precision else np.float64
        self._ctype = np.complex64 if single_precision else np.complex128
        self._slm = slm.astype(self._ctype)
        self._lmax = lmax
        self._kmax = kmax

        # now prepare the beam, Mueller matrix response
        # and create the C = TM_HWMPT^\dag

        tmp = self.mueller_tc_prep(blm, mueller, self._lmax, self._kmax) 
        tmp = self.pseudo_fft(tmp)

        # construct the five interpolator (maybe only 3 are required) for
        # the individual components. All blm are checked against their
        # beammmax
        tmp = truncate_blm(tmp, self._lmax, self._kmax+4)
        print (len(tmp),len(tmp[0]),tmp[0][0],tmp[0][1])
        self._inter = []
        intertype = (
            ducc0.totalconvolve.Interpolator_f
            if self._ctype == np.complex64
            else ducc0.totalconvolve.Interpolator
        )

        for i in range(5):
            if tmp[i] is not None:
                self._inter.append(
                    intertype(
                        self._slm,
                        tmp[i][0],
                        single_precision,
                        self._lmax,
                        tmp[i][1],
                        epsilon=epsilon,
                        ofactor=ofactor,
                        nthreads = nthreads
                    )
                )
            else:
                self._inter.append(None)
    
    def signal(self,ptg,alpha):
        ptg = ptg.astype(self._ftype)
        alpha = alpha.astype(self._ftype)
        if self._inter[0] is not None:
            res = self._inter[0].interpol(ptg)[0]
        else:
            res = np.zeros(ptg.shape[0],dtype=self._ftype)
        if self._inter[1] is not None:
            res += np.cos(2.*alpha) * self._inter[1].interpol(ptg)[0]
        if self._inter[2] is not None:
            res += np.sin(2.*alpha) * self._inter[2].interpol(ptg)[0]
        if self._inter[3] is not None:
            res += np.cos(4.*alpha) * self._inter[3].interpol(ptg)[0]
        if self._inter[4] is not None:
            res += np.sin(4.*alpha) * self._inter[4].interpol(ptg)[0]
        return res


class OpSimMueller(Operator):
    """ Operator which uses ducc to generate beam convolved timestreams.

    This passes through each observation and loops over each detector.
    For each detector, it produces the beam-convolved timestream.

    Args:
        comm (MPI.Comm): MPI communicator to use for the convolution.
        sky_file (dict or str): File containing the sky a_lm expansion.
            Tag {detector} will be replaced with detector name
            If sky_file is a dict, then each detector must have an entry.
        beam_file (dict or str): File containing the beam a_lm expansion.
            Tag {detector} will be replaced with detector name.
            Beam could be with 1,3 or 4 components according to T, TEB or
            TEBV components
        lmax (int): Maximum all (and m). Actual resolution in the Healpix FITS
            file may differ. If not set, will use the maximum expansion 
            order from file.
        pol (boolean): boolean to determine if polarization simulation is needed
        fwhm (float): width of symmetric gaussian beam [in arcmin] already
            present in the sky file (it will be deconvolved away)
        mueller (float): an array of [4,4] with Mueller matrix of the optic
            element in front of the detector
        ofactor (float): oversampling factor to be used in interpolation grid.
            In the range [1.2,2] with typical value of 1.5
        dxx (bool): The beam frame is either Dxx or Pxx. Pxx includes the
            rotation to the polarization sensitive bases, Dxx does not. When 
            Dxx = True, detector orientation from attitude quaternions is
            corrected for the polarisation angle.
        out (str): name of the cache object (<name>_<detector>) to use 
            for output of the detector timestream.    
    """
    def __init__(
      self,
      comm,
      sky_file,
      beam_file,
      lmax=0,
      beammmax=0,
      pol=True,
      fwhm=4.0,
      order=1.5,
      calibrate=True,
      dxx=True,
      out="mueller",
      quat_name=None,
      flag_name=None,
      flag_mask=255,
      common_flag_name=None,
      common_flag_mask=255,
      apply_flags=False,
      remove_monopole=False,
      remove_dipole=False,
      normalize_beam=False,
      verbosity=0,
      mc=None,
    ):

       # call the parent class constructor
       super().__init__()

       self._comm = comm
       self._sky_file = sky_file
       self._beam_file = beam_file
       self._slm = {}
       self._lmax = lmax
       self._beammmax = beammmax
       self._pol = pol
       self._fwhm = fwhm
       self._order = order
       self._calibrate = calibrate
       self._dxx = dxx
       self._quat_name = quat_name
       self._flag_name = flag_name
       self._flag_mask = flag_mask
       self._common_flag_name = common_flag_name
       self._common_flag_mask = common_flag_mask
       self._apply_flag = apply_flags
       self._remove_monopole = remove_monopole
       self._remove_dipole = remove_dipole
       self._normalize_beam = normalize_beam
       self._verbosity = verbosity
       self._mc = mc
       self._interp=[]
       self._out = out
       self._single_precision = True
       self._epsilon = 1e-4
       self._ofactor = 1.8
       self._nthreads = 1
       
       # Define Mueller matrix for an ideal HWP
       self.mueller = np.zeros((4,4),dtype=np.float32)
       self.mueller[0,0] = 1.
       self.mueller[1,1] = 1.
       self.mueller[2,2] = -1.
       self.mueller[3,3] = -1.

    @function_timer
    def exec(self, data):
        """Loop over all observations and perform the convolution.
        
        This is done one detector at a time. For each detector, all data
        products are read from disk.
        
        Args:
             data (toast.Data): the distributed data.
        
        """
        timer = Timer()
        timer.start()

        detectors = self._get_detectors(data)

        for det in detectors:
            verbose = self._comm.rank ==0 and self._verbosity > 0

            try:
                sky_file = self._sky_file[det]
            except TypeError:
                sky_file = self._sky_file.format(detector=det,mc=self._mc)
            slm = self.get_sky_hp(sky_file+".fits",det,verbose)
            try:
                beam_file = self._beam_file[det]
            except TypeError:
                beam_file = self._beam_file.format(detector=det, mc=self._mc)
            
            blm = self.get_beam_hp(beam_file+".fits",det,verbose)

            # call the Mueller convolver object
            fullconv = MuellerConvoler(
                self._lmax,self._beammmax,slm, blm,self.mueller,True,
                self._epsilon,self._ofactor, self._nthreads)
                         
            theta, phi, psi, psi_pol = self.get_pointing(data, det, verbose)
            pnt = self. get_buffer(theta, phi, psi, det, verbose)

            res = fullconv.signal(pnt,psi_pol)
#            # I-beam convolution
#            if self._inter[0] is not None:
#                res = self._inter[0].interpol(pnt)[0]
#            else:
#                res = np.zeros(pnt.shape[0], dtype=self._ftype)
#            # first 2 exp(-2ialpha) convolutions (basically Q and U for
#            # ideal HWP
#            if self._inter[1] is not None:
#                res += np.cos(2.* psi_pol) * self._inter[1].interpol(pnt)[0]
#            if self._inter[2] is not None:
#                res += np.sin(2.* psi_pol) * self._inter[2].interpol(pnt)[0]
#            
#            # second 2 exp(-4ialpha) convolutions
#            if self._inter[3] is not None:
#                res += np.cos(4.* psi_pol) * self._inter[3].interpol(pnt)[0]
#            if self._inter[4] is not None:
 #               res += np.sin(4.* psi_pol) * self._inter[4].interpol(pnt)[0]

            del theta, phi, psi
            
            self.cache(data, det, res, verbose)

            del pnt, detectors, slm , blm, tmp,

            if verbose:
                timer.report_clear("ducc process detector {}".format(det))
        return

    def _get_detectors(self,data):
        """ assemble a list of detectors across all processes and
        observations in `self._comm`.
        """
        dets = set()
        for obs in data.obs:
            tod = obs["tod"]
            for det in tod.local_dets:
                dets.add(det)
        all_dets = self._comm.gather(dets,root=0)
        if self._comm.rank ==0:
            for some_dets in all_dets:
                dets.update(some_dets)
            dets = sorted(dets)
        all_dets = self._comm.bcast(dets,root=0)
        return all_dets
    
    def _get_psi_pol(self, focalplane, det):
        """Parse polarization angle in radians from the focalplane
        dictionary. The angle is relative to the Pxx basis.
        """
        if det not in focalplane:
            raise RuntimeError("focalplane does not include {}".format(det))
        if "psi_pol_deg" in focalplane[det]:
            psi_pol = np.radians(focalplane[det]["psi_pol_deg"])
        elif "psi_pol_rad" in focalplane[det]:
            psi_pol = focalplane[det]["psi_pol_rad"]
        elif "pol_angle_deg" in focalplane[det]:
            warnings.warn(
                "use psi_pol and psi_uv rather than pol_angle",DeprecationWarning
            )
            psi_pol = np.radians(focalplane[det]["pol_angle_deg"])
        elif "pol_angle_rad" in focalplane[det]:
            warnings.warn(
                "use psi_pol and psi_uv rather than pol_angle",DeprecationWarning
            )
            pi_pol = focalplane[det]["pol_angle_rad"]
        else:
            raise RuntimeError("focalplane[{}] does not include any psi_pol".format(det))
        return psi_pol
    
    def _get_psi_uv(self, focalplane, det):
        """Parse Pxx basis angle in radians from the focalplane
        dictionary. The angle is measured from Dxx to Pxx basis.
        """
        if det not in focalplane:
            raise RuntimeError("focalplane[{}] does not include psi_uv",format(det))
        if "psi_uv_deg" in focalplane[det]:
            psi_uv = np.radians(focalplane[det]["psi_uv_deg"])
        elif "psi_uv_rad" in focalplane[det]:
            psi_uv = focalplane[det]["psi_uv_rad"]
        else:
            raise RuntimeError("focalplane[{}] does not include psi_uv".format(det))
        return psi_uv
    
    def _get_epsilon(self, focalplane, det):
        """Parse polarization leakage (epsilon) from the focalplane
        dictionary.
        """
        if det not in focalplane:
            raise RuntimeError("focalplane does not include {}".format(det))
        if "pol_leakage" in focalplane[det]:
            epsilon = focalplane[det]["pol_leakage"]
        else:
            #assume zero polarization leakage
            epsilon = 0
        return epsilon

    def get_sky_hp(self, skyfile, det, verbose):
        timer = Timer()
        timer.start()
        sky = hp.read_alm(skyfile, hdu=[1,2,3])
        ## to implement methods for removing monopole and dipole
        if verbose:
            timer.report_clear("initialize sky for detector {}".format(det))
        return sky

    def get_beam_hp(self, beamfile, det, verbose):
        timer = Timer()
        timer.start()
        beam = hp.read_alm(beamfile, hdu=[1,2,3])
        ## to implement methods for removing monopole and dipole
        if verbose:
            timer.report_clear("initialize beam for detector {}".format(det))
        return beam


    def get_sky(self, skyfile, det, verbose):
        timer = Timer()
        timer.start()
        sky = conviqt.Sky(self._lmax, self._pol, skyfile, self._fwhm, self._comm)
        if self._remove_monopole:
            sky.remove_monopole()
        if self._remove_dipole:
            sky.remove_dipole()
        if verbose:
            timer.report_clear("initialize sky for detector {}".format(det))
        return sky
    
    def get_beam(self, beamfile, det, verbose):
        timer = Timer()
        timer.start()
        beam = conviqt.Beam(self._lmax,self._beammmax, self._pol, beamfile, self._comm)
        if self._normalize_beam:
            beam.normalize()
        if verbose:
            timer.report_clear("initialize beam for detector {}".format(det))
        return beam
    
    def get_detector(self, data, det, verbose):
        """Return the detector pointin as ZYZ Euler angles without the
        polarization sensitive angle. These angles are to be compatible
        with Pxx or Dxx frame beam products
        """
        # We need the three pointing angles to describe the
        # pointing. local_pointing() return the attitude quaternions.
        nullquat = np.array([0,0,0,1],dtype=np.float64)
        timer = Timer()
        timer.start()
        all_theta,all_phi, all_psi, all_psi_pol = [], [], [], []
        for obs in data.obs:
            tod = obs["tod"]
            if det not in tod.local_dets:
                continue
            focalplane = obs["focalplane"]
            quats = tod.local_pointing(det, self._quat_name)
            if verbose:
                timer.report_clear("get detector pointing for {}".format(det))
            
            if self._apply_flag:
                common = tod.local_common_flags(self._common_flag_name)
                flags = tod.local_flags(det, self._flag_name)
                common = common & self._common_flag_mask
                flags = flags & self._flag_mask
                totflags = np.copy(flags)
                totflags |= common
                quats = quats.copy()
                quats[totflags !=0] = nullquat
                if verbose:
                    timer.report_clear("initiazlie flags for detector {}".format(det))

            theta, phi, psi = qa.to_angles(quat)
            # Polarization angle in the Pxx basis
            psi_pol = self._get_psi_pol(focalplane, det)
            if self._dxx:
                # Add angle between Dxx and Pxx
                psi_pol += self._get_psi_uv(focalplane, det)
            # Add also a potential HWP angle
            psi_pol = np.ones(psi.size) * psi_pol
            try:
                hwpang = tod.local_hwp_angle()
                psi_pol += 2* hwpang
            except:
                pass
            all_theta.append(theta)
            all_phi.append(phi)
            all_psi.append(psi)
            all_psi_pol.append(psi_pol)
            if len(all_theta) > 0:
                all_theta = np.hstack(all_theta)
                all_phi = np.hstack(all_phi)
                all_psi = np.hstack(all_psi)
                all_psi_pol = np.hstack(all_psi_pol)
            if verbose:
                timer.report_clear("compute pointing angles for detector {}".format(det))
            return all_theta, all_phi, all_psi, all_psi_pol
  
    def get_buffer(self, theta, phi, psi, det, verbose):
        """Pack the pointing into the conviqt pointing array
        This is also valid for the Mueller Convolver code"""
        timer = Timer()
        timer.start()
        pnt = conviqt.Pointing(len(theta))
        if pnt._nrow > 0:
            arr = pnt.data()
            arr[:,1] = phi
            arr[:,0] = theta
            arr[:,2] = psi
        if verbose:
            timer.report_clear("pack input array for detecotr {}".format(det))
        return pnt
    
    def cache(self, data, det, convolved_data, verbose):
        """Inject the convolved data into the TOD cache."""    
        timer = Timer()
        timer.start()
        offset = 0
        for obs in data.obs:
            tod = obs["tod"]
            if det not in tod.local_dets:
                continue
            nsample = tod.local_samples[1]
            cachename = "{}_{}".format(self._out, det)
            if not tod.cache.exists(cachename):
                tod.cache.create(cachename, np.float64, (nsample,))
            ref = tod.cache.reference(cachename)
            ref[:] += convolved_data[offset: offset+nsample]
            offset += nsample
        if verbose:
            timer.repor_clear("cache detector {}".format(det))
        return
