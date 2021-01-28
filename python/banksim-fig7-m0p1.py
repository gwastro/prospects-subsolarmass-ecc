import pycbc
from pycbc.waveform import get_fd_waveform, get_td_waveform
from pycbc.filter import match
from pycbc.types import FrequencySeries
from pycbc import distributions
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.filter import match, sigmasq
import pycbc.conversions as conversions

import time
import h5py
import numpy as np
import sys

bankfile = np.loadtxt(
    '/work/yifan.wang/2-subsolar/1-gen-bank/20-1024/params.txt', delimiter=',')
eccfile = np.loadtxt('mass0p1ecc.txt')
cumsum = np.cumsum(eccfile[:, 1])


class bankstudy(object):
    def __init__(self, bankfile, f_lower=20,
                 f_upper=1024, delta_f=0.0001, **args):
        self.seed = int(sys.argv[1])
        np.random.seed(self.seed)
        self.bank_m1 = bankfile[:, 0]
        self.bank_m2 = bankfile[:, 1]
        self.f_lower = f_lower
        self.f_upper = f_upper
        self.delta_f = delta_f
        self.bank_tau0 = conversions.tau0_from_mass1_mass2(
            self.bank_m1, self.bank_m2, f_lower=self.f_lower)
        self.bank_size = len(self.bank_m1)
        self.inj_num = int(args['inj_num'])
        self.injection = distributions.uniform.Uniform(
            pol=(0, 2 * np.pi), inc=(-np.pi / 2, np.pi / 2)).rvs(size=self.inj_num)
        self.eccrand = np.random.uniform(0, cumsum[-1], self.inj_num)
        self.injecc = eccfile[np.digitize(self.eccrand, cumsum), 0]
        self.psd = aLIGOZeroDetHighPower(
            int(self.f_upper / self.delta_f) + 1, self.delta_f, self.f_lower)

    def match(self):
        match_result = []
        injtau0 = conversions.tau0_from_mass1_mass2(
            0.1, 0.1, f_lower=self.f_lower)
        for i in range(self.inj_num):
            print('i is %s' % i)
            hpinj, _ = get_fd_waveform(approximant="TaylorF2e",
                                       mass1=0.1,
                                       mass2=0.1,
                                       eccentricity=self.injecc[i],
                                       long_asc_nodes=self.injection['pol'][i],
                                       inclination=self.injection['inc'][i],
                                       delta_f=self.delta_f, f_lower=self.f_lower, f_final=self.f_upper)
           # scan the template bank to find the maximum match
            index = np.where(np.abs(injtau0 - self.bank_tau0) < 3)
            max_match, max_m1, max_m2 = None, None, None
            for k in index[0]:
                hpbank, __ = get_fd_waveform(approximant="TaylorF2",
                                             mass1=self.bank_m1[k],
                                             mass2=self.bank_m2[k],
                                             phase_order=6,
                                             delta_f=self.delta_f, f_lower=self.f_lower, f_final=self.f_upper)
                cache_match, _ = match(hpinj,
                                       hpbank,
                                       psd=self.psd,
                                       low_frequency_cutoff=self.f_lower,
                                       high_frequency_cutoff=self.f_upper)
                print('ecc=%f,m1=%f,m2=%f,match=%f' % (self.injecc[i],self.bank_m1[k],self.bank_m2[k],cache_match))
                if max_match == None:
                    max_match = cache_match
                    max_m1 = self.bank_m1[k]
                    max_m2 = self.bank_m2[k]
                elif cache_match > max_match:
                    max_match = cache_match
                    max_m1 = self.bank_m1[k]
                    max_m2 = self.bank_m2[k]
            match_result.append([0.1, 0.1, self.injecc[i], self.injection['pol']
                                 [i], self.injection['inc'][i], max_match, max_m1, max_m2])

        np.savetxt('result'+str(sys.argv[1])+'.txt',
                   match_result, fmt='%f', header='injm1  injm2  injecc  injpol injinc maxmatch  maxm1  maxm2')
        return match_result


time_start = time.time()
study = bankstudy(bankfile, inj_num=2)
study.match()
