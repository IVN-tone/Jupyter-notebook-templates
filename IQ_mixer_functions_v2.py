import numpy as np
import time
import visa

from libs.qdevices import ADC_ZI
from libs.qcircuit import Readout

from itertools import zip_longest
from tqdm.notebook import tqdm

def clip_dc(x):
    x = [np.real(x), np.imag(x)]
    for c in (0,1):
        if x[c] < -0.5:
            x[c] = -0.5
        if x[c] > 0.5:
            x[c] = 0.5
    x = x[0] + 1j * x[1]
    return x
    
    
class Mixer_calibration_DC:
    def __init__(self, awg_name, awg, sa, qubit_channels):
        """
            awg: hdawg (ZI), uhfqa (ZI)
        """
        self.awg = awg
        self.awg_name = awg_name
        self.sa = sa
        self.qubit_channels = qubit_channels       
    
    def configure_devices_q(self, center_freq):
        LO = self.qubit_channels['LO']
        
        # groupping: 0 - 4x2, 1 - 2x4, 2 - 1x8, -1 - MDS
        self.awg.setInt(f'/{self.awg_name}/system/awg/channelgrouping',2)
        # setting range for all the channels
        self.awg.setDouble(f'/{self.awg_name}/sigouts/0/range', 1e0)
        self.awg.setDouble(f'/{self.awg_name}/sigouts/1/range', 1e0)
        self.awg.setDouble(f"/{self.awg_name}/sigouts/{self.qubit_channels['ch_I']}/range", self.qubit_channels['range_q'])
        self.awg.setDouble(f"/{self.awg_name}/sigouts/{self.qubit_channels['ch_Q']}/range", self.qubit_channels['range_q'])
        
        # SA and LO configuration, HARDCODDED settings
        res_bw = 4e6
        video_bw = 2e2
        trace = 1
        sweep_time = 50e-3
        span = 0
        nop = 1
        LO.set_frequency(center_freq)
        LO.set_power(16)
        LO.set_status(1)
        
        self.sa.set_res_bw(res_bw)
        self.sa.set_video_bw(video_bw)
        self.sa.set_tr_avg(trace)
        self.sa.set_center_frequency(center_freq)
        self.sa.set_sweep_time(sweep_time)
        self.sa.set_span(span)
        self.sa.set_nop(nop)
        time.sleep(0.05)
        
        # enable wave output
        self.awg.setInt(f"/{self.awg_name}/sigouts/{self.qubit_channels['ch_I']}/on",1)
        self.awg.setInt(f"/{self.awg_name}/sigouts/{self.qubit_channels['ch_Q']}/on",1)
        
        print('SA, LO and HDAWG configured for DC calibration!')

    def configure_devices_res(self, center_freq):
        LO = self.qubit_channels['LO_res']
        
        # groupping: 0 - 4x2, 1 - 2x4, 2 - 1x8, -1 - MDS
        self.awg.setInt(f'/{self.awg_name}/system/awg/channelgrouping',2)
        # setting range for all the channels
        self.awg.setDouble(f"/{self.awg_name}/sigouts/{self.qubit_channels['ch_I_ro']}/range", 1e0)
        self.awg.setDouble(f"{self.awg_name}/sigouts/{self.qubit_channels['ch_Q_ro']}/range", 1e0)
        
        # SA and LO configuration, HARDCODDED settings
        res_bw = 4e6
        video_bw = 2e2
        trace = 1
        sweep_time = 50e-3
        span = 0
        nop = 1
        # LO.set_power_on()
        LO.set_power(16)
        LO.set_frequency(center_freq)
        LO.set_sweep_mode("CW")
        LO.set_bandwidth(100)
        LO.set_nop(2001)
        
        self.sa.set_res_bw(res_bw)
        self.sa.set_video_bw(video_bw)
        self.sa.set_tr_avg(trace)
        self.sa.set_center_frequency(center_freq)
        self.sa.set_sweep_time(sweep_time)
        self.sa.set_span(span)
        self.sa.set_nop(nop)
        time.sleep(0.05)
        
        # enable wave output
        self.awg.setInt(f"/{self.awg_name}/sigouts/{self.qubit_channels['ch_I_ro']}/on",1)
        self.awg.setInt(f"/{self.awg_name}/sigouts/{self.qubit_channels['ch_Q_ro']}/on",1)
        
        print('SA, LO and HDAWG configured for DC calibration!')
        
    def LO_minimize_q(self, x):
        dc = clip_dc(x[0]+x[1]*1j)
        self.awg.setDouble(f'/{self.awg_name}/sigouts/'+str(self.qubit_channels['ch_I'])+'/offset', np.real(dc))
        dc -= np.real(dc)
        self.awg.setDouble(f'/{self.awg_name}/sigouts/'+str(self.qubit_channels['ch_Q'])+'/offset', np.imag(dc))
        dc -= 1j*np.imag(dc)

        time.sleep(0.05)

        self.sa.set_common_commands()
        self.sa.set_data_format()
        result = self.sa.get_trace_data()[1]
        time.sleep(0.5)
        print(f"\r solution: {x}, LO level: {result[0]} dBm", end="   ")
        #print(x, result) # for debug
        return result
        
    def LO_minimize_res(self, x):   
        dc = clip_dc(x[0]+x[1]*1j)
        self.awg.setDouble(f"/{self.awg_name}/sigouts/{self.qubit_channels['ch_I_ro']}/offset", np.real(dc))
        dc -= np.real(dc)
        self.awg.setDouble(f"/{self.awg_name}/sigouts/{self.qubit_channels['ch_Q_ro']}/offset", np.imag(dc))
        dc -= 1j*np.imag(dc)

        time.sleep(0.05)

        self.sa.set_common_commands()
        self.sa.set_data_format()
        result = self.sa.get_trace_data()[1]
        print(x, result)
        return result

    def switch_off_q(self):
        # disable wave output
        self.awg.setInt(f"/{self.awg_name}/sigouts/{self.qubit_channels['ch_I']}/on",0)
        self.awg.setInt(f"/{self.awg_name}/sigouts/{self.qubit_channels['ch_Q']}/on",0)
        
    def switch_off_res(self):
        # disable wave output
        self.awg.setInt(f"/{self.awg_name}/sigouts/{self.qubit_channels['ch_I_ro']}/on",0)
        self.awg.setInt(f"/{self.awg_name}/sigouts/{self.qubit_channels['ch_Q_ro']}/on",0)
        
class Mixer_calibration_RF:
    def __init__(self, awg, awg_name, sa, qubit_channels, solution_DC, if_q=None, if_res=None):
        self.awg = awg
        self.awg_name = awg_name
        self.sa = sa
        self.qubit_channels = qubit_channels
        self.solution_DC = solution_DC
        self.if_q = if_q
        self.if_res = if_res
        if self.if_q==None or self.if_res==None:
            raise Exception("Intermediate frequency is not entered! Please, enter if_q and if_res")

#calibration with sine generators
    def SB_minimize_q(self, x):

        num_sidebands = 3
        sign = 1 if self.if_q > 0 else -1
        target_sideband_id = sign
        
        sideband_ids = np.asarray(np.linspace(-(num_sidebands-1)/2, (num_sidebands - 1)/2, num_sidebands), dtype = int)
        bad_sidebands = np.logical_and(sideband_ids != target_sideband_id, sideband_ids != 0) #for sign = -1, output: False, False, True
        
        dc = clip_dc(self.solution_DC[0]+self.solution_DC[1]*1j)
        
        I = 0.4
        Q_raw = x[0] + x[1]*1j
        ampl_Q = (x[0]**2+x[1]**2)**0.5
        phase_Q = np.arctan2(Q_raw.imag,Q_raw.real) # ?np.angle
        
        if self.if_q<0:
    #         phase_Q = (-1)*phase_Q # 45-09
            phase_Q = phase_Q
        
        Q_new = ampl_Q*np.exp(1j*(phase_Q+0.5*np.pi))
    #     Q_new = ampl_Q*np.exp(1j*(-phase_Q+0.5*np.pi)) # 03-07
        
        Q = Q_new.real + Q_new.imag*1j

        self.awg.setDouble(f'/{self.awg_name}/sigouts/'+str(self.qubit_channels['ch_I'])+'/offset', np.real(dc))
        self.awg.setDouble(f'/{self.awg_name}/sigouts/'+str(self.qubit_channels['ch_Q'])+'/offset', np.imag(dc))
        self.awg.setDouble(f'/{self.awg_name}/sines/'+str(self.qubit_channels['ch_Q'])+'/phaseshift',
                        np.degrees(np.arctan2(Q_new.imag,Q_new.real)))
        self.awg.setDouble(f'/{self.awg_name}/sines/'+str(self.qubit_channels['ch_Q'])+'/amplitudes/1', abs(Q))
        
        time.sleep(0.2) 
        
        max_amplitude = np.max([np.max(np.abs(0.4)), np.max(np.abs(Q))])
        if max_amplitude < 1:
            clipping = 0
        else:
            clipping = max_amplitude - 1
        
        result = []
        self.sa.set_common_commands()
               
        left_SB = self.qubit_channels['LO'].get_frequency()+(0-(num_sidebands-1)/2)*np.abs(self.if_q)
        LO = self.qubit_channels['LO'].get_frequency()+(1-(num_sidebands-1)/2)*np.abs(self.if_q)
        right_SB = self.qubit_channels['LO'].get_frequency()+(2-(num_sidebands-1)/2)*np.abs(self.if_q)
        
        print([left_SB, LO, right_SB])
        for sideband_id in range(num_sidebands):
            self.sa.set_center_frequency(self.qubit_channels['LO'].get_frequency()+(sideband_id-(num_sidebands-1)/2)\
            *np.abs(self.if_q))
            
    #         print(sa.get_center_frequency())
            # OLD DATA READ
            result.append(np.log10(np.sum(10**(self.sa.get_trace_data()[1]/10)))*10)
            # new data reading
            #rm_sa = visa.ResourceManager().open_resource('TCPIP0::localhost::hislip4::INSTR')
            #result.append(np.log10(np.sum(10**(rm_sa.query_ascii_values("CALCulate:DATA?")[1]/10)))*10)
            # result.append(sa.get_trace_data()[1][0])
        result = np.asarray(result)
        
        bad_power_mW = np.sum(10**(result[bad_sidebands]/20))
        # print(bad_sidebands)
        good_power_mW = np.sum(10**(result[sideband_ids == target_sideband_id]/20))
    #     print(sideband_ids)
        # print('target: ',target_sideband_id)
        bad_power_dBm = np.log10(bad_power_mW)*20
        good_power_dBm = np.log10(good_power_mW)*20
        
        print('dc: {0: 4.2e}\tI: {1: 4.2e}\tQ:{2: 4.2e}\tB: {3:4.2f} G: {4:4.2f}, C:{5:4.2f}\r'.format(self.solution_DC[0]+self.solution_DC[1]*1j, I, Q, bad_power_dBm, good_power_dBm, clipping))
        
        #print('\tB: {3:4.2f} G: {4:4.2f}, C:{5:4.2f}\r'.format(bad_power_dBm, good_power_dBm, clipping))
        
        print(result)
        return -good_power_mW/bad_power_mW + np.abs(good_power_mW/bad_power_mW)*10*clipping
        
    #calibration with sine generators
    def SB_minimize_res(self, x, awg=None):

        num_sidebands = 3
        sign = 1 if self.if_res > 0 else -1
        target_sideband_id = sign
        
        sideband_ids = np.asarray(np.linspace(-(num_sidebands-1)/2, (num_sidebands - 1)/2, num_sidebands), dtype = int)
        bad_sidebands = np.logical_and(sideband_ids != target_sideband_id, sideband_ids != 0) #for sign = -1, output: False, False, True
        
        dc = clip_dc(self.solution_DC[0]+self.solution_DC[1]*1j)
        
        I = 0.4
        Q_raw = x[0] + x[1]*1j
        ampl_Q = (x[0]**2+x[1]**2)**0.5
        phase_Q = np.arctan2(Q_raw.imag,Q_raw.real)
        
        if self.if_res<0:
            phase_Q = (-1)*phase_Q
        
        Q_new = ampl_Q*np.exp(1j*(phase_Q+0.5*np.pi))
        Q = Q_new.real + Q_new.imag*1j
        
        self.awg.setDouble(f'/{self.awg_name}/sigouts/0/offset', np.real(dc))
        self.awg.setDouble(f'/{self.awg_name}/sigouts/1/offset', np.imag(dc))
        
        if not awg == 'uhfqa':
            self.awg.setDouble(f'/{self.awg_name}/sines/1/phaseshift', np.degrees(np.arctan2(Q_new.imag,Q_new.real)))
            self.awg.setDouble(f'/{self.awg_name}/sines/1/amplitudes/1', abs(Q))
        
        else:
            # UHFQA AWG settings
            pulse_len = 3600
            pulse_duration = pulse_len/1.8e9
            N_cycles = abs(self.if_res)*pulse_duration
            phase = np.arctan2(Q_new.imag,Q_new.real)
            amp_q = abs(Q)
            
            # empty container
            mixer_cals_temp = {}
            mixer_cals_temp['DC_res'] = (0,0)
            mixer_cals_temp['RF_res'] = (0,0)
            
            ro1 = Readout(self.qubit_channels, mixer_cals_temp, pulse_duration)
            awg_ro = ADC_ZI(self.awg)
            
            pWaves = ""
            for i in range(10):
                pWaves += "playWave(1, I_cal, 2, Q_cal);\nwaitWave();\n"
            
            program = f
            """
            const phase = """+str(phase)+""";
            const amp_q = """+str(amp_q)+""";
            const pulse_len = """+str(pulse_len)+""";
            const N_cycles = """+str(N_cycles)+""";
            
            //Waveforms definition
            wave I_cal = sine(pulse_len, 0.4, 0, N_cycles);
            wave Q_cal = sine(pulse_len, amp_q, phase, N_cycles);
            
            repeat(1000)
            {
                """+pWaves+"""
            }

            """
            
            awg_ro.compile_and_upload_seq_program(program)
            awg_ro.enable_awg_output(0)
            awg_ro.enable_awg_output(1)
            awg_ro.enable_awg_repeat_mode()
            awg_ro.enable_sequencer()

        time.sleep(0.2)
        
        max_amplitude = np.max([np.max(np.abs(0.4)), np.max(np.abs(Q))])
        if max_amplitude < 1:
            clipping = 0
        else:
            clipping = max_amplitude - 1
        
        result = []
        self.sa.set_common_commands()
        for sideband_id in range(num_sidebands):
            self.sa.set_center_frequency(self.qubit_channels['LO_res'].get_frequency()+(sideband_id-(num_sidebands-1)/2)\
            *np.abs(self.if_res))
            result.append(np.log10(np.sum(10**(self.sa.get_trace_data()[1]/10)))*10)
            #result.append(sa.get_trace_data()[1][0])
        result = np.asarray(result)
        
        bad_power_mW = np.sum(10**(result[bad_sidebands]/20))
        #print(bad_sidebands)
        good_power_mW = np.sum(10**(result[sideband_ids == target_sideband_id]/20))
        #print(sideband_ids)
        #print(target_sideband_id)
        bad_power_dBm = np.log10(bad_power_mW)*20
        good_power_dBm = np.log10(good_power_mW)*20
        
        print('dc: {0: 4.2e}\tI: {1: 4.2e}\tQ:{2: 4.2e}\tB: {3:4.2f} G: {4:4.2f}, C:{5:4.2f}\r'.format(self.solution_DC[0]+self.solution_DC[1]*1j, I, Q, bad_power_dBm, good_power_dBm, clipping))
        print(result)
        return -good_power_mW/bad_power_mW + np.abs(good_power_mW/bad_power_mW)*10*clipping

#calibration with sine generators
    def SB_minimize_q_sign(self, x):

        num_sidebands = 3
        sign = -1 if self.if_q > 0 else 1
        target_sideband_id = sign
        
        sideband_ids = np.asarray(np.linspace(-(num_sidebands-1)/2, (num_sidebands - 1)/2, num_sidebands), dtype = int)
        bad_sidebands = np.logical_and(sideband_ids != target_sideband_id, sideband_ids != 0) #for sign = -1, output: False, False, True
        
        dc = clip_dc(self.solution_DC[0]+self.solution_DC[1]*1j)
        
        I = 0.4
        Q_raw = x[0] + x[1]*1j
        ampl_Q = (x[0]**2+x[1]**2)**0.5
        phase_Q = np.arctan2(Q_raw.imag,Q_raw.real) # ?np.angle
        
        if self.if_q<0:
    #         phase_Q = (-1)*phase_Q # 45-09
            phase_Q = phase_Q
        
        Q_new = ampl_Q*np.exp(1j*(phase_Q+0.5*np.pi))
    #     Q_new = ampl_Q*np.exp(1j*(-phase_Q+0.5*np.pi)) # 03-07
        
        Q = Q_new.real + Q_new.imag*1j

        self.awg.setDouble(f'/{self.awg_name}/sigouts/'+str(self.qubit_channels['ch_I'])+'/offset', np.real(dc))
        self.awg.setDouble(f'/{self.awg_name}/sigouts/'+str(self.qubit_channels['ch_Q'])+'/offset', np.imag(dc))
        self.awg.setDouble(f'/{self.awg_name}/sines/'+str(self.qubit_channels['ch_Q'])+'/phaseshift',
                        np.degrees(np.arctan2(Q_new.imag,Q_new.real)))
        self.awg.setDouble(f'/{self.awg_name}/sines/'+str(self.qubit_channels['ch_Q'])+'/amplitudes/1', abs(Q))
        
        time.sleep(0.2)
        
        max_amplitude = np.max([np.max(np.abs(0.4)), np.max(np.abs(Q))])
        if max_amplitude < 1:
            clipping = 0
        else:
            clipping = max_amplitude - 1
        
        result = []
        self.sa.set_common_commands()
        for sideband_id in range(num_sidebands):
            self.sa.set_center_frequency(self.qubit_channels['LO'].get_frequency()+(sideband_id-(num_sidebands-1)/2)\
            *np.abs(self.if_q))
    #         print(sa.get_center_frequency())
            # OLD DATA READ
            result.append(np.log10(np.sum(10**(self.sa.get_trace_data()[1]/10)))*10)
            # new data reading
    #         result.append(np.log10(np.sum(10**(rm_sa.query_ascii_values("CALCulate:DATA?")[1]/10)))*10)
    #         result.append(sa.get_trace_data()[1][0])
        result = np.asarray(result)
        
        bad_power_mW = np.sum(10**(result[bad_sidebands]/20))
        # print(bad_sidebands)
        good_power_mW = np.sum(10**(result[sideband_ids == target_sideband_id]/20))
    #     print(sideband_ids)
        # print('target: ',target_sideband_id)
        bad_power_dBm = np.log10(bad_power_mW)*20
        good_power_dBm = np.log10(good_power_mW)*20
        
        print('dc: {0: 4.2e}\tI: {1: 4.2e}\tQ:{2: 4.2e}\tB: {3:4.2f} G: {4:4.2f}, C:{5:4.2f}\r'.format(self.solution_DC[0]+self.solution_DC[1]*1j, I, Q, bad_power_dBm, good_power_dBm, clipping))
        print(result)
        return -good_power_mW/bad_power_mW + np.abs(good_power_mW/bad_power_mW)*10*clipping

    def configure_devices_q(self, center_freq, if_q):
        LO = self.qubit_channels['LO']
        
        # setting amplitude, values HARDCODDED
        self.awg.setDouble(f"/{self.awg_name}/awgs/{self.qubit_channels['awg_num']}/outputs/0/amplitude", 0.3)
        self.awg.setDouble(f"/{self.awg_name}/awgs/{self.qubit_channels['awg_num']}/outputs/1/amplitude", 0.3)

        # enable sine generators and channels
        self.awg.setInt(f"/{self.awg_name}/sines/{self.qubit_channels['ch_I']}/enables/0", 1)
        self.awg.setInt(f"/{self.awg_name}/sines/{self.qubit_channels['ch_Q']}/enables/1", 1)
        self.awg.setInt(f"/{self.awg_name}/sigouts/{self.qubit_channels['ch_I']}/on", 1)
        self.awg.setInt(f"/{self.awg_name}/sigouts/{self.qubit_channels['ch_Q']}/on", 1)
        
        # setting starting amplitudes for I- and Q-channels
        self.awg.setDouble(f"/{self.awg_name}/sines/{self.qubit_channels['ch_I']}/amplitudes/0",0.4)
        self.awg.setDouble(f"/{self.awg_name}/sines/{self.qubit_channels['ch_Q']}/amplitudes/1",0.2)
        # set oscillator
        self.awg.setDouble(f"/{self.awg_name}/oscs/{self.qubit_channels['osc_num']}/freq", abs(if_q))
        
        # SA and LO configuration
        res_bw = 1e6
        video_bw = 2e2
        trace = 1
        sweep_time = 50e-3
        nop = 1
        
        self.sa.set_tr_avg(trace)
        self.sa.set_res_bw(res_bw)
        self.sa.set_video_bw(video_bw)
        self.sa.set_span(0)
        self.sa.set_sweep_time(sweep_time)
        self.sa.set_nop(nop)
        LO.set_status(1)

        # set initial center frequency for LO and IF frequency
        self.sa.set_center_frequency(center_freq)
        time.sleep(0.1)
        
        print('SA, LO and HDAWG configured for RF calibration!')
        
    def configure_devices_res(self, center_freq, if_res):
        # HDAWG channels HARDCODDED!
        LO = self.qubit_channels['LO_res']
        
        # setting amplitude, values HARDCODDED
        self.awg.setDouble(f"/{self.awg_name}/awgs/{0}/outputs/0/amplitude", 0.4)
        self.awg.setDouble(f"/{self.awg_name}/awgs/{0}/outputs/1/amplitude", 0.4)

        # enable sine generators and channels
        self.awg.setInt(f"/{self.awg_name}/sines/{0}/enables/0", 1)
        self.awg.setInt(f"/{self.awg_name}/sines/{1}/enables/1", 1)
        self.awg.setInt(f"/{self.awg_name}/sigouts/{0}/on", 1)
        self.awg.setInt(f"/{self.awg_name}/sigouts/{1}/on", 1)
        
        # setting starting amplitudes for I- and Q-channels
        self.awg.setDouble(f"/{self.awg_name}/sines/{0}/amplitudes/0",0.4)
        self.awg.setDouble(f"/{self.awg_name}/sines/{1}/amplitudes/1",0.2)
        # set oscillator
        self.awg.setDouble(f"/{self.awg_name}/oscs/{0}/freq", abs(if_res))
        
        # SA and LO configuration
        res_bw = 1e6
        video_bw = 2e2
        trace = 1
        sweep_time = 50e-3
        nop = 1
        
        self.sa.set_tr_avg(trace)
        self.sa.set_res_bw(res_bw)
        self.sa.set_video_bw(video_bw)
        self.sa.set_span(0)
        self.sa.set_sweep_time(sweep_time)
        self.sa.set_nop(nop)
        # LO.set_power_on()
        LO.set_power(16)
        LO.set_sweep_mode("CW")

        # set initial center frequency for LO and IF frequency
        self.sa.set_center_frequency(center_freq)
        time.sleep(0.1)
        
        print('SA, LO and HDAWG configured for RF calibration!')
        
    def switch_off_q(self):
        # disable sine generators and wave outputs
        self.awg.setInt(f"/{self.awg_name}/sines/{self.qubit_channels['ch_I']}/enables/0", 0)
        self.awg.setInt(f"/{self.awg_name}/sines/{self.qubit_channels['ch_Q']}/enables/1", 0)
        self.awg.setInt(f"/{self.awg_name}/sigouts/{self.qubit_channels['ch_I']}/on", 0)
        self.awg.setInt(f"/{self.awg_name}/sigouts/{self.qubit_channels['ch_Q']}/on", 0)
        
    def switch_off_res(self):
        # HDAWG channels HARDCODDED
        # disable sine generators and wave outputs
        self.awg.setInt(f"/{self.awg_name}/sines/{0}/enables/0", 0)
        self.awg.setInt(f"/{self.awg_name}/sines/{1}/enables/1", 0)
        self.awg.setInt(f"/{self.awg_name}/sigouts/{0}/on", 0)
        self.awg.setInt(f"/{self.awg_name}/sigouts/{1}/on", 0)

        